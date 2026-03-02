import { mapWithConcurrency } from "./concurrency";
import {
  buildOpenAiChatCompletionBody,
  parseOpenAiContent,
  runChatCompletion,
  type ChatMessage,
  type NativeToolDefinition,
  type OpenAiChatCompletionBody,
} from "./backend";
import { extractPdfCandidates, iteratePdfPageImages } from "./pdf";
import { DEFAULT_PROMPT_CONFIG } from "./prompts";
import { withRetries } from "./retry";
import { sanitizeQualitySettings } from "./settings";
import type {
  BackendConfig,
  ChunkDecision,
  ExtractionQualitySettings,
  ExtractionRow,
  KeepDecision,
  ParagraphCandidate,
  PromptConfig,
  RunProgress,
} from "../types";

interface ExtractionCallbacks {
  onLog?: (level: "info" | "warning" | "error", message: string) => void;
  onProgress?: (progress: RunProgress) => void;
}

interface VisionParagraphDecision {
  text: string;
  section_heading?: string;
  note?: string;
  confidence?: number;
  possible_boilerplate?: boolean;
}

interface VisionPageDecision {
  paragraphs: VisionParagraphDecision[];
  warnings: string[];
}

const OCR_LIMIT_WARNING_PATTERN =
  /(low[- ]resolution|higher[- ]resolution|too small|unable to reliably|cannot reliably|can't reliably|not reliable|insufficient detail)/i;

export interface RunExtractionOptions extends ExtractionCallbacks {
  config: BackendConfig;
  prompts?: PromptConfig;
  quality?: ExtractionQualitySettings;
  fileConcurrency: number;
  retries: number;
  signal?: AbortSignal;
}

export interface ExtractionBatchFilePlan {
  pdfName: string;
  fileIndex: number;
  mode: "text" | "vision";
  totalPages: number;
}

interface ExtractionBatchTaskBase {
  customId: string;
  pdfName: string;
  fileIndex: number;
  taskIndex: number;
}

export interface TextFilterBatchTask extends ExtractionBatchTaskBase {
  kind: "text_filter";
  chunkIndex: number;
  totalChunks: number;
  chunk: ParagraphCandidate[];
}

export interface VisionPageBatchTask extends ExtractionBatchTaskBase {
  kind: "vision_page";
  pageNumber: number;
  totalPages: number;
}

export type ExtractionBatchTask = TextFilterBatchTask | VisionPageBatchTask;

export interface OpenAiExtractionBatchManifest {
  version: 1;
  createdAt: string;
  model: string;
  quality: ExtractionQualitySettings;
  files: ExtractionBatchFilePlan[];
  tasks: ExtractionBatchTask[];
}

export interface PreparedOpenAiExtractionBatch {
  manifest: OpenAiExtractionBatchManifest;
  inputFile: Blob;
  requestCount: number;
  requestBytes: number;
}

interface BatchRequestLine {
  custom_id: string;
  method: "POST";
  url: "/v1/chat/completions";
  body: OpenAiChatCompletionBody;
}

interface BatchResultLine {
  custom_id?: string;
  response?: {
    status_code?: number;
    request_id?: string;
    body?: unknown;
  };
  error?: {
    code?: string;
    message?: string;
  } | null;
}

const TEXT_FILTER_TOOL: NativeToolDefinition = {
  name: "return_text_filter_decision",
  description:
    "Return paragraph IDs to keep as main narrative content for qualitative coding.",
  parameters: {
    type: "object",
    properties: {
      keep: {
        type: "array",
        items: {
          type: "object",
          properties: {
            id: { type: "string" },
            possible_boilerplate: { type: "boolean" },
            section_heading: { type: "string" },
            note: { type: "string" },
            confidence: { type: "number" },
          },
          required: ["id"],
        },
      },
      warnings: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: ["keep"],
  },
};

const VISION_PAGE_TOOL: NativeToolDefinition = {
  name: "return_vision_page_paragraphs",
  description:
    "Return OCR paragraph extraction for one PDF page image, excluding non-core text.",
  parameters: {
    type: "object",
    properties: {
      paragraphs: {
        type: "array",
        items: {
          type: "object",
          properties: {
            text: { type: "string" },
            section_heading: { type: "string" },
            note: { type: "string" },
            possible_boilerplate: { type: "boolean" },
            confidence: { type: "number" },
          },
          required: ["text"],
        },
      },
      warnings: {
        type: "array",
        items: { type: "string" },
      },
    },
    required: ["paragraphs"],
  },
};

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

function normalizeParagraph(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function isLikelyCodingParagraph(
  text: string,
  quality: ExtractionQualitySettings,
): boolean {
  const normalized = normalizeParagraph(text);
  if (!normalized) return false;

  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length < quality.minWordsPerParagraph) {
    return false;
  }

  const alphaChars = (normalized.match(/[A-Za-z]/g) ?? []).length;
  if (alphaChars < quality.minAlphaCharsPerParagraph) {
    return false;
  }

  if (!quality.requireSentenceTerminatorForShortParagraphs) {
    return true;
  }

  const hasSentenceEnding = /[.!?]["')\]]?$/.test(normalized);
  if (!hasSentenceEnding && words.length < quality.shortParagraphWordThreshold) {
    return false;
  }

  return true;
}

function chunkParagraphs(
  paragraphs: ParagraphCandidate[],
  maxChars = 7000,
): ParagraphCandidate[][] {
  const chunks: ParagraphCandidate[][] = [];
  let currentChunk: ParagraphCandidate[] = [];
  let currentLength = 0;

  for (const paragraph of paragraphs) {
    const serializedLength = paragraph.text.length + 120;
    if (
      currentChunk.length > 0 &&
      currentLength + serializedLength > maxChars
    ) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentLength = 0;
    }
    currentChunk.push(paragraph);
    currentLength += serializedLength;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

function extractJsonObject(raw: string): string {
  const fencedMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fencedMatch?.[1]) {
    return fencedMatch[1].trim();
  }

  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new Error("Model output did not contain JSON.");
  }
  return raw.slice(start, end + 1);
}

function clampConfidence(value: unknown): number | undefined {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return undefined;
  }
  return Math.max(0, Math.min(1, value));
}

function buildVisionRequestMessages(
  visionSystemPrompt: string,
  quality: ExtractionQualitySettings,
  pdfName: string,
  pageNumber: number,
  imageDataUrl: string,
): ChatMessage[] {
  const userContent = [
    {
      type: "text" as const,
      text: [
        `Document: ${pdfName}`,
        `Page: ${pageNumber}`,
        "Extract main-body paragraphs from this page image.",
        `Minimum words per paragraph: ${quality.minWordsPerParagraph}`,
        `Minimum alphabetic characters per paragraph: ${quality.minAlphaCharsPerParagraph}`,
        quality.requireSentenceTerminatorForShortParagraphs
          ? `If a paragraph has fewer than ${quality.shortParagraphWordThreshold} words, require terminal punctuation (.,!,?).`
          : "Terminal punctuation is optional for short paragraphs.",
      ].join("\n"),
    },
    {
      type: "image_url" as const,
      image_url: { url: imageDataUrl },
    },
  ];

  return [
    { role: "system", content: visionSystemPrompt },
    { role: "user", content: userContent },
  ];
}

function buildTextFilterRequestMessages(
  textFilterSystemPrompt: string,
  quality: ExtractionQualitySettings,
  chunk: ParagraphCandidate[],
): ChatMessage[] {
  const input = chunk.map((paragraph) => ({
    id: paragraph.id,
    page_number: paragraph.pageNumber,
    text: paragraph.text,
  }));

  const userPrompt = [
    "Input paragraphs JSON:",
    JSON.stringify(input),
    `Minimum words per paragraph: ${quality.minWordsPerParagraph}`,
    `Minimum alphabetic characters per paragraph: ${quality.minAlphaCharsPerParagraph}`,
    quality.requireSentenceTerminatorForShortParagraphs
      ? `If paragraph is shorter than ${quality.shortParagraphWordThreshold} words, keep only if it ends with ., !, or ?.`
      : "Terminal punctuation is optional for short paragraphs.",
    "Return only JSON with IDs to keep.",
  ].join("\n");

  return [
    { role: "system", content: textFilterSystemPrompt },
    { role: "user", content: userPrompt },
  ];
}

function createBatchRequestLine(
  customId: string,
  model: string,
  messages: ChatMessage[],
): BatchRequestLine {
  return {
    custom_id: customId,
    method: "POST",
    url: "/v1/chat/completions",
    body: buildOpenAiChatCompletionBody(model, messages, {
      responseFormat: "json_object",
    }),
  };
}

function createBatchCustomId(
  fileIndex: number,
  taskIndex: number,
  kind: "text_filter" | "vision_page",
): string {
  return `f${fileIndex}-${kind}-${taskIndex}`;
}

function parseOpenAiJsonResponse(body: unknown): unknown {
  const content = parseOpenAiContent(body);
  const jsonText = extractJsonObject(content);
  return JSON.parse(jsonText) as unknown;
}

function parseBatchLines(text: string): BatchResultLine[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as BatchResultLine);
}

function formatBatchError(line: BatchResultLine | undefined): string {
  if (!line) {
    return "No batch result was returned for this request.";
  }
  if (line.error?.message?.trim()) {
    return line.error.message.trim();
  }
  if (typeof line.response?.status_code === "number" && line.response.status_code >= 400) {
    const body = line.response.body;
    const message =
      typeof (body as { error?: { message?: unknown } })?.error?.message === "string"
        ? ((body as { error?: { message?: string } }).error?.message ?? "")
        : "";
    return message
      ? `HTTP ${line.response.status_code}: ${message}`
      : `HTTP ${line.response.status_code} returned by batch result.`;
  }
  return "Batch request did not return a usable response.";
}

function normalizeDecision(raw: unknown): ChunkDecision {
  const parsed = raw as { keep?: unknown; warnings?: unknown };
  if (!Array.isArray(parsed.keep)) {
    throw new Error("Missing keep array in model JSON response.");
  }

  const keep: KeepDecision[] = [];
  for (const item of parsed.keep) {
    const record = item as Record<string, unknown>;
    if (typeof record.id !== "string" || !record.id.trim()) {
      continue;
    }
    keep.push({
      id: record.id,
      section_heading:
        typeof record.section_heading === "string"
          ? record.section_heading.trim()
          : undefined,
      note: typeof record.note === "string" ? record.note.trim() : undefined,
      possible_boilerplate: Boolean(record.possible_boilerplate),
      confidence: clampConfidence(record.confidence),
    });
  }

  const warnings = Array.isArray(parsed.warnings)
    ? parsed.warnings
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
    : [];

  return { keep, warnings };
}

function normalizeVisionDecision(raw: unknown): VisionPageDecision {
  const parsed = raw as { paragraphs?: unknown; warnings?: unknown };
  if (!Array.isArray(parsed.paragraphs)) {
    throw new Error("Missing paragraphs array in vision JSON response.");
  }

  const paragraphs: VisionParagraphDecision[] = [];
  for (const item of parsed.paragraphs) {
    const record = item as Record<string, unknown>;
    if (typeof record.text !== "string") {
      continue;
    }
    const text = record.text.trim();
    if (!text) {
      continue;
    }
    paragraphs.push({
      text,
      section_heading:
        typeof record.section_heading === "string"
          ? record.section_heading.trim()
          : undefined,
      note: typeof record.note === "string" ? record.note.trim() : undefined,
      possible_boilerplate: Boolean(record.possible_boilerplate),
      confidence: clampConfidence(record.confidence),
    });
  }

  const warnings = Array.isArray(parsed.warnings)
    ? parsed.warnings
        .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
        .filter(Boolean)
    : [];

  return { paragraphs, warnings };
}

async function extractPageParagraphsWithVision(
  config: BackendConfig,
  visionSystemPrompt: string,
  quality: ExtractionQualitySettings,
  pdfName: string,
  pageNumber: number,
  imageDataUrl: string,
  signal?: AbortSignal,
): Promise<VisionPageDecision> {
  const responseText = await runChatCompletion(
    config,
    buildVisionRequestMessages(
      visionSystemPrompt,
      quality,
      pdfName,
      pageNumber,
      imageDataUrl,
    ),
    signal,
    {
      responseFormat: "json_object",
      tool: VISION_PAGE_TOOL,
    },
  );

  const jsonText = extractJsonObject(responseText);
  const rawObject = JSON.parse(jsonText) as unknown;
  return normalizeVisionDecision(rawObject);
}

async function filterChunkWithLlm(
  config: BackendConfig,
  textFilterSystemPrompt: string,
  quality: ExtractionQualitySettings,
  chunk: ParagraphCandidate[],
  signal?: AbortSignal,
): Promise<ChunkDecision> {
  const responseText = await runChatCompletion(
    config,
    buildTextFilterRequestMessages(textFilterSystemPrompt, quality, chunk),
    signal,
    {
      responseFormat: "json_object",
      tool: TEXT_FILTER_TOOL,
    },
  );

  const jsonText = extractJsonObject(responseText);
  const rawObject = JSON.parse(jsonText) as unknown;
  return normalizeDecision(rawObject);
}

function dedupeRowsWithinPdf(rows: ExtractionRow[]): ExtractionRow[] {
  const seen = new Set<string>();
  const deduped: ExtractionRow[] = [];

  for (const row of rows) {
    const key = row.paragraph.replace(/\s+/g, " ").trim();
    if (!key) {
      deduped.push(row);
      continue;
    }
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
  }

  return deduped.map((row, index) => ({ ...row, paragraph_index: index + 1 }));
}

function fallbackRowsFromChunk(
  pdfName: string,
  chunk: ParagraphCandidate[],
  errorMessage: string,
): ExtractionRow[] {
  return chunk.map((paragraph) => ({
    pdf_name: pdfName,
    paragraph: paragraph.text,
    paragraph_index: 0,
    page_number: paragraph.pageNumber,
    section_heading: "",
    notes: `LLM filtering failed for this chunk; included paragraph as fallback. Error: ${errorMessage}`,
    confidence: null,
  }));
}

async function processPdfWithVisionFallback(
  file: File,
  options: RunExtractionOptions,
  totalPdfs: number,
  getCompletedPdfs: () => number,
): Promise<ExtractionRow[]> {
  const visionSystemPrompt =
    options.prompts?.visionSystem?.trim() || DEFAULT_PROMPT_CONFIG.visionSystem;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const rows: ExtractionRow[] = [];
  let droppedByQuality = 0;

  await iteratePdfPageImages(
    file,
    async ({ pageNumber, totalPages, imageDataUrl }) => {
      throwIfAborted(options.signal);
      options.onProgress?.({
        totalPdfs,
        completedPdfs: getCompletedPdfs(),
        currentPdf: file.name,
        currentPage: pageNumber,
        totalPagesForCurrent: totalPages,
      });

      try {
        const decision = await withRetries(
          () =>
            extractPageParagraphsWithVision(
              options.config,
              visionSystemPrompt,
              qualitySettings,
              file.name,
              pageNumber,
              imageDataUrl,
              options.signal,
            ),
          { retries: options.retries, signal: options.signal },
        );

        decision.warnings.forEach((warning) => {
          const hasParagraphs = decision.paragraphs.length > 0;
          const looksLikeOcrLimit = OCR_LIMIT_WARNING_PATTERN.test(warning);
          if (hasParagraphs && looksLikeOcrLimit) {
            options.onLog?.(
              "info",
              `${file.name} page ${pageNumber}: model reported partial OCR limits but still returned extractable paragraphs.`,
            );
            return;
          }
          options.onLog?.(
            "warning",
            `${file.name} page ${pageNumber}: ${warning}`,
          );
        });

        if (decision.paragraphs.length === 0) {
          options.onLog?.(
            "warning",
            `${file.name} page ${pageNumber}: vision model returned no main-body paragraphs.`,
          );
          return;
        }

        decision.paragraphs.forEach((paragraph) => {
          if (!isLikelyCodingParagraph(paragraph.text, qualitySettings)) {
            droppedByQuality += 1;
            return;
          }

          const notes: string[] = [];
          if (paragraph.note) notes.push(paragraph.note);
          if (paragraph.possible_boilerplate) notes.push("possible boilerplate");

          rows.push({
            pdf_name: file.name,
            paragraph: paragraph.text,
            paragraph_index: 0,
            page_number: pageNumber,
            section_heading: paragraph.section_heading ?? "",
            notes: notes.join("; "),
            confidence: paragraph.confidence ?? null,
          });
        });
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }
        const message =
          error instanceof Error ? error.message : "Unknown vision extraction error";
        options.onLog?.(
          "error",
          `${file.name} page ${pageNumber}: vision extraction failed after retries. ${message}`,
        );
        rows.push({
          pdf_name: file.name,
          paragraph: "",
          paragraph_index: 0,
          page_number: pageNumber,
          section_heading: "",
          notes: `Vision extraction failed on page ${pageNumber}: ${message}`,
          confidence: null,
        });
      }
    },
    options.signal,
  );

  if (rows.length === 0) {
    rows.push({
      pdf_name: file.name,
      paragraph: "",
      paragraph_index: 0,
      page_number: null,
      section_heading: "",
      notes:
        "No extractable paragraphs were found by text-layer parsing or vision OCR extraction.",
      confidence: null,
    });
  }

  if (droppedByQuality > 0) {
    options.onLog?.(
      "info",
      `${file.name}: dropped ${droppedByQuality} short/non-sentence fragments during quality filtering.`,
    );
  }

  const deduped = dedupeRowsWithinPdf(rows);
  const removedDuplicates = rows.length - deduped.length;
  if (removedDuplicates > 0) {
    options.onLog?.(
      "info",
      `${file.name}: removed ${removedDuplicates} exact duplicate paragraphs within this PDF.`,
    );
  }

  return deduped;
}

async function processSinglePdf(
  file: File,
  options: RunExtractionOptions,
  totalPdfs: number,
  getCompletedPdfs: () => number,
): Promise<ExtractionRow[]> {
  const textFilterSystemPrompt =
    options.prompts?.textFilterSystem?.trim() ||
    DEFAULT_PROMPT_CONFIG.textFilterSystem;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const parsed = await extractPdfCandidates(
    file,
    (page, totalPages) => {
      options.onProgress?.({
        totalPdfs,
        completedPdfs: getCompletedPdfs(),
        currentPdf: file.name,
        currentPage: page,
        totalPagesForCurrent: totalPages,
      });
    },
    options.signal,
  );

  parsed.warnings.forEach((warning) =>
    options.onLog?.("warning", `${file.name}: ${warning}`),
  );

  if (parsed.paragraphs.length === 0) {
    options.onLog?.(
      "warning",
      `${file.name}: no paragraph candidates remained after text-layer parsing.`,
    );
    options.onLog?.(
      "info",
      `${file.name}: attempting vision OCR fallback on rendered page images (requires a vision-capable model).`,
    );
    return processPdfWithVisionFallback(
      file,
      options,
      totalPdfs,
      getCompletedPdfs,
    );
  }

  const chunks = chunkParagraphs(parsed.paragraphs);
  const rows: ExtractionRow[] = [];
  let droppedByQuality = 0;

  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    throwIfAborted(options.signal);
    const chunk = chunks[chunkIndex];
    const chunkLabel = `chunk ${chunkIndex + 1}/${chunks.length}`;

    try {
      const decision = await withRetries(
        () =>
          filterChunkWithLlm(
            options.config,
            textFilterSystemPrompt,
            qualitySettings,
            chunk,
            options.signal,
          ),
        { retries: options.retries, signal: options.signal },
      );

      const candidateMap = new Map(chunk.map((item) => [item.id, item]));
      const used = new Set<string>();
      for (const keep of decision.keep) {
        const candidate = candidateMap.get(keep.id);
        if (!candidate || used.has(keep.id)) continue;
        if (!isLikelyCodingParagraph(candidate.text, qualitySettings)) {
          droppedByQuality += 1;
          continue;
        }
        used.add(keep.id);

        const notes: string[] = [];
        if (keep.note) notes.push(keep.note);
        if (keep.possible_boilerplate) notes.push("possible boilerplate");

        rows.push({
          pdf_name: parsed.pdfName,
          paragraph: candidate.text,
          paragraph_index: 0,
          page_number: candidate.pageNumber,
          section_heading: keep.section_heading ?? "",
          notes: notes.join("; "),
          confidence: keep.confidence ?? null,
        });
      }

      if ((decision.warnings?.length ?? 0) > 0) {
        options.onLog?.(
          "warning",
          `${file.name} ${chunkLabel}: ${decision.warnings?.join(" | ")}`,
        );
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        throw error;
      }
      const message =
        error instanceof Error ? error.message : "Unknown model error";
      options.onLog?.(
        "error",
        `${file.name} ${chunkLabel}: LLM filtering failed after retries. ${message}`,
      );
      rows.push(...fallbackRowsFromChunk(parsed.pdfName, chunk, message));
    }
  }

  const deduped = dedupeRowsWithinPdf(rows);
  const removedDuplicates = rows.length - deduped.length;
  if (removedDuplicates > 0) {
    options.onLog?.(
      "info",
      `${file.name}: removed ${removedDuplicates} exact duplicate paragraphs within this PDF.`,
    );
  }

  if (droppedByQuality > 0) {
    options.onLog?.(
      "info",
      `${file.name}: dropped ${droppedByQuality} short/non-sentence fragments during quality filtering.`,
    );
  }

  return deduped;
}

async function prepareSinglePdfForBatch(
  file: File,
  fileIndex: number,
  options: RunExtractionOptions,
  totalPdfs: number,
  getCompletedPdfs: () => number,
): Promise<{
  filePlan: ExtractionBatchFilePlan;
  tasks: ExtractionBatchTask[];
  requests: BatchRequestLine[];
}> {
  const textFilterSystemPrompt =
    options.prompts?.textFilterSystem?.trim() ||
    DEFAULT_PROMPT_CONFIG.textFilterSystem;
  const visionSystemPrompt =
    options.prompts?.visionSystem?.trim() || DEFAULT_PROMPT_CONFIG.visionSystem;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const parsed = await extractPdfCandidates(
    file,
    (page, totalPages) => {
      options.onProgress?.({
        totalPdfs,
        completedPdfs: getCompletedPdfs(),
        currentPdf: file.name,
        currentPage: page,
        totalPagesForCurrent: totalPages,
      });
    },
    options.signal,
  );

  parsed.warnings.forEach((warning) =>
    options.onLog?.("warning", `${file.name}: ${warning}`),
  );

  if (parsed.paragraphs.length === 0) {
    options.onLog?.(
      "warning",
      `${file.name}: no paragraph candidates remained after text-layer parsing.`,
    );
    options.onLog?.(
      "info",
      `${file.name}: attempting vision OCR fallback on rendered page images (requires a vision-capable model).`,
    );

    const tasks: ExtractionBatchTask[] = [];
    const requests: BatchRequestLine[] = [];
    let taskIndex = 0;

    await iteratePdfPageImages(
      file,
      async ({ pageNumber, totalPages, imageDataUrl }) => {
        throwIfAborted(options.signal);
        options.onProgress?.({
          totalPdfs,
          completedPdfs: getCompletedPdfs(),
          currentPdf: file.name,
          currentPage: pageNumber,
          totalPagesForCurrent: totalPages,
        });

        const customId = createBatchCustomId(fileIndex, taskIndex, "vision_page");
        const task: VisionPageBatchTask = {
          kind: "vision_page",
          customId,
          pdfName: file.name,
          fileIndex,
          taskIndex,
          pageNumber,
          totalPages,
        };
        tasks.push(task);
        requests.push(
          createBatchRequestLine(
            customId,
            options.config.model,
            buildVisionRequestMessages(
              visionSystemPrompt,
              qualitySettings,
              file.name,
              pageNumber,
              imageDataUrl,
            ),
          ),
        );
        taskIndex += 1;
      },
      options.signal,
    );

    return {
      filePlan: {
        pdfName: file.name,
        fileIndex,
        mode: "vision",
        totalPages: parsed.totalPages,
      },
      tasks,
      requests,
    };
  }

  const chunks = chunkParagraphs(parsed.paragraphs);
  const tasks: ExtractionBatchTask[] = [];
  const requests: BatchRequestLine[] = [];

  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
    const chunk = chunks[chunkIndex];
    const customId = createBatchCustomId(fileIndex, chunkIndex, "text_filter");
    const task: TextFilterBatchTask = {
      kind: "text_filter",
      customId,
      pdfName: file.name,
      fileIndex,
      taskIndex: chunkIndex,
      chunkIndex,
      totalChunks: chunks.length,
      chunk,
    };
    tasks.push(task);
    requests.push(
      createBatchRequestLine(
        customId,
        options.config.model,
        buildTextFilterRequestMessages(
          textFilterSystemPrompt,
          qualitySettings,
          chunk,
        ),
      ),
    );
  }

  return {
    filePlan: {
      pdfName: file.name,
      fileIndex,
      mode: "text",
      totalPages: parsed.totalPages,
    },
    tasks,
    requests,
  };
}

export async function prepareOpenAiExtractionBatch(
  files: File[],
  options: RunExtractionOptions,
): Promise<PreparedOpenAiExtractionBatch> {
  if (options.config.kind !== "openai") {
    throw new Error("OpenAI batch mode requires an OpenAI-compatible backend.");
  }

  const totalPdfs = files.length;
  let completedPdfs = 0;
  const qualitySettings = sanitizeQualitySettings(options.quality);

  const preparedFiles = await mapWithConcurrency(
    files,
    options.fileConcurrency,
    async (file, fileIndex) => {
      try {
        options.onLog?.("info", `Preparing batch requests for ${file.name}`);
        const prepared = await prepareSinglePdfForBatch(
          file,
          fileIndex,
          options,
          totalPdfs,
          () => completedPdfs,
        );
        options.onLog?.(
          "info",
          `Prepared ${file.name} with ${prepared.requests.length} batch request(s).`,
        );
        return prepared;
      } finally {
        completedPdfs += 1;
        options.onProgress?.({
          totalPdfs,
          completedPdfs,
          currentPdf: file.name,
          currentPage: 0,
          totalPagesForCurrent: 0,
        });
      }
    },
    options.signal,
  );

  const manifest: OpenAiExtractionBatchManifest = {
    version: 1,
    createdAt: new Date().toISOString(),
    model: options.config.model,
    quality: qualitySettings,
    files: preparedFiles.map((prepared) => prepared.filePlan),
    tasks: preparedFiles.flatMap((prepared) => prepared.tasks),
  };

  const lines = preparedFiles.flatMap((prepared) => prepared.requests);
  if (lines.length === 0) {
    throw new Error("No batch requests were generated from the selected PDFs.");
  }

  const jsonl = lines.map((line) => JSON.stringify(line)).join("\n");
  const inputFile = new Blob([`${jsonl}\n`], {
    type: "application/x-ndjson;charset=utf-8",
  });

  const maxBatchBytes = 200 * 1024 * 1024;
  if (inputFile.size > maxBatchBytes) {
    throw new Error(
      `Batch input file is too large (${Math.round(inputFile.size / (1024 * 1024))} MB). Reduce the PDF set or disable vision-heavy batches.`,
    );
  }

  const maxRequestsPerBatch = 50000;
  if (lines.length > maxRequestsPerBatch) {
    throw new Error(
      `Batch contains ${lines.length} requests, exceeding the supported limit of ${maxRequestsPerBatch}. Reduce the PDF set.`,
    );
  }

  return {
    manifest,
    inputFile,
    requestCount: lines.length,
    requestBytes: inputFile.size,
  };
}

export function importOpenAiBatchResultFiles(
  manifest: OpenAiExtractionBatchManifest,
  outputFileText: string | null,
  errorFileText: string | null,
  callbacks?: ExtractionCallbacks,
): ExtractionRow[] {
  const outputMap = new Map(
    parseBatchLines(outputFileText ?? "")
      .filter((line): line is BatchResultLine & { custom_id: string } =>
        typeof line.custom_id === "string" && line.custom_id.length > 0,
      )
      .map((line) => [line.custom_id, line]),
  );
  const errorMap = new Map(
    parseBatchLines(errorFileText ?? "")
      .filter((line): line is BatchResultLine & { custom_id: string } =>
        typeof line.custom_id === "string" && line.custom_id.length > 0,
      )
      .map((line) => [line.custom_id, line]),
  );

  const rowsByPdf = new Map<string, ExtractionRow[]>();
  const droppedByQuality = new Map<string, number>();
  const qualitySettings = sanitizeQualitySettings(manifest.quality);

  function pushRow(pdfName: string, row: ExtractionRow): void {
    const current = rowsByPdf.get(pdfName) ?? [];
    current.push(row);
    rowsByPdf.set(pdfName, current);
  }

  function addDropped(pdfName: string, count = 1): void {
    droppedByQuality.set(pdfName, (droppedByQuality.get(pdfName) ?? 0) + count);
  }

  for (const task of manifest.tasks) {
    const line = outputMap.get(task.customId) ?? errorMap.get(task.customId);

    try {
      if (
        !line?.response?.body ||
        typeof line.response.status_code !== "number" ||
        line.response.status_code >= 400
      ) {
        throw new Error(formatBatchError(line));
      }

      const rawObject = parseOpenAiJsonResponse(line.response.body);

      if (task.kind === "text_filter") {
        const decision = normalizeDecision(rawObject);
        const candidateMap = new Map(task.chunk.map((item) => [item.id, item]));
        const used = new Set<string>();

        for (const keep of decision.keep) {
          const candidate = candidateMap.get(keep.id);
          if (!candidate || used.has(keep.id)) continue;
          if (!isLikelyCodingParagraph(candidate.text, qualitySettings)) {
            addDropped(task.pdfName);
            continue;
          }
          used.add(keep.id);

          const notes: string[] = [];
          if (keep.note) notes.push(keep.note);
          if (keep.possible_boilerplate) notes.push("possible boilerplate");

          pushRow(task.pdfName, {
            pdf_name: task.pdfName,
            paragraph: candidate.text,
            paragraph_index: 0,
            page_number: candidate.pageNumber,
            section_heading: keep.section_heading ?? "",
            notes: notes.join("; "),
            confidence: keep.confidence ?? null,
          });
        }

        if ((decision.warnings?.length ?? 0) > 0) {
          callbacks?.onLog?.(
            "warning",
            `${task.pdfName} chunk ${task.chunkIndex + 1}/${task.totalChunks}: ${decision.warnings?.join(" | ")}`,
          );
        }
        continue;
      }

      const decision = normalizeVisionDecision(rawObject);

      decision.warnings.forEach((warning) => {
        const hasParagraphs = decision.paragraphs.length > 0;
        const looksLikeOcrLimit = OCR_LIMIT_WARNING_PATTERN.test(warning);
        if (hasParagraphs && looksLikeOcrLimit) {
          callbacks?.onLog?.(
            "info",
            `${task.pdfName} page ${task.pageNumber}: model reported partial OCR limits but still returned extractable paragraphs.`,
          );
          return;
        }
        callbacks?.onLog?.(
          "warning",
          `${task.pdfName} page ${task.pageNumber}: ${warning}`,
        );
      });

      if (decision.paragraphs.length === 0) {
        callbacks?.onLog?.(
          "warning",
          `${task.pdfName} page ${task.pageNumber}: vision model returned no main-body paragraphs.`,
        );
        continue;
      }

      decision.paragraphs.forEach((paragraph) => {
        if (!isLikelyCodingParagraph(paragraph.text, qualitySettings)) {
          addDropped(task.pdfName);
          return;
        }

        const notes: string[] = [];
        if (paragraph.note) notes.push(paragraph.note);
        if (paragraph.possible_boilerplate) notes.push("possible boilerplate");

        pushRow(task.pdfName, {
          pdf_name: task.pdfName,
          paragraph: paragraph.text,
          paragraph_index: 0,
          page_number: task.pageNumber,
          section_heading: paragraph.section_heading ?? "",
          notes: notes.join("; "),
          confidence: paragraph.confidence ?? null,
        });
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown batch result error";

      if (task.kind === "text_filter") {
        callbacks?.onLog?.(
          "error",
          `${task.pdfName} chunk ${task.chunkIndex + 1}/${task.totalChunks}: LLM batch request failed. ${message}`,
        );
        fallbackRowsFromChunk(task.pdfName, task.chunk, message).forEach((row) =>
          pushRow(task.pdfName, row),
        );
        continue;
      }

      callbacks?.onLog?.(
        "error",
        `${task.pdfName} page ${task.pageNumber}: vision batch request failed. ${message}`,
      );
      pushRow(task.pdfName, {
        pdf_name: task.pdfName,
        paragraph: "",
        paragraph_index: 0,
        page_number: task.pageNumber,
        section_heading: "",
        notes: `Vision extraction failed on page ${task.pageNumber}: ${message}`,
        confidence: null,
      });
    }
  }

  const finalRows: ExtractionRow[] = [];
  const sortedFiles = [...manifest.files].sort((a, b) => a.fileIndex - b.fileIndex);

  for (const filePlan of sortedFiles) {
    const rows = rowsByPdf.get(filePlan.pdfName) ?? [];
    if (filePlan.mode === "vision" && rows.length === 0) {
      rows.push({
        pdf_name: filePlan.pdfName,
        paragraph: "",
        paragraph_index: 0,
        page_number: null,
        section_heading: "",
        notes:
          "No extractable paragraphs were found by text-layer parsing or vision OCR extraction.",
        confidence: null,
      });
    }

    const deduped = dedupeRowsWithinPdf(rows);
    const removedDuplicates = rows.length - deduped.length;
    if (removedDuplicates > 0) {
      callbacks?.onLog?.(
        "info",
        `${filePlan.pdfName}: removed ${removedDuplicates} exact duplicate paragraphs within this PDF.`,
      );
    }

    const dropped = droppedByQuality.get(filePlan.pdfName) ?? 0;
    if (dropped > 0) {
      callbacks?.onLog?.(
        "info",
        `${filePlan.pdfName}: dropped ${dropped} short/non-sentence fragments during quality filtering.`,
      );
    }

    finalRows.push(...deduped);
  }

  return finalRows;
}

export async function runExtraction(
  files: File[],
  options: RunExtractionOptions,
): Promise<ExtractionRow[]> {
  const totalPdfs = files.length;
  let completedPdfs = 0;

  const perFileRows = await mapWithConcurrency(
    files,
    options.fileConcurrency,
    async (file) => {
      try {
        options.onLog?.("info", `Starting extraction for ${file.name}`);
        const rows = await processSinglePdf(file, options, totalPdfs, () => completedPdfs);
        options.onLog?.("info", `Finished ${file.name} with ${rows.length} paragraphs.`);
        return rows;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          throw error;
        }

        const message =
          error instanceof Error ? error.message : "Unknown processing error";
        options.onLog?.("error", `${file.name}: failed to process file. ${message}`);
        return [
          {
            pdf_name: file.name,
            paragraph: "",
            paragraph_index: 1,
            page_number: null,
            section_heading: "",
            notes: `File processing failed: ${message}`,
            confidence: null,
          },
        ] as ExtractionRow[];
      } finally {
        completedPdfs += 1;
        options.onProgress?.({
          totalPdfs,
          completedPdfs,
          currentPdf: file.name,
          currentPage: 0,
          totalPagesForCurrent: 0,
        });
      }
    },
    options.signal,
  );

  return perFileRows.flat();
}
