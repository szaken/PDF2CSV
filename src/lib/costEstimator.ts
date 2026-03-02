import { mapWithConcurrency } from "./concurrency";
import { extractPdfCandidates } from "./pdf";
import { DEFAULT_PROMPT_CONFIG } from "./prompts";
import { sanitizeQualitySettings } from "./settings";
import type {
  ExtractionQualitySettings,
  ParagraphCandidate,
  PromptConfig,
  RunProgress,
} from "../types";

interface ChatMessage {
  role: "system" | "user";
  content: string | Array<{ type: "text"; text: string } | { type: "image_url" }>;
}

interface TokenEncoder {
  encode: (text: string) => number[];
}

interface TiktokenModule {
  encodingForModel: (model: string) => TokenEncoder;
  getEncoding: (encoding: string) => TokenEncoder;
}

export interface TokenCostSettings {
  inputPricePer1M: number;
  outputPricePer1M: number;
  estimatedOutputTokensPerRequest: number;
  estimatedVisionImageTokens: number;
  applyBatchDiscount: boolean;
  batchDiscountPercent: number;
}

interface FileEstimate {
  pdfName: string;
  mode: "text" | "vision";
  requests: number;
  inputTokens: number;
}

export interface CostEstimateResult {
  usedTiktoken: boolean;
  requestCount: number;
  textRequestCount: number;
  visionRequestCount: number;
  inputTokens: number;
  outputTokens: number;
  estimatedTotalTokens: number;
  estimatedCostUsd: number;
  discountMultiplier: number;
  files: FileEstimate[];
  warnings: string[];
}

export interface EstimateExtractionCostOptions {
  model: string;
  files: File[];
  prompts?: PromptConfig;
  quality?: ExtractionQualitySettings;
  fileConcurrency: number;
  cost: TokenCostSettings;
  onProgress?: (progress: RunProgress) => void;
  onLog?: (level: "info" | "warning" | "error", message: string) => void;
  signal?: AbortSignal;
}

let tiktokenLoaderPromise: Promise<TiktokenModule | null> | null = null;

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
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

function estimateTokensFallback(text: string): number {
  return Math.ceil(text.length / 4);
}

function normalizePriceInput(value: number): number {
  if (!Number.isFinite(value) || value < 0) return 0;
  return value;
}

function normalizeBatchDiscount(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

function resolveEncodingNameFromModel(model: string): string {
  const normalized = model.trim().toLowerCase();
  if (!normalized) return "cl100k_base";
  if (
    normalized.includes("gpt-4o") ||
    normalized.startsWith("o1") ||
    normalized.startsWith("o3") ||
    normalized.startsWith("gpt-5")
  ) {
    return "o200k_base";
  }
  return "cl100k_base";
}

async function loadTiktokenModule(): Promise<TiktokenModule | null> {
  if (!tiktokenLoaderPromise) {
    const runtimeImport = new Function(
      "specifier",
      "return import(specifier)",
    ) as (specifier: string) => Promise<unknown>;

    tiktokenLoaderPromise = runtimeImport(
      "https://esm.sh/js-tiktoken@1.0.21/lite?bundle",
    )
      .then((mod) => {
        const candidate = mod as Partial<TiktokenModule>;
        if (
          typeof candidate.encodingForModel === "function" &&
          typeof candidate.getEncoding === "function"
        ) {
          return candidate as TiktokenModule;
        }
        return null;
      })
      .catch(() => null);
  }
  return tiktokenLoaderPromise;
}

function chooseEncoder(tiktoken: TiktokenModule, model: string): TokenEncoder {
  try {
    return tiktoken.encodingForModel(model);
  } catch {
    return tiktoken.getEncoding(resolveEncodingNameFromModel(model));
  }
}

function countMessageTokens(
  message: ChatMessage,
  encodeText: (text: string) => number,
  estimatedVisionImageTokens: number,
): number {
  let total = 4 + encodeText(message.role);
  if (typeof message.content === "string") {
    total += encodeText(message.content);
    return total;
  }

  for (const part of message.content) {
    if (part.type === "text") {
      total += encodeText(part.text);
      continue;
    }
    total += estimatedVisionImageTokens;
  }

  return total;
}

function countChatRequestTokens(
  messages: ChatMessage[],
  encodeText: (text: string) => number,
  estimatedVisionImageTokens: number,
): number {
  let total = 3;
  for (const message of messages) {
    total += countMessageTokens(
      message,
      encodeText,
      estimatedVisionImageTokens,
    );
  }
  return total + 3;
}

function buildTextFilterMessages(
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

function buildVisionMessages(
  visionSystemPrompt: string,
  quality: ExtractionQualitySettings,
  pdfName: string,
  pageNumber: number,
): ChatMessage[] {
  const userText = [
    `Document: ${pdfName}`,
    `Page: ${pageNumber}`,
    "Extract main-body paragraphs from this page image.",
    `Minimum words per paragraph: ${quality.minWordsPerParagraph}`,
    `Minimum alphabetic characters per paragraph: ${quality.minAlphaCharsPerParagraph}`,
    quality.requireSentenceTerminatorForShortParagraphs
      ? `If a paragraph has fewer than ${quality.shortParagraphWordThreshold} words, require terminal punctuation (.,!,?).`
      : "Terminal punctuation is optional for short paragraphs.",
  ].join("\n");

  return [
    { role: "system", content: visionSystemPrompt },
    {
      role: "user",
      content: [{ type: "text", text: userText }, { type: "image_url" }],
    },
  ];
}

export async function estimateExtractionCost(
  options: EstimateExtractionCostOptions,
): Promise<CostEstimateResult> {
  throwIfAborted(options.signal);

  const quality = sanitizeQualitySettings(options.quality);
  const textFilterSystemPrompt =
    options.prompts?.textFilterSystem?.trim() ||
    DEFAULT_PROMPT_CONFIG.textFilterSystem;
  const visionSystemPrompt =
    options.prompts?.visionSystem?.trim() || DEFAULT_PROMPT_CONFIG.visionSystem;

  const tiktoken = await loadTiktokenModule();
  const encoder = tiktoken ? chooseEncoder(tiktoken, options.model) : null;
  const encodeText = encoder
    ? (text: string) => encoder.encode(text).length
    : estimateTokensFallback;

  const warnings: string[] = [];
  if (!tiktoken) {
    warnings.push(
      "tiktoken could not be loaded from CDN. Falling back to rough token estimation (~4 chars/token).",
    );
  }

  const estimatedVisionImageTokens = Math.max(
    0,
    Math.round(options.cost.estimatedVisionImageTokens),
  );

  let completedPdfs = 0;
  const files = await mapWithConcurrency(
    options.files,
    options.fileConcurrency,
    async (file) => {
      throwIfAborted(options.signal);
      options.onLog?.("info", `Estimating token usage for ${file.name}`);
      try {
        const parsed = await extractPdfCandidates(
          file,
          (currentPage, totalPages) => {
            options.onProgress?.({
              totalPdfs: options.files.length,
              completedPdfs,
              currentPdf: file.name,
              currentPage,
              totalPagesForCurrent: totalPages,
            });
          },
          options.signal,
        );

        if (parsed.paragraphs.length === 0) {
          const visionRequests = parsed.totalPages;
          let visionInputTokens = 0;
          for (let page = 1; page <= parsed.totalPages; page += 1) {
            const messages = buildVisionMessages(
              visionSystemPrompt,
              quality,
              file.name,
              page,
            );
            visionInputTokens += countChatRequestTokens(
              messages,
              encodeText,
              estimatedVisionImageTokens,
            );
          }

          return {
            pdfName: file.name,
            mode: "vision" as const,
            requests: visionRequests,
            inputTokens: visionInputTokens,
          };
        }

        const chunks = chunkParagraphs(parsed.paragraphs);
        let textInputTokens = 0;
        for (const chunk of chunks) {
          const messages = buildTextFilterMessages(
            textFilterSystemPrompt,
            quality,
            chunk,
          );
          textInputTokens += countChatRequestTokens(
            messages,
            encodeText,
            estimatedVisionImageTokens,
          );
        }

        return {
          pdfName: file.name,
          mode: "text" as const,
          requests: chunks.length,
          inputTokens: textInputTokens,
        };
      } finally {
        completedPdfs += 1;
        options.onProgress?.({
          totalPdfs: options.files.length,
          completedPdfs,
          currentPdf: file.name,
          currentPage: 0,
          totalPagesForCurrent: 0,
        });
      }
    },
    options.signal,
  );

  const requestCount = files.reduce((sum, file) => sum + file.requests, 0);
  const textRequestCount = files
    .filter((file) => file.mode === "text")
    .reduce((sum, file) => sum + file.requests, 0);
  const visionRequestCount = files
    .filter((file) => file.mode === "vision")
    .reduce((sum, file) => sum + file.requests, 0);
  const inputTokens = files.reduce((sum, file) => sum + file.inputTokens, 0);
  const outputTokens = Math.max(
    0,
    Math.round(options.cost.estimatedOutputTokensPerRequest),
  ) * requestCount;
  const estimatedTotalTokens = inputTokens + outputTokens;

  const inputPricePer1M = normalizePriceInput(options.cost.inputPricePer1M);
  const outputPricePer1M = normalizePriceInput(options.cost.outputPricePer1M);
  const baseCost =
    (inputTokens / 1_000_000) * inputPricePer1M +
    (outputTokens / 1_000_000) * outputPricePer1M;
  const discountMultiplier =
    options.cost.applyBatchDiscount && normalizeBatchDiscount(options.cost.batchDiscountPercent) > 0
      ? 1 - normalizeBatchDiscount(options.cost.batchDiscountPercent) / 100
      : 1;
  const estimatedCostUsd = baseCost * discountMultiplier;

  if (requestCount === 0) {
    warnings.push("No requests were generated from the selected PDFs.");
  }
  if (visionRequestCount > 0) {
    warnings.push(
      "Vision pricing is approximated using your per-image token estimate.",
    );
  }

  return {
    usedTiktoken: Boolean(tiktoken),
    requestCount,
    textRequestCount,
    visionRequestCount,
    inputTokens,
    outputTokens,
    estimatedTotalTokens,
    estimatedCostUsd,
    discountMultiplier,
    files,
    warnings,
  };
}
