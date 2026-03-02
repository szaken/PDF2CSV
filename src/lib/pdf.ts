import { GlobalWorkerOptions, getDocument } from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import type { PDFPageProxy } from "pdfjs-dist/types/src/display/api";
import type { ParagraphCandidate, ParsedPdf } from "../types";

GlobalWorkerOptions.workerSrc = workerSrc;

interface RawTextItem {
  str: string;
  transform: number[];
}

interface Line {
  y: number;
  text: string;
}

interface PageImagePayload {
  pageNumber: number;
  totalPages: number;
  imageDataUrl: string;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
}

function normalizeSpaces(input: string): string {
  return input.replace(/\s+/g, " ").trim();
}

function appendToken(line: string, token: string): string {
  if (!line) return token;
  if (line.endsWith("-") && /^[a-z]/.test(token)) {
    return `${line.slice(0, -1)}${token}`;
  }
  if (/[\(\[\{\/]$/.test(line) || /^[,.;:!?%\)\]\}]/.test(token)) {
    return `${line}${token}`;
  }
  return `${line} ${token}`;
}

function appendLine(paragraph: string, line: string): string {
  if (!paragraph) return line;
  if (paragraph.endsWith("-") && /^[a-z]/.test(line)) {
    return `${paragraph.slice(0, -1)}${line}`;
  }
  return `${paragraph} ${line}`;
}

function isRawTextItem(item: unknown): item is RawTextItem {
  const candidate = item as Partial<RawTextItem> | null;
  return Boolean(
    candidate &&
      typeof candidate.str === "string" &&
      Array.isArray(candidate.transform),
  );
}

function groupItemsIntoLines(items: RawTextItem[]): Line[] {
  const sorted = [...items].sort((a, b) => {
    const ay = a.transform[5] ?? 0;
    const by = b.transform[5] ?? 0;
    if (Math.abs(by - ay) > 0.2) return by - ay;
    const ax = a.transform[4] ?? 0;
    const bx = b.transform[4] ?? 0;
    return ax - bx;
  });

  const groups: Array<{ y: number; tokens: Array<{ x: number; text: string }> }> =
    [];

  for (const item of sorted) {
    const text = normalizeSpaces(item.str ?? "");
    if (!text) continue;

    const y = item.transform[5] ?? 0;
    const x = item.transform[4] ?? 0;
    const current = groups[groups.length - 1];

    if (!current || Math.abs(current.y - y) > 2.2) {
      groups.push({ y, tokens: [{ x, text }] });
      continue;
    }

    current.tokens.push({ x, text });
  }

  return groups
    .map((group) => {
      const orderedTokens = group.tokens.sort((a, b) => a.x - b.x);
      let line = "";
      for (const token of orderedTokens) {
        line = appendToken(line, token.text);
      }
      return { y: group.y, text: normalizeSpaces(line) };
    })
    .filter((line) => Boolean(line.text));
}

function median(values: number[]): number {
  if (values.length === 0) return 12;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function linesToParagraphs(lines: Line[]): string[] {
  if (lines.length === 0) return [];

  const gaps: number[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const gap = lines[i - 1].y - lines[i].y;
    if (gap > 0.5 && gap < 40) {
      gaps.push(gap);
    }
  }

  const baselineGap = median(gaps);
  const gapThreshold = baselineGap * 1.65;

  const paragraphs: string[] = [];
  let current = "";

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (!current) {
      current = line.text;
      continue;
    }

    const prev = lines[i - 1];
    const gap = prev.y - line.y;
    const isParagraphBreak = gap > gapThreshold;

    if (isParagraphBreak) {
      const normalized = normalizeSpaces(current);
      if (normalized) paragraphs.push(normalized);
      current = line.text;
      continue;
    }

    current = appendLine(current, line.text);
  }

  const normalized = normalizeSpaces(current);
  if (normalized) paragraphs.push(normalized);

  return paragraphs;
}

function normalizeForDuplicateCheck(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function isLikelyPageNumber(text: string): boolean {
  return (
    /^\d{1,4}$/.test(text) ||
    /^page\s+\d{1,4}$/i.test(text) ||
    /^\d{1,4}\s*\/\s*\d{1,4}$/.test(text)
  );
}

function isLikelyBoilerplate(text: string): boolean {
  const compact = normalizeForDuplicateCheck(text);
  if (!compact) return true;

  const boilerplatePattern =
    /\b(cookie|privacy|accept all|manage preferences|subscribe|newsletter|all rights reserved|terms of use|contact us|follow us|advertisement|sponsored|promo code|sign in|log in)\b/i;

  if (boilerplatePattern.test(compact)) {
    return true;
  }

  if (isLikelyPageNumber(compact)) {
    return true;
  }

  if (compact.length < 4) {
    return true;
  }

  const alphaCount = (compact.match(/[A-Za-z]/g) ?? []).length;
  if (compact.length < 50 && alphaCount / compact.length < 0.35) {
    return true;
  }

  return false;
}

function removeGlobalRepeats(
  candidates: ParagraphCandidate[],
): { paragraphs: ParagraphCandidate[]; warnings: string[] } {
  const keyToPages = new Map<string, Set<number>>();

  for (const candidate of candidates) {
    const key = normalizeForDuplicateCheck(candidate.text);
    if (key.length < 8) continue;
    const set = keyToPages.get(key) ?? new Set<number>();
    set.add(candidate.pageNumber);
    keyToPages.set(key, set);
  }

  let removedByRepeat = 0;
  const kept: ParagraphCandidate[] = [];
  for (const candidate of candidates) {
    const key = normalizeForDuplicateCheck(candidate.text);
    const pageSet = keyToPages.get(key);
    const repeatedAcrossPages =
      Boolean(pageSet) &&
      (pageSet?.size ?? 0) >= 3 &&
      key.length < 220 &&
      !/[.!?]$/.test(key);

    if (repeatedAcrossPages) {
      removedByRepeat += 1;
      continue;
    }
    kept.push(candidate);
  }

  const warnings: string[] = [];
  if (removedByRepeat > 0) {
    warnings.push(
      `Removed ${removedByRepeat} repeated short paragraphs likely to be template/header/footer text.`,
    );
  }

  return { paragraphs: kept, warnings };
}

async function renderPageAsJpegDataUrl(
  page: PDFPageProxy,
  maxDimension = 2300,
): Promise<string> {
  const baseViewport = page.getViewport({ scale: 1 });
  const largestSide = Math.max(baseViewport.width, baseViewport.height);
  const scale =
    largestSide > 0 ? Math.min(3, maxDimension / largestSide) : 1;
  const viewport = page.getViewport({ scale: Math.max(scale, 0.6) });

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(viewport.width));
  canvas.height = Math.max(1, Math.floor(viewport.height));
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not create canvas context for PDF page rendering.");
  }

  await page.render({ canvasContext: context, viewport, intent: "display" }).promise;
  return canvas.toDataURL("image/jpeg", 0.92);
}

export async function extractPdfCandidates(
  file: File,
  onPageProgress?: (page: number, totalPages: number) => void,
  signal?: AbortSignal,
): Promise<ParsedPdf> {
  throwIfAborted(signal);

  const loadingTask = getDocument({ data: await file.arrayBuffer() });
  const pdf = await loadingTask.promise;
  const paragraphCandidates: ParagraphCandidate[] = [];
  const warnings: string[] = [];
  const totalPages = pdf.numPages;
  let pagesWithoutTextLayer = 0;

  try {
    for (let pageNumber = 1; pageNumber <= totalPages; pageNumber += 1) {
      throwIfAborted(signal);
      const page = await pdf.getPage(pageNumber);
      const textContent = await page.getTextContent();
      const rawItems: RawTextItem[] = [];
      for (const item of textContent.items) {
        if (!isRawTextItem(item)) continue;
        rawItems.push({
          str: item.str,
          transform: item.transform,
        });
      }
      if (rawItems.length === 0) {
        pagesWithoutTextLayer += 1;
        onPageProgress?.(pageNumber, totalPages);
        continue;
      }
      const lines = groupItemsIntoLines(rawItems);
      const paragraphs = linesToParagraphs(lines);

      let removedByRule = 0;
      paragraphs.forEach((paragraph, index) => {
        if (isLikelyBoilerplate(paragraph)) {
          removedByRule += 1;
          return;
        }
        paragraphCandidates.push({
          id: `p${pageNumber}-${index + 1}`,
          pageNumber,
          text: paragraph,
        });
      });

      if (removedByRule > 0) {
        warnings.push(
          `Page ${pageNumber}: removed ${removedByRule} likely boilerplate candidates before LLM filtering.`,
        );
      }

      onPageProgress?.(pageNumber, totalPages);
    }
  } finally {
    loadingTask.destroy();
  }

  const afterRepeatFilter = removeGlobalRepeats(paragraphCandidates);
  warnings.push(...afterRepeatFilter.warnings);
  if (pagesWithoutTextLayer > 0) {
    warnings.push(
      `Detected ${pagesWithoutTextLayer}/${totalPages} page(s) without an extractable text layer.`,
    );
  }

  return {
    pdfName: file.name,
    paragraphs: afterRepeatFilter.paragraphs,
    totalPages,
    pagesWithoutTextLayer,
    warnings,
  };
}

export async function iteratePdfPageImages(
  file: File,
  onPage: (payload: PageImagePayload) => Promise<void>,
  signal?: AbortSignal,
): Promise<void> {
  throwIfAborted(signal);

  const loadingTask = getDocument({ data: await file.arrayBuffer() });
  const pdf = await loadingTask.promise;
  const totalPages = pdf.numPages;

  try {
    for (let pageNumber = 1; pageNumber <= totalPages; pageNumber += 1) {
      throwIfAborted(signal);
      const page = await pdf.getPage(pageNumber);
      const imageDataUrl = await renderPageAsJpegDataUrl(page);
      await onPage({
        pageNumber,
        totalPages,
        imageDataUrl,
      });
    }
  } finally {
    loadingTask.destroy();
  }
}
