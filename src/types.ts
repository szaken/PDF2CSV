export type BackendKind = "openai" | "ollama";

export interface BackendConfig {
  kind: BackendKind;
  baseUrl: string;
  apiKey: string;
  model: string;
  ollama?: OllamaGenerationSettings;
}

export type LogLevel = "info" | "warning" | "error";

export interface LogEntry {
  id: string;
  level: LogLevel;
  message: string;
  timestamp: string;
}

export interface ExtractionRow {
  pdf_name: string;
  url: string;
  paragraph: string;
  paragraph_index: number;
  page_number: number | null;
  section_heading: string;
  notes: string;
  confidence: number | null;
}

export interface ParagraphCandidate {
  id: string;
  pageNumber: number;
  text: string;
}

export interface ParsedPdf {
  pdfName: string;
  paragraphs: ParagraphCandidate[];
  totalPages: number;
  pagesWithoutTextLayer: number;
  warnings: string[];
}

export interface KeepDecision {
  id: string;
  url?: string;
  section_heading?: string;
  note?: string;
  confidence?: number;
  possible_boilerplate?: boolean;
}

export interface ChunkDecision {
  keep: KeepDecision[];
  warnings?: string[];
}

export interface RunProgress {
  totalPdfs: number;
  completedPdfs: number;
  currentPdf: string;
  currentPage: number;
  totalPagesForCurrent: number;
}

export interface PromptConfig {
  textFilterSystem: string;
  visionSystem: string;
}

export interface ExtractionQualitySettings {
  minWordsPerParagraph: number;
  minAlphaCharsPerParagraph: number;
  shortParagraphWordThreshold: number;
  requireSentenceTerminatorForShortParagraphs: boolean;
}

export interface OllamaGenerationSettings {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
  contextSize: number;
  useNativeToolCalling: boolean;
}
