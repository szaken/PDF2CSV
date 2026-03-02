import { useEffect, useMemo, useRef, useState } from "react";
import {
  cancelOpenAiBatch,
  createOpenAiBatch,
  downloadOpenAiFileContent,
  listModels,
  retrieveOpenAiBatch,
  uploadOpenAiBatchInputFile,
  type OpenAiBatchObject,
} from "./lib/backend";
import {
  importOpenAiBatchResultFiles,
  prepareOpenAiExtractionBatch,
  runExtraction,
  type OpenAiExtractionBatchManifest,
} from "./lib/extractor";
import {
  estimateExtractionCost,
  type CostEstimateResult,
  type TokenCostSettings,
} from "./lib/costEstimator";
import { downloadCsv, downloadXlsx } from "./lib/exporters";
import {
  DEFAULT_PROMPT_CONFIG,
  parsePromptConfigMarkdown,
  promptConfigToMarkdown,
} from "./lib/prompts";
import {
  DEFAULT_OLLAMA_SETTINGS,
  DEFAULT_QUALITY_SETTINGS,
  sanitizeOllamaSettings,
  sanitizeQualitySettings,
} from "./lib/settings";
import type {
  BackendKind,
  ExtractionQualitySettings,
  ExtractionRow,
  LogEntry,
  LogLevel,
  OllamaGenerationSettings,
  PromptConfig,
  RunProgress,
} from "./types";

interface BackendModelState {
  models: string[];
  selectedModel: string;
  manualModel: string;
}

interface SavedSettings {
  rememberSettings: boolean;
  backendKind: BackendKind;
  baseUrl: string;
  backendModels?: Record<BackendKind, BackendModelState>;
  openAiExecutionMode?: OpenAiExecutionMode;
  selectedModel?: string;
  manualModel?: string;
  concurrency: number;
  retries: number;
  promptConfig: PromptConfig;
  qualitySettings: ExtractionQualitySettings;
  ollamaSettings: OllamaGenerationSettings;
  tokenCostSettings: TokenCostSettings;
}

type OpenAiExecutionMode = "live" | "batch";

type BatchActionState = "" | "refresh" | "import" | "cancel";

interface PersistedOpenAiBatchJob {
  batchId: string;
  baseUrl: string;
  status: string;
  inputFileId: string;
  outputFileId: string | null;
  errorFileId: string | null;
  requestCounts?: {
    total: number;
    completed: number;
    failed: number;
  };
  manifest: OpenAiExtractionBatchManifest;
  requestCount: number;
  requestBytes: number;
  createdAt: string;
  lastCheckedAt: string;
}

const SETTINGS_KEY = "pdf2csv.settings.v1";
const BATCH_SESSION_KEY = "pdf2csv.openai-batch.v1";
const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1/";
const DEFAULT_OLLAMA_BASE_URL = "http://192.168.4.35:11434";
const DEFAULT_TOKEN_COST_SETTINGS: TokenCostSettings = {
  inputPricePer1M: 0,
  outputPricePer1M: 0,
  estimatedOutputTokensPerRequest: 450,
  estimatedVisionImageTokens: 1100,
  applyBatchDiscount: true,
  batchDiscountPercent: 50,
};

function defaultBaseUrlForBackend(kind: BackendKind): string {
  return kind === "openai" ? DEFAULT_OPENAI_BASE_URL : DEFAULT_OLLAMA_BASE_URL;
}

function createLog(level: LogLevel, message: string): LogEntry {
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random()}`;
  return {
    id,
    level,
    message,
    timestamp: new Date().toLocaleTimeString(),
  };
}

function normalizePdfList(inputFiles: File[]): File[] {
  const map = new Map<string, File>();
  for (const file of inputFiles) {
    if (!file.name.toLowerCase().endsWith(".pdf")) continue;
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    map.set(key, file);
  }
  return Array.from(map.values());
}

function mergeFiles(existing: File[], incoming: File[]): File[] {
  return normalizePdfList([...existing, ...incoming]);
}

function buildDownloadName(extension: "csv" | "xlsx"): string {
  const stamp = new Date().toISOString().replace(/[:]/g, "-");
  return `pdf-paragraphs-${stamp}.${extension}`;
}

function triggerTextDownload(text: string, fileName: string): void {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function toNumber(value: string, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed;
}

function sanitizeTokenCostSettings(
  settings: Partial<TokenCostSettings> | undefined,
): TokenCostSettings {
  return {
    inputPricePer1M: Math.max(
      0,
      toNumber(
        String(settings?.inputPricePer1M ?? DEFAULT_TOKEN_COST_SETTINGS.inputPricePer1M),
        DEFAULT_TOKEN_COST_SETTINGS.inputPricePer1M,
      ),
    ),
    outputPricePer1M: Math.max(
      0,
      toNumber(
        String(settings?.outputPricePer1M ?? DEFAULT_TOKEN_COST_SETTINGS.outputPricePer1M),
        DEFAULT_TOKEN_COST_SETTINGS.outputPricePer1M,
      ),
    ),
    estimatedOutputTokensPerRequest: Math.max(
      0,
      Math.round(
        toNumber(
          String(
            settings?.estimatedOutputTokensPerRequest ??
              DEFAULT_TOKEN_COST_SETTINGS.estimatedOutputTokensPerRequest,
          ),
          DEFAULT_TOKEN_COST_SETTINGS.estimatedOutputTokensPerRequest,
        ),
      ),
    ),
    estimatedVisionImageTokens: Math.max(
      0,
      Math.round(
        toNumber(
          String(
            settings?.estimatedVisionImageTokens ??
              DEFAULT_TOKEN_COST_SETTINGS.estimatedVisionImageTokens,
          ),
          DEFAULT_TOKEN_COST_SETTINGS.estimatedVisionImageTokens,
        ),
      ),
    ),
    applyBatchDiscount:
      settings?.applyBatchDiscount ?? DEFAULT_TOKEN_COST_SETTINGS.applyBatchDiscount,
    batchDiscountPercent: Math.min(
      100,
      Math.max(
        0,
        toNumber(
          String(
            settings?.batchDiscountPercent ??
              DEFAULT_TOKEN_COST_SETTINGS.batchDiscountPercent,
          ),
          DEFAULT_TOKEN_COST_SETTINGS.batchDiscountPercent,
        ),
      ),
    ),
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isTerminalBatchStatus(status: string): boolean {
  return ["completed", "failed", "expired", "cancelled"].includes(status);
}

function restoreBatchJob(raw: string | null): PersistedOpenAiBatchJob | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as PersistedOpenAiBatchJob;
    if (
      !parsed.batchId ||
      !parsed.status ||
      !parsed.manifest ||
      !Array.isArray(parsed.manifest.tasks) ||
      !Array.isArray(parsed.manifest.files)
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function toPersistedBatchJob(
  batch: OpenAiBatchObject,
  manifest: OpenAiExtractionBatchManifest,
  requestCount: number,
  requestBytes: number,
  baseUrl: string,
): PersistedOpenAiBatchJob {
  const now = new Date().toISOString();
  return {
    batchId: batch.id,
    baseUrl,
    status: batch.status,
    inputFileId: batch.input_file_id,
    outputFileId: batch.output_file_id ?? null,
    errorFileId: batch.error_file_id ?? null,
    requestCounts: batch.request_counts,
    manifest,
    requestCount,
    requestBytes,
    createdAt: batch.created_at
      ? new Date(batch.created_at * 1000).toISOString()
      : now,
    lastCheckedAt: now,
  };
}

function mergeBatchJobUpdate(
  previous: PersistedOpenAiBatchJob,
  batch: OpenAiBatchObject,
): PersistedOpenAiBatchJob {
  return {
    ...previous,
    status: batch.status,
    inputFileId: batch.input_file_id,
    outputFileId: batch.output_file_id ?? null,
    errorFileId: batch.error_file_id ?? null,
    requestCounts: batch.request_counts,
    lastCheckedAt: new Date().toISOString(),
  };
}

function createDefaultBackendModels(): Record<BackendKind, BackendModelState> {
  return {
    openai: {
      models: [],
      selectedModel: "",
      manualModel: "",
    },
    ollama: {
      models: [],
      selectedModel: "",
      manualModel: "",
    },
  };
}

function sanitizeBackendModelState(
  state: Partial<BackendModelState> | undefined,
): BackendModelState {
  return {
    models: Array.isArray(state?.models)
      ? state.models.filter((model): model is string => typeof model === "string")
      : [],
    selectedModel:
      typeof state?.selectedModel === "string" ? state.selectedModel : "",
    manualModel: typeof state?.manualModel === "string" ? state.manualModel : "",
  };
}

function loadSavedBackendModels(
  parsed: SavedSettings,
): Record<BackendKind, BackendModelState> {
  if (parsed.backendModels) {
    return {
      openai: sanitizeBackendModelState(parsed.backendModels.openai),
      ollama: sanitizeBackendModelState(parsed.backendModels.ollama),
    };
  }

  const fallback = createDefaultBackendModels();
  fallback[parsed.backendKind] = sanitizeBackendModelState({
    selectedModel: parsed.selectedModel,
    manualModel: parsed.manualModel,
  });
  return fallback;
}

export default function App(): JSX.Element {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const promptFileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [backendKind, setBackendKind] = useState<BackendKind>("openai");
  const [baseUrl, setBaseUrl] = useState(
    defaultBaseUrlForBackend("openai"),
  );
  const [apiKey, setApiKey] = useState("");
  const [openAiExecutionMode, setOpenAiExecutionMode] =
    useState<OpenAiExecutionMode>("live");
  const [backendModels, setBackendModels] = useState<Record<
    BackendKind,
    BackendModelState
  >>(createDefaultBackendModels);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");

  const [files, setFiles] = useState<File[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [concurrency, setConcurrency] = useState(2);
  const [retries, setRetries] = useState(2);
  const [rememberSettings, setRememberSettings] = useState(false);
  const [promptConfig, setPromptConfig] =
    useState<PromptConfig>(DEFAULT_PROMPT_CONFIG);
  const [qualitySettings, setQualitySettings] = useState<ExtractionQualitySettings>(
    DEFAULT_QUALITY_SETTINGS,
  );
  const [ollamaSettings, setOllamaSettings] = useState<OllamaGenerationSettings>(
    DEFAULT_OLLAMA_SETTINGS,
  );
  const [tokenCostSettings, setTokenCostSettings] = useState<TokenCostSettings>(
    DEFAULT_TOKEN_COST_SETTINGS,
  );

  const [rows, setRows] = useState<ExtractionRow[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [batchJob, setBatchJob] = useState<PersistedOpenAiBatchJob | null>(null);
  const [batchAction, setBatchAction] = useState<BatchActionState>("");
  const [isEstimatingCost, setIsEstimatingCost] = useState(false);
  const [costEstimate, setCostEstimate] = useState<CostEstimateResult | null>(null);

  const currentBackendModels = backendModels[backendKind];
  const models = currentBackendModels.models;
  const selectedModel = currentBackendModels.selectedModel;
  const manualModel = currentBackendModels.manualModel;
  const isBatchMode =
    backendKind === "openai" && openAiExecutionMode === "batch";
  const isBatchActionRunning = batchAction !== "";
  const activeModel = useMemo(
    () => manualModel.trim() || selectedModel.trim(),
    [manualModel, selectedModel],
  );

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as SavedSettings;
      if (!parsed.rememberSettings) return;
      setRememberSettings(true);
      setBackendKind(parsed.backendKind);
      setBaseUrl(parsed.baseUrl);
      setOpenAiExecutionMode(parsed.openAiExecutionMode ?? "live");
      setBackendModels(loadSavedBackendModels(parsed));
      setConcurrency(parsed.concurrency);
      setRetries(parsed.retries);
      setQualitySettings(sanitizeQualitySettings(parsed.qualitySettings));
      setOllamaSettings(sanitizeOllamaSettings(parsed.ollamaSettings));
      setTokenCostSettings(sanitizeTokenCostSettings(parsed.tokenCostSettings));
      if (
        parsed.promptConfig?.textFilterSystem &&
        parsed.promptConfig?.visionSystem
      ) {
        setPromptConfig(parsed.promptConfig);
      }
    } catch {
      // Ignore malformed local settings.
    }
  }, []);

  useEffect(() => {
    try {
      setBatchJob(restoreBatchJob(sessionStorage.getItem(BATCH_SESSION_KEY)));
    } catch {
      setBatchJob(null);
    }
  }, []);

  useEffect(() => {
    if (!rememberSettings) {
      localStorage.removeItem(SETTINGS_KEY);
      return;
    }

    const payload: SavedSettings = {
      rememberSettings,
      backendKind,
      baseUrl,
      backendModels,
      openAiExecutionMode,
      concurrency,
      retries,
      promptConfig,
      qualitySettings,
      ollamaSettings,
      tokenCostSettings,
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(payload));
  }, [
    rememberSettings,
    backendKind,
    baseUrl,
    backendModels,
    openAiExecutionMode,
    concurrency,
    retries,
    promptConfig,
    qualitySettings,
    ollamaSettings,
    tokenCostSettings,
  ]);

  useEffect(() => {
    try {
      if (!batchJob) {
        sessionStorage.removeItem(BATCH_SESSION_KEY);
        return;
      }
      sessionStorage.setItem(BATCH_SESSION_KEY, JSON.stringify(batchJob));
    } catch {
      // Ignore session storage quota or serialization failures.
    }
  }, [batchJob]);

  function appendLog(level: LogLevel, message: string): void {
    setLogs((previous) => [...previous, createLog(level, message)]);
  }

  function updateBackendModels(
    kind: BackendKind,
    updater: (current: BackendModelState) => BackendModelState,
  ): void {
    setBackendModels((previous) => ({
      ...previous,
      [kind]: updater(previous[kind]),
    }));
  }

  function onChooseFiles(filesLike: FileList | null): void {
    if (!filesLike) return;
    const incoming = Array.from(filesLike);
    const merged = mergeFiles(files, incoming);
    const added = merged.length - files.length;
    setFiles(merged);
    if (added > 0) {
      appendLog("info", `Added ${added} PDF file(s).`);
    }
  }

  function removeFile(fileToRemove: File): void {
    setFiles((previous) => {
      const next = previous.filter(
        (file) =>
          !(
            file.name === fileToRemove.name &&
            file.size === fileToRemove.size &&
            file.lastModified === fileToRemove.lastModified
          ),
      );
      return next;
    });
  }

  async function handleLoadModels(): Promise<void> {
    const kind = backendKind;
    setModelError("");
    setModelLoading(true);
    try {
      const loaded = await listModels({
        kind,
        baseUrl,
        apiKey,
      });
      updateBackendModels(kind, (current) => ({
        ...current,
        models: loaded,
        selectedModel:
          current.selectedModel && loaded.includes(current.selectedModel)
            ? current.selectedModel
            : (loaded[0] ?? ""),
      }));
      appendLog("info", `Loaded ${loaded.length} model(s) from endpoint.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setModelError(message);
      updateBackendModels(kind, (current) => ({
        ...current,
        models: [],
      }));
      appendLog(
        "warning",
        `Model listing failed. You can still enter a manual model ID. ${message}`,
      );
    } finally {
      setModelLoading(false);
    }
  }

  function warnAboutOpenAiBrowserProxyIfNeeded(): void {
    if (
      backendKind !== "openai" ||
      !/^https:\/\/api\.openai\.com(?:\/v1)?\/?$/i.test(baseUrl.trim())
    ) {
      return;
    }

    const host = window.location.hostname;
    const isLocalDevHost = host === "localhost" || host === "127.0.0.1";
    if (!isLocalDevHost) {
      appendLog(
        "warning",
        "Official OpenAI endpoint from a browser may require a relay/proxy. In local development, run via `npm run dev` to use built-in proxy routing.",
      );
    }
  }

  function clearBatchJob(): void {
    setBatchJob(null);
  }

  async function handleSubmitBatch(): Promise<void> {
    setRows([]);
    setLogs([]);
    setProgress({
      totalPdfs: files.length,
      completedPdfs: 0,
      currentPdf: "",
      currentPage: 0,
      totalPagesForCurrent: 0,
    });
    setIsRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;

    warnAboutOpenAiBrowserProxyIfNeeded();

    try {
      const prepared = await prepareOpenAiExtractionBatch(files, {
        config: {
          kind: "openai",
          baseUrl: baseUrl.trim(),
          apiKey,
          model: activeModel,
        },
        prompts: promptConfig,
        quality: qualitySettings,
        fileConcurrency: concurrency,
        retries,
        signal: controller.signal,
        onLog: appendLog,
        onProgress: setProgress,
      });

      appendLog(
        "info",
        `Prepared ${prepared.requestCount} batch request(s) in ${formatBytes(prepared.requestBytes)}.`,
      );

      const uploaded = await uploadOpenAiBatchInputFile(
        {
          baseUrl: baseUrl.trim(),
          apiKey,
        },
        prepared.inputFile,
        `pdf2csv-batch-${Date.now()}.jsonl`,
        controller.signal,
      );
      appendLog("info", `Uploaded batch input file ${uploaded.id}.`);

      const batch = await createOpenAiBatch(
        {
          baseUrl: baseUrl.trim(),
          apiKey,
        },
        uploaded.id,
        controller.signal,
        {
          app: "pdf2csv",
          model: activeModel,
        },
      );

      setBatchJob(
        toPersistedBatchJob(
          batch,
          prepared.manifest,
          prepared.requestCount,
          prepared.requestBytes,
          baseUrl.trim(),
        ),
      );
      setProgress(null);
      appendLog(
        "info",
        `Submitted OpenAI batch ${batch.id}. Refresh status later, then import the results file when processing finishes.`,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        appendLog("warning", "Batch submission canceled by user.");
      } else {
        const message =
          error instanceof Error ? error.message : "Unknown batch submission error";
        appendLog("error", `Batch submission failed: ${message}`);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }

  async function handleBatchRefresh(): Promise<void> {
    if (!batchJob) return;
    setBatchAction("refresh");
    try {
      const batch = await retrieveOpenAiBatch(
        {
          baseUrl: batchJob.baseUrl,
          apiKey,
        },
        batchJob.batchId,
      );
      const nextJob = mergeBatchJobUpdate(batchJob, batch);
      setBatchJob(nextJob);
      appendLog(
        "info",
        `Batch ${nextJob.batchId} status: ${nextJob.status}.`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown batch refresh error";
      appendLog("error", `Failed to refresh batch status: ${message}`);
    } finally {
      setBatchAction("");
    }
  }

  async function handleBatchImport(): Promise<void> {
    if (!batchJob) return;
    setBatchAction("import");
    setLogs([]);
    try {
      const batch = await retrieveOpenAiBatch(
        {
          baseUrl: batchJob.baseUrl,
          apiKey,
        },
        batchJob.batchId,
      );
      const nextJob = mergeBatchJobUpdate(batchJob, batch);
      setBatchJob(nextJob);

      if (!isTerminalBatchStatus(nextJob.status) && !nextJob.outputFileId) {
        throw new Error(
          `Batch is still ${nextJob.status}. Wait for completion, expiry, or cancellation before importing results.`,
        );
      }

      if (!nextJob.outputFileId && !nextJob.errorFileId) {
        throw new Error("Batch has no output or error file available to import.");
      }

      const [outputText, errorText] = await Promise.all([
        nextJob.outputFileId
          ? downloadOpenAiFileContent(
              {
                baseUrl: batchJob.baseUrl,
                apiKey,
              },
              nextJob.outputFileId,
            )
          : Promise.resolve(null),
        nextJob.errorFileId
          ? downloadOpenAiFileContent(
              {
                baseUrl: batchJob.baseUrl,
                apiKey,
              },
              nextJob.errorFileId,
            )
          : Promise.resolve(null),
      ]);

      const extractionRows = importOpenAiBatchResultFiles(
        nextJob.manifest,
        outputText,
        errorText,
        { onLog: appendLog },
      );
      setRows(extractionRows);
      appendLog(
        "info",
        `Imported batch results. Final dataset contains ${extractionRows.length} row(s).`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown batch import error";
      appendLog("error", `Batch import failed: ${message}`);
    } finally {
      setBatchAction("");
    }
  }

  async function handleBatchCancel(): Promise<void> {
    if (!batchJob) return;
    setBatchAction("cancel");
    try {
      const batch = await cancelOpenAiBatch(
        {
          baseUrl: batchJob.baseUrl,
          apiKey,
        },
        batchJob.batchId,
      );
      const nextJob = mergeBatchJobUpdate(batchJob, batch);
      setBatchJob(nextJob);
      appendLog(
        "warning",
        `Cancellation requested for batch ${nextJob.batchId}. Current status: ${nextJob.status}.`,
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown batch cancel error";
      appendLog("error", `Failed to cancel batch: ${message}`);
    } finally {
      setBatchAction("");
    }
  }

  async function handleEstimateCost(): Promise<void> {
    if (!activeModel) {
      appendLog(
        "error",
        "Select a model from the dropdown or provide a manual model ID before estimating cost.",
      );
      return;
    }
    if (files.length === 0) {
      appendLog("error", "Add at least one PDF before estimating cost.");
      return;
    }

    setLogs([]);
    setProgress({
      totalPdfs: files.length,
      completedPdfs: 0,
      currentPdf: "",
      currentPage: 0,
      totalPagesForCurrent: 0,
    });
    setIsEstimatingCost(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const estimate = await estimateExtractionCost({
        model: activeModel,
        files,
        prompts: promptConfig,
        quality: qualitySettings,
        fileConcurrency: concurrency,
        cost: tokenCostSettings,
        signal: controller.signal,
        onLog: appendLog,
        onProgress: setProgress,
      });
      setCostEstimate(estimate);
      appendLog(
        "info",
        `Estimated ${estimate.requestCount} request(s), ${estimate.inputTokens.toLocaleString()} input tokens, ${estimate.outputTokens.toLocaleString()} output tokens, estimated cost $${estimate.estimatedCostUsd.toFixed(4)}.`,
      );
      estimate.warnings.forEach((warning) => appendLog("warning", warning));
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        appendLog("warning", "Cost estimation canceled by user.");
      } else {
        const message =
          error instanceof Error ? error.message : "Unknown estimation error";
        appendLog("error", `Cost estimation failed: ${message}`);
      }
    } finally {
      setIsEstimatingCost(false);
      setProgress(null);
      abortRef.current = null;
    }
  }

  async function handleRun(): Promise<void> {
    if (!baseUrl.trim()) {
      appendLog("error", "Base URL is required.");
      return;
    }
    if (!activeModel) {
      appendLog(
        "error",
        "Select a model from the dropdown or provide a manual model ID.",
      );
      return;
    }
    if (files.length === 0) {
      appendLog("error", "Add at least one PDF before running extraction.");
      return;
    }

    if (isBatchMode) {
      await handleSubmitBatch();
      return;
    }

    setRows([]);
    setLogs([]);
    setProgress({
      totalPdfs: files.length,
      completedPdfs: 0,
      currentPdf: "",
      currentPage: 0,
      totalPagesForCurrent: 0,
    });
    setIsRunning(true);
    const controller = new AbortController();
    abortRef.current = controller;
    warnAboutOpenAiBrowserProxyIfNeeded();

    try {
      const extractionRows = await runExtraction(files, {
        config: {
          kind: backendKind,
          baseUrl: baseUrl.trim(),
          apiKey,
          model: activeModel,
          ollama: backendKind === "ollama" ? ollamaSettings : undefined,
        },
        prompts: promptConfig,
        quality: qualitySettings,
        fileConcurrency: concurrency,
        retries,
        signal: controller.signal,
        onLog: appendLog,
        onProgress: setProgress,
      });
      setRows(extractionRows);
      appendLog(
        "info",
        `Completed. Final dataset contains ${extractionRows.length} row(s).`,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        appendLog("warning", "Extraction canceled by user.");
      } else {
        const message =
          error instanceof Error ? error.message : "Unknown extraction error";
        appendLog("error", `Extraction stopped with an error: ${message}`);
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }

  function handleCancel(): void {
    abortRef.current?.abort();
    appendLog("warning", "Cancel requested. Stopping in-flight work...");
  }

  async function handlePromptFileImport(filesLike: FileList | null): Promise<void> {
    if (!filesLike || filesLike.length === 0) return;
    try {
      const file = filesLike[0];
      const markdown = await file.text();
      const parsed = parsePromptConfigMarkdown(markdown);
      setPromptConfig(parsed);
      appendLog("info", `Loaded prompt configuration from ${file.name}.`);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown prompt import error";
      appendLog("error", `Prompt import failed: ${message}`);
    }
  }

  function handlePromptDownload(): void {
    const markdown = promptConfigToMarkdown(promptConfig);
    triggerTextDownload(markdown, "pdf2csv-prompts.md");
    appendLog("info", "Downloaded prompt configuration markdown.");
  }

  const overallPercent = useMemo(() => {
    if (!progress || progress.totalPdfs === 0) return 0;
    return Math.round((progress.completedPdfs / progress.totalPdfs) * 100);
  }, [progress]);

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>PDF Main-Body Paragraph Extractor</h1>
        <p>
          Upload PDFs, keep core narrative paragraphs, deduplicate exact
          repeats per file, and download CSV or XLSX.
        </p>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>1) Backend Configuration</h2>

          <label className="field">
            <span>Backend type</span>
            <select
              value={backendKind}
              onChange={(event) => {
                const nextKind = event.target.value as BackendKind;
                setBackendKind(nextKind);
                setBaseUrl(defaultBaseUrlForBackend(nextKind));
                setModelError("");
              }}
              disabled={isRunning}
            >
              <option value="openai">OpenAI-compatible</option>
              <option value="ollama">Ollama-compatible</option>
            </select>
          </label>

          <label className="field">
            <span>Base URL</span>
            <input
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
              placeholder={
                backendKind === "openai"
                  ? "https://your-endpoint.example/v1"
                  : "http://localhost:11434"
              }
              disabled={isRunning}
            />
          </label>

          <label className="field">
            <span>API key (OpenAI-compatible only)</span>
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="sk-..."
              disabled={isRunning || backendKind === "ollama"}
            />
          </label>

          {backendKind === "openai" && (
            <label className="field">
              <span>OpenAI request mode</span>
              <select
                value={openAiExecutionMode}
                onChange={(event) =>
                  setOpenAiExecutionMode(
                    event.target.value as OpenAiExecutionMode,
                  )
                }
                disabled={isRunning || isBatchActionRunning}
              >
                <option value="live">Live requests</option>
                <option value="batch">Batch mode</option>
              </select>
            </label>
          )}

          <div className="inline-actions">
            <button
              type="button"
              onClick={handleLoadModels}
              disabled={isRunning || modelLoading || !baseUrl.trim()}
            >
              {modelLoading ? "Loading..." : "Load Models"}
            </button>
          </div>

          <label className="field">
            <span>Endpoint model list</span>
            <select
              value={selectedModel}
              onChange={(event) =>
                updateBackendModels(backendKind, (current) => ({
                  ...current,
                  selectedModel: event.target.value,
                }))
              }
              disabled={isRunning || models.length === 0}
            >
              <option value="">
                {models.length === 0 ? "No models loaded" : "Select model"}
              </option>
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Manual model ID (overrides dropdown when filled)</span>
            <input
              value={manualModel}
              onChange={(event) =>
                updateBackendModels(backendKind, (current) => ({
                  ...current,
                  manualModel: event.target.value,
                }))
              }
              placeholder="e.g. gpt-4o-mini or llama3.1:8b"
              disabled={isRunning}
            />
          </label>
          <p className="muted">
            Use a vision-capable model for scanned/image-only PDFs.
          </p>

          {backendKind === "ollama" && (
            <div className="subpanel">
              <h3>Ollama Advanced Settings</h3>
              <div className="split">
                <label className="field">
                  <span>Temperature</span>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.05}
                    value={ollamaSettings.temperature}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          temperature: toNumber(
                            event.target.value,
                            previous.temperature,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Top P</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={ollamaSettings.topP}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          topP: toNumber(event.target.value, previous.topP),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Top K</span>
                  <input
                    type="number"
                    min={0}
                    max={500}
                    step={1}
                    value={ollamaSettings.topK}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          topK: toNumber(event.target.value, previous.topK),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Min P</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={ollamaSettings.minP}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          minP: toNumber(event.target.value, previous.minP),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Repeat penalty</span>
                  <input
                    type="number"
                    min={0.5}
                    max={3}
                    step={0.01}
                    value={ollamaSettings.repeatPenalty}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          repeatPenalty: toNumber(
                            event.target.value,
                            previous.repeatPenalty,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
                <label className="field">
                  <span>Context size (num_ctx)</span>
                  <input
                    type="number"
                    min={256}
                    max={262144}
                    step={256}
                    value={ollamaSettings.contextSize}
                    onChange={(event) =>
                      setOllamaSettings((previous) =>
                        sanitizeOllamaSettings({
                          ...previous,
                          contextSize: toNumber(
                            event.target.value,
                            previous.contextSize,
                          ),
                        }),
                      )
                    }
                    disabled={isRunning}
                  />
                </label>
              </div>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={ollamaSettings.useNativeToolCalling}
                  onChange={(event) =>
                    setOllamaSettings((previous) => ({
                      ...previous,
                      useNativeToolCalling: event.target.checked,
                    }))
                  }
                  disabled={isRunning}
                />
                <span>Enable Ollama native tool calling for structured JSON</span>
              </label>

              <div className="inline-actions">
                <button
                  type="button"
                  onClick={() => setOllamaSettings(DEFAULT_OLLAMA_SETTINGS)}
                  disabled={isRunning}
                >
                  Reset Ollama Defaults
                </button>
              </div>
            </div>
          )}

          {modelError && <p className="alert warning">{modelError}</p>}

          <div className="split">
            <label className="field">
              <span>PDF concurrency</span>
              <input
                type="number"
                min={1}
                max={8}
                value={concurrency}
                onChange={(event) =>
                  setConcurrency(Math.max(1, Number(event.target.value) || 1))
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>
                {isBatchMode ? "Retries per LLM chunk (live only)" : "Retries per LLM chunk"}
              </span>
              <input
                type="number"
                min={0}
                max={6}
                value={retries}
                onChange={(event) =>
                  setRetries(Math.max(0, Number(event.target.value) || 0))
                }
                disabled={isRunning || isBatchMode}
              />
            </label>
          </div>

          {isBatchMode && (
            <p className="muted">
              Batch mode submits a JSONL file to the OpenAI Files + Batches API,
              then lets you refresh status and import results later. It does not
              issue live chat completions.
            </p>
          )}

          <label className="checkbox">
            <input
              type="checkbox"
              checked={rememberSettings}
              onChange={(event) => setRememberSettings(event.target.checked)}
              disabled={isRunning}
            />
            <span>
              Remember settings on this browser (opt-in, excludes API key)
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>2) Add PDFs and Run</h2>

          <div
            className={`dropzone ${isDragActive ? "active" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragActive(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              setIsDragActive(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragActive(false);
              onChooseFiles(event.dataTransfer.files);
            }}
          >
            <p>Drag and drop one or more PDFs here.</p>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRunning}
            >
              Browse PDFs
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={(event) => onChooseFiles(event.target.files)}
              hidden
            />
          </div>

          <div className="file-list">
            {files.length === 0 && <p className="muted">No PDFs selected yet.</p>}
            {files.map((file) => (
              <div key={`${file.name}-${file.size}-${file.lastModified}`} className="file-item">
                <div>
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024).toFixed(1)} KB</small>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(file)}
                  disabled={isRunning}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <div className="inline-actions">
            <button
              type="button"
              className="primary"
              onClick={handleRun}
              disabled={
                isRunning ||
                isEstimatingCost ||
                isBatchActionRunning ||
                files.length === 0 ||
                !baseUrl.trim()
              }
            >
              {isBatchMode ? "Submit Batch" : "Run Extraction"}
            </button>
            <button
              type="button"
              onClick={handleCancel}
              disabled={!isRunning && !isEstimatingCost}
            >
              Cancel
            </button>
          </div>

          <div className="subpanel">
            <h3>Token + Cost Estimate (tiktoken)</h3>
            <p className="muted">
              Enter model pricing in USD per 1M tokens, then estimate before running.
            </p>
            <div className="split">
              <label className="field">
                <span>Input price / 1M tokens (USD)</span>
                <input
                  type="number"
                  min={0}
                  step={0.0001}
                  value={tokenCostSettings.inputPricePer1M}
                  onChange={(event) =>
                    setTokenCostSettings((previous) =>
                      sanitizeTokenCostSettings({
                        ...previous,
                        inputPricePer1M: toNumber(
                          event.target.value,
                          previous.inputPricePer1M,
                        ),
                      }),
                    )
                  }
                  disabled={isRunning || isEstimatingCost}
                />
              </label>
              <label className="field">
                <span>Output price / 1M tokens (USD)</span>
                <input
                  type="number"
                  min={0}
                  step={0.0001}
                  value={tokenCostSettings.outputPricePer1M}
                  onChange={(event) =>
                    setTokenCostSettings((previous) =>
                      sanitizeTokenCostSettings({
                        ...previous,
                        outputPricePer1M: toNumber(
                          event.target.value,
                          previous.outputPricePer1M,
                        ),
                      }),
                    )
                  }
                  disabled={isRunning || isEstimatingCost}
                />
              </label>
              <label className="field">
                <span>Estimated output tokens per request</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={tokenCostSettings.estimatedOutputTokensPerRequest}
                  onChange={(event) =>
                    setTokenCostSettings((previous) =>
                      sanitizeTokenCostSettings({
                        ...previous,
                        estimatedOutputTokensPerRequest: toNumber(
                          event.target.value,
                          previous.estimatedOutputTokensPerRequest,
                        ),
                      }),
                    )
                  }
                  disabled={isRunning || isEstimatingCost}
                />
              </label>
              <label className="field">
                <span>Fallback vision tokens per page</span>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={tokenCostSettings.estimatedVisionImageTokens}
                  onChange={(event) =>
                    setTokenCostSettings((previous) =>
                      sanitizeTokenCostSettings({
                        ...previous,
                        estimatedVisionImageTokens: toNumber(
                          event.target.value,
                          previous.estimatedVisionImageTokens,
                        ),
                      }),
                    )
                  }
                  disabled={isRunning || isEstimatingCost}
                />
              </label>
            </div>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={tokenCostSettings.applyBatchDiscount}
                onChange={(event) =>
                  setTokenCostSettings((previous) =>
                    sanitizeTokenCostSettings({
                      ...previous,
                      applyBatchDiscount: event.target.checked,
                    }),
                  )
                }
                disabled={isRunning || isEstimatingCost}
              />
              <span>Apply batch discount</span>
            </label>

            <label className="field">
              <span>Batch discount (%)</span>
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={tokenCostSettings.batchDiscountPercent}
                onChange={(event) =>
                  setTokenCostSettings((previous) =>
                    sanitizeTokenCostSettings({
                      ...previous,
                      batchDiscountPercent: toNumber(
                        event.target.value,
                        previous.batchDiscountPercent,
                      ),
                    }),
                  )
                }
                disabled={
                  isRunning ||
                  isEstimatingCost ||
                  !tokenCostSettings.applyBatchDiscount
                }
              />
            </label>

            <div className="inline-actions">
              <button
                type="button"
                onClick={handleEstimateCost}
                disabled={
                  isRunning ||
                  isEstimatingCost ||
                  files.length === 0 ||
                  !activeModel
                }
              >
                {isEstimatingCost ? "Estimating..." : "Estimate Tokens + Cost"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setTokenCostSettings(DEFAULT_TOKEN_COST_SETTINGS);
                  setCostEstimate(null);
                }}
                disabled={isRunning || isEstimatingCost}
              >
                Reset Estimate Defaults
              </button>
            </div>

            {costEstimate && (
              <>
                <p className="muted">
                  Requests: {costEstimate.requestCount} ({costEstimate.textRequestCount} text,{" "}
                  {costEstimate.visionRequestCount} vision)
                </p>
                <p className="muted">
                  Input tokens: {costEstimate.inputTokens.toLocaleString()} | Output tokens
                  (estimated): {costEstimate.outputTokens.toLocaleString()}
                </p>
                <p className="muted">
                  Total tokens (estimated): {costEstimate.estimatedTotalTokens.toLocaleString()}
                </p>
                <p className="muted">
                  Estimated cost: <strong>${costEstimate.estimatedCostUsd.toFixed(4)}</strong>
                  {costEstimate.discountMultiplier < 1 && (
                    <span>
                      {" "}
                      (after {(100 - costEstimate.discountMultiplier * 100).toFixed(0)}% discount)
                    </span>
                  )}
                </p>
                <p className="muted">
                  Tokenizer: {costEstimate.usedTiktoken ? "tiktoken" : "fallback estimate"}
                </p>
              </>
            )}
          </div>

          {isBatchMode && (
            <div className="subpanel">
              <h3>OpenAI Batch Job</h3>
              {!batchJob && (
                <p className="muted">
                  No batch submitted in this tab session yet.
                </p>
              )}
              {batchJob && (
                <>
                  <p className="muted">
                    Batch ID: <code>{batchJob.batchId}</code>
                  </p>
                  <p className="muted">
                    Status: <strong>{batchJob.status}</strong>
                  </p>
                  <p className="muted">
                    Requests:{" "}
                    {batchJob.requestCounts
                      ? `${batchJob.requestCounts.completed}/${batchJob.requestCounts.total} complete, ${batchJob.requestCounts.failed} failed`
                      : `${batchJob.requestCount} submitted`}
                  </p>
                  <p className="muted">
                    Input size: {formatBytes(batchJob.requestBytes)}
                  </p>
                  <div className="inline-actions">
                    <button
                      type="button"
                      onClick={handleBatchRefresh}
                      disabled={isRunning || isBatchActionRunning}
                    >
                      {batchAction === "refresh" ? "Refreshing..." : "Refresh Status"}
                    </button>
                    <button
                      type="button"
                      onClick={handleBatchImport}
                      disabled={
                        isRunning ||
                        isBatchActionRunning ||
                        (!isTerminalBatchStatus(batchJob.status) &&
                          !batchJob.outputFileId)
                      }
                    >
                      {batchAction === "import" ? "Importing..." : "Import Results"}
                    </button>
                    <button
                      type="button"
                      onClick={handleBatchCancel}
                      disabled={
                        isRunning ||
                        isBatchActionRunning ||
                        isTerminalBatchStatus(batchJob.status)
                      }
                    >
                      {batchAction === "cancel" ? "Canceling..." : "Cancel Remote Batch"}
                    </button>
                    <button
                      type="button"
                      onClick={clearBatchJob}
                      disabled={isRunning || isBatchActionRunning}
                    >
                      Clear Batch
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          <div className="progress-panel">
            <p>
              Overall: {progress?.completedPdfs ?? 0}/{progress?.totalPdfs ?? 0} PDFs (
              {overallPercent}%)
            </p>
            {progress?.currentPdf && progress.currentPage > 0 && (
              <p>
                Current: {progress.currentPdf} | page {progress.currentPage}/
                {progress.totalPagesForCurrent}
              </p>
            )}
          </div>
        </section>

        <section className="panel">
          <h2>3) Download Results</h2>
          <p className="muted">
            Output columns: <code>pdf_name</code>, <code>paragraph</code>,
            <code> paragraph_index</code>, <code>page_number</code>,
            <code> section_heading</code>, <code>notes</code>,{" "}
            <code>confidence</code>.
          </p>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => downloadCsv(rows, buildDownloadName("csv"))}
              disabled={rows.length === 0}
            >
              Download CSV
            </button>
            <button
              type="button"
              onClick={() => downloadXlsx(rows, buildDownloadName("xlsx"))}
              disabled={rows.length === 0}
            >
              Download XLSX
            </button>
          </div>

          <p className="muted">Rows ready: {rows.length}</p>
        </section>

        <section className="panel full-width">
          <h2>4) Extraction Quality Knobs</h2>
          <p className="muted">
            Tune minimum paragraph quality to reduce short fragments and headings.
          </p>

          <div className="split">
            <label className="field">
              <span>Minimum words per paragraph</span>
              <input
                type="number"
                min={1}
                max={100}
                step={1}
                value={qualitySettings.minWordsPerParagraph}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      minWordsPerParagraph: toNumber(
                        event.target.value,
                        previous.minWordsPerParagraph,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>Minimum alphabetic characters</span>
              <input
                type="number"
                min={1}
                max={600}
                step={1}
                value={qualitySettings.minAlphaCharsPerParagraph}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      minAlphaCharsPerParagraph: toNumber(
                        event.target.value,
                        previous.minAlphaCharsPerParagraph,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
            <label className="field">
              <span>Short paragraph word threshold</span>
              <input
                type="number"
                min={1}
                max={200}
                step={1}
                value={qualitySettings.shortParagraphWordThreshold}
                onChange={(event) =>
                  setQualitySettings((previous) =>
                    sanitizeQualitySettings({
                      ...previous,
                      shortParagraphWordThreshold: toNumber(
                        event.target.value,
                        previous.shortParagraphWordThreshold,
                      ),
                    }),
                  )
                }
                disabled={isRunning}
              />
            </label>
          </div>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={qualitySettings.requireSentenceTerminatorForShortParagraphs}
              onChange={(event) =>
                setQualitySettings((previous) => ({
                  ...previous,
                  requireSentenceTerminatorForShortParagraphs:
                    event.target.checked,
                }))
              }
              disabled={isRunning}
            />
            <span>Require punctuation on short paragraphs</span>
          </label>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => setQualitySettings(DEFAULT_QUALITY_SETTINGS)}
              disabled={isRunning}
            >
              Reset Quality Defaults
            </button>
          </div>
        </section>

        <section className="panel full-width">
          <h2>5) Prompt Templates (Editable)</h2>
          <p className="muted">
            These prompts control inclusion/exclusion logic. Tune them to favor
            full sentence paragraphs and suppress short heading fragments.
          </p>

          <div className="inline-actions">
            <button
              type="button"
              onClick={() => promptFileInputRef.current?.click()}
              disabled={isRunning}
            >
              Import .md
            </button>
            <button
              type="button"
              onClick={handlePromptDownload}
              disabled={isRunning}
            >
              Export .md
            </button>
            <button
              type="button"
              onClick={() => {
                setPromptConfig(DEFAULT_PROMPT_CONFIG);
                appendLog("info", "Reset prompts to defaults.");
              }}
              disabled={isRunning}
            >
              Reset Defaults
            </button>
            <input
              ref={promptFileInputRef}
              type="file"
              accept=".md,text/markdown,text/plain"
              hidden
              onChange={(event) => {
                void handlePromptFileImport(event.target.files);
                event.currentTarget.value = "";
              }}
            />
          </div>

          <label className="field">
            <span>Text-layer filter prompt</span>
            <textarea
              value={promptConfig.textFilterSystem}
              onChange={(event) =>
                setPromptConfig((previous) => ({
                  ...previous,
                  textFilterSystem: event.target.value,
                }))
              }
              rows={10}
              disabled={isRunning}
            />
          </label>

          <label className="field">
            <span>Vision OCR page prompt</span>
            <textarea
              value={promptConfig.visionSystem}
              onChange={(event) =>
                setPromptConfig((previous) => ({
                  ...previous,
                  visionSystem: event.target.value,
                }))
              }
              rows={10}
              disabled={isRunning}
            />
          </label>
        </section>

        <section className="panel full-width">
          <h2>Status Log</h2>
          <div className="log-panel">
            {logs.length === 0 && <p className="muted">No status messages yet.</p>}
            {logs.map((entry) => (
              <div key={entry.id} className={`log-entry ${entry.level}`}>
                <span>[{entry.timestamp}]</span>
                <span>{entry.level.toUpperCase()}</span>
                <span>{entry.message}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
