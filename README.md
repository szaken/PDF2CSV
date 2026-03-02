# PDF2CSV Browser App

Browser-based app for extracting **main-body paragraphs** from one or more PDFs and exporting one combined dataset as `.csv` or `.xlsx`.

## Features

- Drag/drop and file-picker support for multiple PDFs
- Backend configuration for:
  - OpenAI-compatible endpoints
  - Ollama-compatible endpoints
- Model listing from endpoint, with manual model ID fallback if listing fails
- OpenAI batch mode:
  - Build one JSONL batch input from all extraction requests
  - Submit via the OpenAI Files + Batches API
  - Refresh remote status and import completed results later
- Ollama advanced generation controls:
  - `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `num_ctx`
  - Optional native tool calling for structured extraction outputs
- Main-content filtering workflow:
  - Local PDF parsing in the browser (no app server)
  - Local pre-filtering for obvious boilerplate
  - Automatic vision-OCR fallback for image-only/scanned PDFs (page rendering + VLM)
  - LLM chunk filtering to keep main body paragraphs
  - Post-filtering to drop short/non-sentence fragments
- Exact paragraph deduplication **within each PDF**
- Bounded PDF concurrency (user configurable)
- Retry logic for transient backend failures
- Cancel in-flight processing with `AbortController`
- Progress indicators and status log panel
- Download outputs:
  - CSV
  - XLSX
- Editable prompt templates in-app with `.md` import/export
- Extraction quality knobs (minimum words, minimum alphabetic chars, short-paragraph sentence rule)
- Token + cost estimator:
  - tiktoken-based token counting for planned extraction requests
  - Per-1M token pricing inputs (input/output) and optional batch discount
  - Vision image token assumption for OCR fallback pages

## Project Structure

- `src/App.tsx`: UI and orchestration
- `src/lib/backend.ts`: backend adapters (OpenAI-compatible + Ollama-compatible)
- `src/lib/pdf.ts`: client-side PDF text extraction and boilerplate heuristics
- `src/lib/extractor.ts`: chunking, LLM filtering, retries, dedupe, pipeline
- `src/lib/prompts.ts`: default prompt templates + markdown parser/serializer
- `src/lib/settings.ts`: default/sanitized quality + Ollama generation settings
- `src/lib/exporters.ts`: CSV/XLSX file generation
- `src/lib/concurrency.ts`: bounded concurrency helper
- `src/lib/retry.ts`: retry helper
- `src/types.ts`: shared types

## Install

```bash
npm install
```

## Run Locally

```bash
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`).

## Build

```bash
npm run build
npm run preview
```

## Backend Configuration

### 1) OpenAI-compatible

1. Set `Backend type` to `OpenAI-compatible`
2. Base URL auto-defaults to `https://api.openai.com/v1/` (switching back to OpenAI resets to this default)
3. Enter API key
4. Choose `OpenAI request mode`:
   - `Live requests` for immediate extraction
   - `Batch mode` to submit all requests asynchronously for later import
5. Click `Load Models`
6. Select a model from dropdown, or fill `Manual model ID`
7. For scanned/image-only PDFs, choose a vision-capable model.

When `OpenAI request mode` is `Batch mode`:

- `Submit Batch` prepares all chunk/page requests locally, uploads one JSONL batch input file, and creates an OpenAI batch job.
- Use `Refresh Status` to poll the remote batch.
- Use `Import Results` after the batch reaches a terminal state (`completed`, `expired`, `cancelled`, or `failed` with error file output).
- Batch jobs are tracked in browser session storage for the current tab session.

When using the official OpenAI endpoint (`https://api.openai.com/v1`) in local development:

- Run the app with `npm run dev`.
- The app automatically routes OpenAI calls through a local Vite proxy (`/__proxy_openai`) to avoid browser fetch/CORS transport failures.
- If you serve a static production build, use your own backend relay/proxy for OpenAI requests.

### 2) Ollama-compatible

1. Set `Backend type` to `Ollama-compatible`
2. Base URL auto-defaults to `http://192.168.4.35:11434` (switching back to Ollama resets to this default)
3. Click `Load Models`
4. Select a model from dropdown, or fill `Manual model ID`
5. For scanned/image-only PDFs, choose a vision-capable model.
6. Optional: tune Ollama advanced settings in the UI.

### Ollama Advanced Settings

Available in the UI when backend is `Ollama-compatible`:

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repeat_penalty`
- `num_ctx` (context size)
- `native tool calling` toggle (uses tool schema for structured JSON extraction)

If tool calling is enabled but not supported by the endpoint/model, the app automatically retries without tools.

## Extraction Quality Knobs

Available in `Extraction Quality Knobs` panel:

- `Minimum words per paragraph`
- `Minimum alphabetic characters`
- `Short paragraph word threshold`
- `Require punctuation on short paragraphs`

## Token + Cost Estimation

In `2) Add PDFs and Run`, use `Token + Cost Estimate (tiktoken)`:

- Enter `Input price / 1M tokens (USD)` and `Output price / 1M tokens (USD)` from your provider pricing page.
- Set `Estimated output tokens per request`.
- Set `Estimated vision image tokens per page` for scanned/image-only PDFs.
- Optional: apply a batch discount percentage.
- Click `Estimate Tokens + Cost` before running extraction.

Notes:

- Token estimates are based on planned extraction requests (text-chunk filtering + vision fallback pages).
- OCR vision image costs are approximate and depend on your image-token assumption.
- The estimator loads `js-tiktoken` at runtime from an ESM CDN; if unavailable, it falls back to rough token estimation.

## Output Columns

Required:

- `pdf_name`
- `paragraph`

Included additional columns:

- `paragraph_index`
- `page_number`
- `section_heading`
- `notes`
- `confidence`

## Prompt Customization

- Default prompts are defined in `src/lib/prompts.ts`.
- A ready-to-edit file is included at `prompts.template.md`.
- In the app, use `Prompt Templates (Editable)` to:
  - Edit prompts directly
  - Import prompts from a `.md` file
  - Export current prompts to `.md`
  - Reset to defaults

Markdown template requires two section headers:

- `## text_filter_system`
- `## vision_page_system`

## Manual Acceptance Checklist

- [ ] Model listing works for OpenAI-compatible and Ollama-compatible backends, or manual model entry works when listing fails.
- [ ] User can add multiple PDFs and run extraction.
- [ ] Output downloads successfully as both CSV and XLSX.
- [ ] Output has one row per paragraph, with correct `pdf_name`.
- [ ] Obvious boilerplate (headers, footers, cookie notices, ads) is removed.
- [ ] Exact duplicate paragraphs within each PDF are removed.
- [ ] Quality knob changes (for example, minimum words) visibly affect which paragraphs are retained.
- [ ] Ollama advanced sampler/context settings are applied without errors.
- [ ] Errors are visible in UI log and recorded in `notes` for affected rows/chunks/files.
- [ ] Image-only PDFs are extracted via vision fallback when no text layer is present.

## Notes on Privacy and Safety

- PDFs are processed in-browser using `pdfjs-dist`.
- The app does not upload raw PDF files directly as files.
- For text-based PDFs, only extracted paragraph text chunks are sent to your configured LLM endpoint.
- For image-only PDFs, rendered page images are sent to your configured vision-capable model.
- Page images for OCR are rendered at higher resolution to improve extraction quality on scanned PDFs.
- The app does not run a server that stores PDFs or extracted data.
- API keys are not logged.
- Settings persistence is opt-in, and API key is intentionally excluded from stored settings.
