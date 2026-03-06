import * as XLSX from "xlsx";
import type { ExtractionRow } from "../types";

const EXPORT_COLUMNS: Array<keyof ExtractionRow> = [
  "pdf_name",
  "url",
  "paragraph",
  "paragraph_index",
  "page_number",
  "section_heading",
  "notes",
  "confidence",
];

function toCsvCell(value: string | number | null): string {
  if (value === null) return "";
  const text = String(value);
  if (/["\n,]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function rowsToCsv(rows: ExtractionRow[]): string {
  const header = EXPORT_COLUMNS.join(",");
  const body = rows
    .map((row) =>
      EXPORT_COLUMNS.map((column) => toCsvCell(row[column])).join(","),
    )
    .join("\n");
  return `${header}\n${body}`;
}

function triggerDownload(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function downloadCsv(rows: ExtractionRow[], fileName: string): void {
  const csvText = rowsToCsv(rows);
  const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
  triggerDownload(blob, fileName);
}

export function downloadXlsx(rows: ExtractionRow[], fileName: string): void {
  const worksheet = XLSX.utils.json_to_sheet(rows, { header: EXPORT_COLUMNS });
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "paragraphs");
  const arrayBuffer = XLSX.write(workbook, {
    type: "array",
    bookType: "xlsx",
  });
  const blob = new Blob([arrayBuffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  triggerDownload(blob, fileName);
}
