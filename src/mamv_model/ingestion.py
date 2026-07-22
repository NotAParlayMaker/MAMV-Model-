"""Local, non-persistent course-reading ingestion utilities."""

from __future__ import annotations

from pathlib import Path
import re


class IngestionError(ValueError):
    """Raised when a local reading cannot be extracted safely."""


class IngestedText(str):
    """Text plus the page where each extracted PDF paragraph originated."""

    def __new__(cls, value: str, paragraph_pages: tuple[int | None, ...] = ()) -> "IngestedText":
        result = super().__new__(cls, value)
        result.paragraph_pages = paragraph_pages
        return result


class DocumentChunk(str):
    """A string-compatible chunk with a human-reviewable source location."""

    def __new__(cls, value: str, source_location: str | None = None) -> "DocumentChunk":
        result = super().__new__(cls, value)
        result.source_location = source_location
        return result


def ingest_file(path: str | Path) -> str:
    """Extract UTF-8 text, Markdown, PDF, or DOCX without writing student data."""
    source = Path(path)
    if not source.is_file():
        raise IngestionError(f"Reading not found: {source}. Provide an existing local file.")
    suffix = source.suffix.lower()
    if suffix in {".txt", ".md"}:
        return IngestedText(source.read_text(encoding="utf-8"))
    if suffix == ".docx":
        from docx import Document

        return IngestedText("\n\n".join(p.text for p in Document(source).paragraphs if p.text.strip()))
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        paragraphs: list[str] = []
        pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if extracted:
                page_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", extracted) if p.strip()]
                paragraphs.extend(page_paragraphs)
                pages.extend([page_number] * len(page_paragraphs))
        if not paragraphs:
            raise IngestionError(
                f"{source} contains no extractable text. It may be a scanned/image-only PDF; "
                "OCR is not supported by MAMV ingestion."
            )
        return IngestedText("\n\n".join(paragraphs), tuple(pages))
    raise IngestionError(
        f"Unsupported reading type '{suffix or '(no extension)'}'. Supported types: .txt, .md, .pdf, .docx."
    )


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]


def chunk_document(text: str, max_words: int = 350) -> list[str]:
    """Chunk text near paragraph/sentence boundaries for a small-model context budget."""
    if max_words < 10:
        raise ValueError("max_words must be at least 10 so sentences can remain intact.")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", str(text)) if p.strip()]
    pages = getattr(text, "paragraph_pages", ())
    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_words = 0
    current_page: int | None = None
    for index, paragraph in enumerate(paragraphs):
        page = pages[index] if index < len(pages) else None
        units = [paragraph] if len(paragraph.split()) <= max_words else _sentences(paragraph)
        for unit in units:
            words = len(unit.split())
            if current and current_words + words > max_words:
                location = f"page {current_page}" if current_page else None
                chunks.append(DocumentChunk(" ".join(current), location))
                current, current_words, current_page = [], 0, None
            current.append(unit)
            current_words += words
            current_page = current_page or page
    if current:
        chunks.append(DocumentChunk(" ".join(current), f"page {current_page}" if current_page else None))
    return chunks
