"""Local, non-persistent course-reading ingestion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DocumentReference:
    document_id: str; name: str; media_type: str; content_hash: str; source_uri: str | None
    created_at: str | None; modified_at: str | None; metadata: Mapping[str, str]


@dataclass(frozen=True)
class DocumentCollection:
    collection_id: str; documents: tuple[DocumentReference, ...]; purpose: str | None = None; limitations: tuple[str, ...] = ()


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

    def __new__(cls, value: str, source_location: str | None = None, *, document_id: str | None = None, chunk_id: str | None = None, page: int | None = None, section: str | None = None, paragraph: int | None = None, content_hash: str | None = None) -> "DocumentChunk":
        result = super().__new__(cls, value)
        result.source_location = source_location
        result.document_id, result.chunk_id, result.page = document_id, chunk_id, page
        result.section, result.paragraph = section, paragraph
        result.content_hash = content_hash or sha256(value.encode()).hexdigest()
        return result


def ingest_documents(paths: Sequence[str | Path], *, purpose: str | None = None) -> tuple[DocumentCollection, dict[str, IngestedText]]:
    """Ingest an ordered local collection while retaining each source identity."""
    refs, texts, seen = [], {}, set()
    for raw in paths:
        path = Path(raw); text = ingest_file(path); digest = sha256(str(text).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        stat = path.stat(); doc_id = "doc-" + digest[:24]
        refs.append(DocumentReference(doc_id, path.name, path.suffix.lower().lstrip(".") or "text", digest, str(path), None, str(int(stat.st_mtime)), MappingProxyType({"filename": path.name})))
        texts[doc_id] = text
    if not refs: raise IngestionError("No documents were supplied.")
    identity = "|".join(f"{r.document_id}:{r.content_hash}" for r in refs)
    return DocumentCollection("collection-" + sha256(identity.encode()).hexdigest()[:24], tuple(refs), purpose, ("Duplicate content hashes were omitted." if len(refs) < len(paths) else "",) if len(refs) < len(paths) else ()), texts


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


def chunk_document(text: str, max_words: int = 350, *, document_id: str | None = None) -> list[str]:
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
                value = " ".join(current); chunks.append(DocumentChunk(value, location, document_id=document_id, chunk_id=f"{document_id or 'document'}-chunk-{len(chunks)+1}", page=current_page, paragraph=index, content_hash=sha256(value.encode()).hexdigest()))
                current, current_words, current_page = [], 0, None
            current.append(unit)
            current_words += words
            current_page = current_page or page
    if current:
        value = " ".join(current); chunks.append(DocumentChunk(value, f"page {current_page}" if current_page else None, document_id=document_id, chunk_id=f"{document_id or 'document'}-chunk-{len(chunks)+1}", page=current_page, paragraph=len(paragraphs), content_hash=sha256(value.encode()).hexdigest()))
    return chunks


def chunk_documents(collection: DocumentCollection, texts: Mapping[str, str], max_words: int = 350) -> list[DocumentChunk]:
    return [chunk for doc in collection.documents for chunk in chunk_document(texts[doc.document_id], max_words, document_id=doc.document_id)]
