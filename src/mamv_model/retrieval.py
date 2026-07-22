"""Backend-neutral retrieval contracts and an in-memory reference backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class IntegrationBudget:
    """Inspectable token-budget accounting for prompt context selection."""

    max_tokens: int
    tokens_used: int
    chunks_included: int
    chunks_dropped: int
    truncated: bool


def select_with_budget(items: list["RetrievedDocument"], max_tokens: int) -> tuple[list["RetrievedDocument"], IntegrationBudget]:
    """Keep ranked chunks that fit a whitespace-token budget; never silently truncate."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    kept: list[RetrievedDocument] = []
    used = 0
    for item in items:
        tokens = len(item.text.split())
        if used + tokens <= max_tokens:
            kept.append(item)
            used += tokens
    dropped = len(items) - len(kept)
    return kept, IntegrationBudget(max_tokens, used, len(kept), dropped, bool(dropped))


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    source_id: str
    score: float
    document_id: str | None = None
    source_type: str | None = None
    modified_at: str | None = None
    content_hash: str | None = None
    source_location: str | None = None


@dataclass(frozen=True)
class RetrievalDiversitySettings:
    top_k: int = 6; max_chunks_per_document: int = 3; min_documents: int = 2; deduplicate: bool = True


def select_diverse(items: Sequence[RetrievedDocument], settings: RetrievalDiversitySettings, *, source_types: set[str] | None = None, prefer_recent: bool = False) -> tuple[list[RetrievedDocument], list[RetrievedDocument], tuple[str, ...]]:
    """Apply bounded diversity rules; scores rank candidates but never assert support."""
    ranked = sorted(items, key=lambda x: (x.score, x.modified_at or "") if prefer_recent else (x.score,), reverse=True)
    selected, dropped, decisions, counts, hashes = [], [], [], {}, set()
    eligible = [x for x in ranked if not source_types or x.source_type in source_types]
    for item in eligible:
        doc = item.document_id or item.source_id
        digest = item.content_hash or item.text
        if settings.deduplicate and digest in hashes:
            dropped.append(item); decisions.append(f"dropped duplicate chunk {item.source_id}"); continue
        if counts.get(doc, 0) >= settings.max_chunks_per_document:
            dropped.append(item); decisions.append(f"dropped {item.source_id}: per-document cap"); continue
        if len(selected) >= settings.top_k:
            dropped.append(item); decisions.append(f"dropped {item.source_id}: top_k"); continue
        selected.append(item); hashes.add(digest); counts[doc] = counts.get(doc, 0) + 1; decisions.append(f"selected {item.source_id}")
    documents = {x.document_id or x.source_id for x in selected}
    if len(documents) < settings.min_documents:
        decisions.append(f"minimum document diversity not met ({len(documents)}/{settings.min_documents})")
    return selected, dropped, tuple(decisions)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]: ...


class InMemoryRetriever:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        terms = set(query.lower().split())
        ranked = [
            RetrievedDocument(
                text, key, len(terms & set(text.lower().split())) / max(len(terms), 1)
            )
            for key, text in self.documents.items()
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]
