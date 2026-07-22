"""Backend-neutral retrieval contracts and an in-memory reference backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


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
