"""Backend-neutral retrieval contracts and an in-memory reference backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


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
