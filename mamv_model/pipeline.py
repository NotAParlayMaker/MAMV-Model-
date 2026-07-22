"""A compact, explainable extractive document-QA pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from importlib.resources import files
import re
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset({"a", "an", "and", "are", "at", "for", "how", "in", "is", "of", "on", "the", "to", "what", "when", "where", "which", "who", "with"})


@dataclass(frozen=True)
class Citation:
    """A source passage supporting an answer."""

    document_id: str
    passage_id: int
    passage: str
    score: float


@dataclass(frozen=True)
class Answer:
    """A grounded answer and the passages used to derive it."""

    answer: str | None
    confidence: float
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class _Passage:
    document_id: str
    passage_id: int
    text: str
    terms: Counter[str]
    length: int


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOPWORDS]


class DocumentQA:
    """Index documents and answer questions with cited, extractive evidence."""

    def __init__(self, weights_path: str | None = None) -> None:
        source = files("mamv_model").joinpath("weights/mamv-base.json")
        with (open(weights_path, encoding="utf-8") if weights_path else source.open(encoding="utf-8")) as handle:
            self.weights = json.load(handle)
        self._passages: list[_Passage] = []

    def add_document(self, document_id: str, text: str) -> None:
        """Add a document, splitting it into overlapping, traceable passages."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        words = text.split()
        if not words:
            return
        size = int(self.weights["passage_words"])
        step = max(1, size - int(self.weights["passage_overlap_words"]))
        for index, start in enumerate(range(0, len(words), step)):
            chunk = " ".join(words[start : start + size]).strip()
            if not chunk:
                continue
            terms = Counter(_tokens(chunk))
            self._passages.append(_Passage(document_id, index, chunk, terms, sum(terms.values())))
            if start + size >= len(words):
                break

    def add_documents(self, documents: Iterable[tuple[str, str]]) -> None:
        """Add an iterable of ``(document_id, text)`` records."""
        for document_id, text in documents:
            self.add_document(document_id, text)

    def ask(self, question: str, top_k: int = 3) -> Answer:
        """Return an evidence-backed answer or ``None`` when evidence is weak."""
        if not self._passages:
            return Answer(None, 0.0, ())
        query = _tokens(question)
        if not query:
            return Answer(None, 0.0, ())
        ranked = self._rank(query)[: max(1, top_k)]
        citations = tuple(
            Citation(p.document_id, p.passage_id, p.text, round(score, 4))
            for score, p in ranked
        )
        best_score, best = ranked[0]
        # N-grams are a retrieval aid, but cannot by themselves ground an answer.
        # Require at least one exact content-term overlap before extracting text.
        if best_score < float(self.weights["minimum_confidence"]) or not (set(query) & set(best.terms)):
            return Answer(None, round(best_score, 4), citations)
        sentence = self._best_sentence(query, best.text)
        return Answer(sentence, round(best_score, 4), citations)

    def _rank(self, query: list[str]) -> list[tuple[float, _Passage]]:
        total = len(self._passages)
        average_length = sum(p.length for p in self._passages) / total
        document_frequency = Counter(
            term for passage in self._passages for term in passage.terms
        )
        k1, b = float(self.weights["bm25_k1"]), float(self.weights["bm25_b"])
        qset = set(query)
        ranked: list[tuple[float, _Passage]] = []
        for passage in self._passages:
            lexical = 0.0
            for term in qset:
                frequency = passage.terms[term]
                if not frequency:
                    continue
                idf = math.log(1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                lexical += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * passage.length / average_length))
            # Character trigrams recover useful matches where tokenization differs.
            query_grams = self._trigrams(" ".join(query))
            passage_grams = self._trigrams(passage.text.lower())
            ngram = len(query_grams & passage_grams) / max(1, len(query_grams))
            score = (float(self.weights["lexical_weight"]) * lexical / max(1, len(qset)) + float(self.weights["ngram_weight"]) * ngram)
            ranked.append((score, passage))
        return sorted(ranked, key=lambda item: item[0], reverse=True)

    @staticmethod
    def _trigrams(value: str) -> set[str]:
        padded = f"  {value}  "
        return {padded[i : i + 3] for i in range(max(0, len(padded) - 2))}

    @staticmethod
    def _best_sentence(query: list[str], passage: str) -> str:
        query_terms = set(query)
        sentences = [part.strip() for part in _SENTENCE.split(passage) if part.strip()]
        return max(sentences or [passage], key=lambda item: (len(query_terms & set(_tokens(item))), -len(item)))
