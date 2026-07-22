"""Grounded document QA with hybrid retrieval and traceable citations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from importlib.resources import files
import math
import re
from typing import Iterable, Protocol

from .documents import DocumentPart, chunk_parts, ingest_file

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset({"a", "an", "and", "are", "at", "for", "how", "in", "is", "of", "on", "the", "to", "what", "when", "where", "which", "who", "with"})


def _tokens(text: str) -> list[str]:
    return [value.lower() for value in _TOKEN.findall(text) if value.lower() not in _STOPWORDS]


class EmbeddingModel(Protocol):
    """Pluggable embedding interface (OpenAI/Anthropic adapters may implement it)."""
    def encode(self, values: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddings:
    """Local embedding model, with deterministic offline fallback for resilience."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, values: list[str]) -> list[list[float]]:
        try:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            return self._model.encode(values, normalize_embeddings=True).tolist()
        except (ImportError, OSError):
            # Offline test environments remain usable without model downloads.
            return [_hashed_embedding(value) for value in values]


def _hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        vector[hash(token) % dimensions] += 1.0
    magnitude = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / magnitude for item in vector]


@dataclass(frozen=True)
class Citation:
    document_id: str
    passage_id: int
    passage: str
    score: float
    source_file: str | None = None
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class Answer:
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
    source_file: str
    page_number: int | None
    section_title: str | None


class DocumentQA:
    """Index local files/text and answer from hybrid-retrieved evidence only."""
    def __init__(self, weights_path: str | None = None, embedding_model: EmbeddingModel | None = None) -> None:
        source = files("mamv_model").joinpath("weights/mamv-base.json")
        with (open(weights_path, encoding="utf-8") if weights_path else source.open(encoding="utf-8")) as handle:
            self.weights = json.load(handle)
        self.embedding_model = embedding_model or SentenceTransformerEmbeddings()
        self._passages: list[_Passage] = []
        self._vectors: list[list[float]] = []
        self._faiss_index = None

    def add_file(self, path: str, *, ocr: bool = True) -> None:
        """Ingest PDF, DOCX, TXT/Markdown, or a supported scanned image."""
        self._add_parts(ingest_file(path, ocr=ocr), document_id=str(path))

    def add_document(self, document_id: str, text: str) -> None:
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        self._add_parts([DocumentPart(text, document_id)], document_id)

    def _add_parts(self, parts: list[DocumentPart], document_id: str) -> None:
        new: list[_Passage] = []
        start_id = len(self._passages)
        for offset, part in enumerate(chunk_parts(parts)):
            terms = Counter(_tokens(part.text))
            if terms:
                new.append(_Passage(document_id, start_id + offset, part.text, terms, sum(terms.values()),
                                    part.source_file, part.page_number, part.section_title))
        self._passages.extend(new)
        if new:
            self._vectors.extend(self.embedding_model.encode([item.text for item in new]))
            self._rebuild_faiss_index()

    def _rebuild_faiss_index(self) -> None:
        """Build a local FAISS cosine-similarity index when its optional wheel exists."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            self._faiss_index = None
            return
        vectors = np.asarray(self._vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss_index = index

    def add_documents(self, documents: Iterable[tuple[str, str]]) -> None:
        for document_id, text in documents:
            self.add_document(document_id, text)

    def retrieve(self, question: str, top_k: int = 3) -> tuple[Citation, ...]:
        query = _tokens(question)
        if not query or not self._passages:
            return ()
        lexical = self._lexical_scores(query)
        vector = self._vector_scores(question)
        # Reciprocal-rank fusion protects exact form field matches while retaining
        # semantic recall. Both component ranks are deterministic and inspectable.
        order_lex = sorted(range(len(self._passages)), key=lambda i: lexical[i], reverse=True)
        order_vec = sorted(range(len(self._passages)), key=lambda i: vector[i], reverse=True)
        fused = [0.0] * len(self._passages)
        for rank, index in enumerate(order_lex, 1): fused[index] += 1 / (60 + rank)
        for rank, index in enumerate(order_vec, 1): fused[index] += 1 / (60 + rank)
        selected = sorted(range(len(self._passages)), key=lambda i: fused[i], reverse=True)[:max(1, top_k)]
        return tuple(self._citation(self._passages[index], fused[index]) for index in selected)

    def ask(self, question: str, top_k: int = 3) -> Answer:
        citations = self.retrieve(question, top_k)
        if not citations:
            return Answer(None, 0.0, ())
        best = citations[0]
        overlap = set(_tokens(question)) & set(_tokens(best.passage))
        # Require lexical evidence before answering; vector-only similarity is not grounding.
        if not overlap:
            return Answer(None, best.score, citations)
        return Answer(self._best_sentence(_tokens(question), best.passage), best.score, citations)

    def build_prompt(self, question: str, top_k: int = 3) -> str:
        """Create a grounded prompt for an optional external answer generator.

        The default reader remains extractive. Callers using an LLM should pass
        this prompt to their provider and obtain credentials from environment
        variables, never from source code.
        """
        citations = self.retrieve(question, top_k)
        context = "\n\n".join(
            f"[{item.source_file or item.document_id}; page {item.page_number or 'unknown'}; "
            f"section {item.section_title or 'unknown'}]\n{item.passage}"
            for item in citations
        )
        return ("Answer only from the supplied context. If the answer is absent, say "
                "'not found in document'. Include the bracketed source citation.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}")

    def _lexical_scores(self, query: list[str]) -> list[float]:
        total = len(self._passages)
        average = sum(item.length for item in self._passages) / total
        df = Counter(term for passage in self._passages for term in passage.terms)
        k1, b = float(self.weights["bm25_k1"]), float(self.weights["bm25_b"])
        scores = []
        for passage in self._passages:
            score = 0.0
            for term in set(query):
                frequency = passage.terms[term]
                if frequency:
                    idf = math.log(1 + (total - df[term] + .5) / (df[term] + .5))
                    score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * passage.length / average))
            scores.append(score)
        return scores

    def _vector_scores(self, question: str) -> list[float]:
        query = self.embedding_model.encode([question])[0]
        if self._faiss_index is not None:
            import numpy as np
            scores, indices = self._faiss_index.search(np.asarray([query], dtype="float32"), len(self._passages))
            ordered = [0.0] * len(self._passages)
            for score, index in zip(scores[0], indices[0]):
                if index >= 0:
                    ordered[int(index)] = float(score)
            return ordered
        return [sum(a * b for a, b in zip(query, vector)) for vector in self._vectors]

    @staticmethod
    def _citation(passage: _Passage, score: float) -> Citation:
        return Citation(passage.document_id, passage.passage_id, passage.text, round(score, 4),
                        passage.source_file, passage.page_number, passage.section_title)

    @staticmethod
    def _best_sentence(query: list[str], passage: str) -> str:
        terms = set(query)
        sentences = [item.strip() for item in _SENTENCE.split(passage) if item.strip()]
        return max(sentences or [passage], key=lambda item: (len(terms & set(_tokens(item))), -len(item)))
