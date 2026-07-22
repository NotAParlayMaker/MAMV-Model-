"""Public MAMV model API."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from .document_qa import Answer, DocumentQABackend
from .genericity import GenericityResult, estimate_genericity
from .reasoning import ReasoningTrace
from .retrieval import IntegrationBudget, Retriever, select_with_budget
from .verifier import LexicalVerifier, VerificationResult

if TYPE_CHECKING:
    from .session import ConversationSession


@dataclass(frozen=True)
class EducationAnswer:
    """Feedback-only answer shape; deliberately contains no grading fields."""

    text: str
    reasoning: ReasoningTrace
    citations: tuple[str, ...]
    stated_confidence: float | None
    consensus_confidence: float | None
    grounding_confidence: float | None
    grounding_status: str


class EducationSession:
    """A non-grading classroom façade which always exposes evidence and trace."""

    def __init__(self, model: "MAMVModel") -> None:
        self.model = model

    @staticmethod
    def _reject_grading_kwargs(kwargs: dict[str, Any]) -> None:
        forbidden = {"as_grade", "grade", "score", "pass_fail"} & set(kwargs)
        if forbidden:
            raise ValueError("EducationSession never grades student work; request feedback instead.")

    def answer(self, document: str, question: str, **kwargs: Any) -> EducationAnswer:
        self._reject_grading_kwargs(kwargs)
        answer = self.model.answer(document, question, **kwargs)
        trace = answer.reasoning or ReasoningTrace()
        verification = self.model.verifier.verify(answer.text, [document])
        consensus = answer.confidence if kwargs.get("mode") == "self_consistency" else None
        return EducationAnswer(
            text=answer.text,
            reasoning=trace,
            citations=answer.sources or ("provided document",),
            stated_confidence=trace.self_confidence,
            consensus_confidence=consensus,
            grounding_confidence=verification.confidence,
            grounding_status=verification.label,
        )

    def answer_file(self, path: str | Path, question: str, **kwargs: Any) -> EducationAnswer:
        from .ingestion import chunk_document, ingest_file

        self._reject_grading_kwargs(kwargs)
        text = ingest_file(path)
        result = self.model.answer_file(path, question, **kwargs)
        trace = result.reasoning or ReasoningTrace()
        verification = self.model.verifier.verify(result.text, [str(c) for c in chunk_document(text)])
        return EducationAnswer(
            result.text, trace, result.sources or (str(path),), trace.self_confidence,
            result.confidence if kwargs.get("mode") == "self_consistency" else None,
            verification.confidence, verification.label,
        )


class MAMVModel:
    """Composable MAMV inference façade backed by a real HF checkpoint."""

    def __init__(
        self,
        qa: DocumentQABackend,
        retriever: Retriever | None = None,
        require_grounding: bool = True,
    ) -> None:
        self.qa, self.retriever, self.verifier = qa, retriever, LexicalVerifier()
        self.require_grounding = require_grounding

    @classmethod
    def load(cls, model_id_or_path: str | None = None, **kwargs: Any) -> "MAMVModel":
        if not model_id_or_path:
            raise ValueError(
                "model_id_or_path is required; train or download a real checkpoint first."
            )
        require_grounding = kwargs.pop("require_grounding", True)
        return cls(
            DocumentQABackend.from_pretrained(model_id_or_path, **kwargs),
            require_grounding=require_grounding,
        )

    def answer(
        self, document: str, question: str, *, integration_mode: Literal["fragmented", "integrated"] = "integrated",
        integration_max_tokens: int | None = None, **kwargs: Any,
    ) -> Answer:
        sources = self.retriever.retrieve(question) if self.retriever else []
        selected = sources
        budget = None
        if sources and integration_max_tokens is not None:
            selected, budget = select_with_budget(sources, integration_max_tokens)
        context = "\n\n".join(item.text for item in selected) or document
        if integration_mode == "fragmented" and selected:
            parts = [self._ask(item.text, question, **kwargs) for item in selected]
            labels = [self.verifier.verify(part.text, [other.text for other in selected if other is not item]).label for item, part in zip(selected, parts)]
            disagreement = len({_normalise(part.text) for part in parts}) > 1 or "not_enough_information" in labels
            text = "\n".join(f"[{item.source_id}] {part.text}" for item, part in zip(selected, parts))
            if disagreement:
                text = "Per-chunk answers disagree; human review required:\n" + text
            answer = Answer(text, confidence=None, reasoning=ReasoningTrace(critiques=("Per-chunk answers disagree" if disagreement else "Per-chunk answers are lexically consistent",)))
        elif integration_mode == "integrated":
            answer = self._ask(context, question, **kwargs)
        else:
            raise ValueError(f"Unsupported integration mode: {integration_mode}")
        reasoning = answer.reasoning
        confidence = answer.confidence
        if self.retriever and self.require_grounding:
            verification = self.verifier.verify(answer.text, [item.text for item in selected])
            if verification.label == "not_enough_information":
                confidence = min(confidence if confidence is not None else 1.0, verification.confidence)
                critique = "Answer not well-supported by retrieved evidence"
                reasoning = (
                    replace(reasoning, critiques=reasoning.critiques + (critique,))
                    if reasoning
                    else ReasoningTrace(critiques=(critique,))
                )
        return Answer(answer.text, confidence, tuple(item.source_id for item in selected), reasoning, budget,
                      answer.notable_convergence, answer.notable_convergence_reason)

    def answer_file(self, path: str | Path, question: str, **kwargs: Any) -> Answer:
        """Answer from a local course reading with chunk locations as sources."""
        from .ingestion import chunk_document, ingest_file
        from .retrieval import InMemoryRetriever

        reading = ingest_file(path)
        chunks = chunk_document(reading)
        if not chunks:
            raise ValueError("The reading contains no usable text.")
        chunk_map = {
            f"{Path(path).name}{(': ' + chunk.source_location) if chunk.source_location else ''}": str(chunk)
            for chunk in chunks
        }
        # A caller-provided retriever may already index this reading; otherwise use
        # the lightweight local retriever so every file works without setup.
        selected = (
            self.retriever.retrieve(question, top_k=3)
            if self.retriever
            else InMemoryRetriever(chunk_map).retrieve(question, top_k=min(3, len(chunk_map)))
        )
        if not selected:
            selected = InMemoryRetriever(chunk_map).retrieve(question, top_k=min(3, len(chunk_map)))
        context = "\n\n".join(item.text for item in selected)
        answer = self.qa.answer(context, question, **kwargs)
        verification = self.verifier.verify(answer.text, [item.text for item in selected])
        reasoning = answer.reasoning
        confidence = answer.confidence
        if self.require_grounding and verification.label == "not_enough_information":
            confidence = min(confidence if confidence is not None else 1.0, verification.confidence)
            critique = "Answer not well-supported by selected reading passages"
            reasoning = replace(reasoning, critiques=reasoning.critiques + (critique,)) if reasoning else ReasoningTrace(critiques=(critique,))
        return Answer(answer.text, confidence, tuple(item.source_id for item in selected), reasoning)

    def education_session(self) -> EducationSession:
        return EducationSession(self)

    def conversation_session(self, document: str, *, max_tokens: int = 512) -> "ConversationSession":
        from .session import ConversationSession
        return ConversationSession(self, document, max_tokens=max_tokens)

    def _ask(self, document: str, question: str, **kwargs: Any) -> Answer:
        """Apply reasoning strategies for minimal backend test doubles as well as HF backends."""
        mode = kwargs.get("mode", "direct")
        if not isinstance(self.qa, DocumentQABackend) and mode in {"self_consistency", "self_refine"}:
            from .reasoning import self_consistency, self_refine
            copied = dict(kwargs)
            copied.pop("mode", None)
            return (self_consistency(self.qa, document, question, **copied)
                    if mode == "self_consistency" else self_refine(self.qa, document, question, **copied))
        return self.qa.answer(document, question, **kwargs)

    def verify_claim(self, claim: str, evidence: list[str]) -> VerificationResult:
        return self.verifier.verify(claim, evidence)


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


__all__ = ["Answer", "EducationAnswer", "EducationSession", "GenericityResult", "IntegrationBudget", "MAMVModel", "ReasoningTrace", "estimate_genericity"]
