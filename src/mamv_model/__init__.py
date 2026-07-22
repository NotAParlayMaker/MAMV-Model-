"""Public MAMV model API."""

from __future__ import annotations
from dataclasses import replace
from typing import Any
from .document_qa import Answer, DocumentQABackend
from .genericity import GenericityResult, estimate_genericity
from .reasoning import ReasoningTrace
from .retrieval import Retriever
from .verifier import LexicalVerifier, VerificationResult


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

    def answer(self, document: str, question: str, **kwargs: Any) -> Answer:
        sources = self.retriever.retrieve(question) if self.retriever else []
        context = "\n\n".join(item.text for item in sources) or document
        answer = self.qa.answer(context, question, **kwargs)
        reasoning = answer.reasoning
        confidence = answer.confidence
        if self.retriever and self.require_grounding:
            verification = self.verifier.verify(answer.text, [item.text for item in sources])
            if verification.label == "not_enough_information":
                confidence = min(confidence if confidence is not None else 1.0, verification.confidence)
                critique = "Answer not well-supported by retrieved evidence"
                reasoning = (
                    replace(reasoning, critiques=reasoning.critiques + (critique,))
                    if reasoning
                    else ReasoningTrace(critiques=(critique,))
                )
        return Answer(answer.text, confidence, tuple(item.source_id for item in sources), reasoning)

    def verify_claim(self, claim: str, evidence: list[str]) -> VerificationResult:
        return self.verifier.verify(claim, evidence)


__all__ = ["Answer", "GenericityResult", "MAMVModel", "ReasoningTrace", "estimate_genericity"]
