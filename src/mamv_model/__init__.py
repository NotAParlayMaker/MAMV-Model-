"""Public MAMV model API."""

from __future__ import annotations
from typing import Any
from .document_qa import Answer, DocumentQABackend
from .genericity import GenericityResult, estimate_genericity
from .retrieval import Retriever
from .verifier import LexicalVerifier, VerificationResult


class MAMVModel:
    """Composable MAMV inference façade backed by a real HF checkpoint."""

    def __init__(self, qa: DocumentQABackend, retriever: Retriever | None = None) -> None:
        self.qa, self.retriever, self.verifier = qa, retriever, LexicalVerifier()

    @classmethod
    def load(cls, model_id_or_path: str | None = None, **kwargs: Any) -> "MAMVModel":
        if not model_id_or_path:
            raise ValueError(
                "model_id_or_path is required; train or download a real checkpoint first."
            )
        return cls(DocumentQABackend.from_pretrained(model_id_or_path, **kwargs))

    def answer(self, document: str, question: str, **kwargs: Any) -> Answer:
        sources = self.retriever.retrieve(question) if self.retriever else []
        context = "\n\n".join(item.text for item in sources) or document
        answer = self.qa.answer(context, question, **kwargs)
        return Answer(answer.text, answer.confidence, tuple(item.source_id for item in sources))

    def verify_claim(self, claim: str, evidence: list[str]) -> VerificationResult:
        return self.verifier.verify(claim, evidence)


__all__ = ["Answer", "GenericityResult", "MAMVModel", "estimate_genericity"]
