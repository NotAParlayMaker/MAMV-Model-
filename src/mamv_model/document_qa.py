"""Hugging Face AutoModel document-QA backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Answer:
    text: str
    confidence: float | None = None
    sources: tuple[str, ...] = ()


class DocumentQABackend:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model, self.tokenizer = model, tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "DocumentQABackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return cls(
            AutoModelForCausalLM.from_pretrained(model_id, **kwargs),
            AutoTokenizer.from_pretrained(model_id, **kwargs),
        )

    def answer(self, document: str, question: str, *, max_new_tokens: int = 128) -> Answer:
        prompt = f"Document:\n{document}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        return Answer(text=text)
