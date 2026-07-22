"""Hugging Face AutoModel document-QA backend."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .reasoning import ReasoningTrace


@dataclass(frozen=True)
class Answer:
    text: str
    confidence: float | None = None
    sources: tuple[str, ...] = ()
    reasoning: ReasoningTrace | None = None


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

    def answer(
        self,
        document: str,
        question: str,
        *,
        max_new_tokens: int = 128,
        mode: Literal["direct", "cot", "self_consistency", "self_refine"] = "direct",
        **gen_kwargs: Any,
    ) -> Answer:
        if mode != "direct":
            from .reasoning import parse_cot_response, self_consistency, self_refine

            if mode == "cot":
                from .reasoning import build_cot_prompt

                raw = self.answer(
                    document,
                    build_cot_prompt(document, question),
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
                text, reasoning = parse_cot_response(raw.text)
                return Answer(text=text, confidence=reasoning.self_confidence, reasoning=reasoning)
            if mode == "self_consistency":
                return self_consistency(
                    self, document, question, max_new_tokens=max_new_tokens, **gen_kwargs
                )
            if mode == "self_refine":
                return self_refine(
                    self, document, question, max_new_tokens=max_new_tokens, **gen_kwargs
                )
            raise ValueError(f"Unsupported reasoning mode: {mode}")
        prompt = f"Document:\n{document}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        return Answer(text=text)
