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
    integration_budget: Any | None = None
    notable_convergence: bool = False
    notable_convergence_reason: str | None = None
    inference_frame: Any | None = None
    frame_transition: Any | None = None
    document_sources: tuple[Any, ...] = ()
    contradiction_candidates: tuple[Any, ...] = ()
    source_agreement_summary: str | None = None
    synthesis_mode: str | None = None
    decision_provenance: Any | None = None
    operation_records: tuple[Any, ...] = ()


class DocumentQABackend:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model, self.tokenizer = model, tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> "DocumentQABackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from pathlib import Path

        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        # A LoRA export contains adapter_config.json rather than full model weights.
        # Reconstruct it with its declared base model so ``outputs/mamv`` is directly usable.
        if (Path(model_id) / "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel

            adapter_config = PeftConfig.from_pretrained(model_id)
            if not adapter_config.base_model_name_or_path:
                raise ValueError("LoRA adapter has no base_model_name_or_path")
            base = AutoModelForCausalLM.from_pretrained(
                adapter_config.base_model_name_or_path, **kwargs
            )
            return cls(PeftModel.from_pretrained(base, model_id), tokenizer)
        return cls(AutoModelForCausalLM.from_pretrained(model_id, **kwargs), tokenizer)

    def _generate_prompt(self, prompt: str, *, max_new_tokens: int, **gen_kwargs: Any) -> Answer:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        capture_hidden = gen_kwargs.pop("output_hidden_states", True)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=capture_hidden,
            output_hidden_states=capture_hidden, **gen_kwargs
        )
        sequences = getattr(output, "sequences", output)
        text = self.tokenizer.decode(
            sequences[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        trace = None
        states = _hidden_vectors(getattr(output, "hidden_states", None)) if capture_hidden else []
        if states:
            from .coherence import HiddenStateTrajectory, compute_coherence_score
            from .reasoning import ReasoningTrace
            trace = ReasoningTrace(coherence_score=compute_coherence_score(HiddenStateTrajectory(states)))
        return Answer(text=text, reasoning=trace)

    def _direct_prompt(self, document: str, question: str) -> str:
        """Use an instruct tokenizer's native format when it advertises one."""
        user = f"Document:\n{document}\n\nQuestion: {question}"
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Answer using only the document evidence."},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{user}\nAnswer:"

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

                # build_cot_prompt is already a complete prompt.  Do not pass it
                # through the direct template, which would duplicate the document.
                raw = self._generate_prompt(
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
        prompt = self._direct_prompt(document, question)
        return self._generate_prompt(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)


def _hidden_vectors(hidden_states: Any) -> list[list[float]]:
    """Best-effort extraction for HF generation's nested hidden-state layout."""
    if hidden_states is None:
        return []
    result: list[list[float]] = []
    for step in hidden_states:
        value = step if isinstance(step, (tuple, list)) and step and isinstance(step[0], (int, float)) else (step[-1] if isinstance(step, (tuple, list)) else step)
        try:
            value = value[0, -1].detach().cpu().tolist()
        except (AttributeError, IndexError, TypeError):
            try:
                if isinstance(value[0], (int, float)):
                    pass
                else:
                    value = value[0][-1]
            except (IndexError, TypeError):
                continue
        try:
            result.append([float(x) for x in value])
        except TypeError:
            continue
    return result
