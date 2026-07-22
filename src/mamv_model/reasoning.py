"""Backend-agnostic, inspectable reasoning strategies for document QA."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

from .document_qa import Answer


@dataclass(frozen=True)
class ReasoningTrace:
    """Structured model-reported reasoning, uncertainty, and self-critique."""

    steps: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    critiques: tuple[str, ...] = ()
    self_confidence: float | None = None
    sample_traces: tuple["ReasoningTrace", ...] = ()
    verification_label: str | None = None
    coherence_score: float | None = None
    session_pattern_note: str | None = None
    notable_convergence: bool = False
    notable_convergence_reason: str | None = None


class AnswerBackend(Protocol):
    def answer(self, document: str, question: str, **kwargs: Any) -> Answer: ...


def build_cot_prompt(document: str, question: str) -> str:
    """Build a deliberately constrained prompt that can be parsed reliably."""
    return (
        f"Document:\n{document}\n\nQuestion: {question}\n\n"
        "Respond using exactly these labelled sections:\n"
        "Reasoning:\n"
        "- one step per line\n"
        "Assumptions:\n"
        "- one assumption per line, or - none\n"
        "Answer: final answer only\n"
        "Confidence (0-1): decimal from 0 to 1"
    )


_SECTION_PATTERN = re.compile(
    r"^\s*(Reasoning|Assumptions|Answer|Confidence\s*(?:\(0-1\))?)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _section(raw_text: str, name: str) -> str | None:
    matches = list(_SECTION_PATTERN.finditer(raw_text))
    for index, match in enumerate(matches):
        if match.group(1).strip().lower().startswith(name.lower()):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
            return raw_text[match.end() : end].strip()
    return None


def _items(section: str | None) -> tuple[str, ...]:
    if not section:
        return ()
    items = tuple(
        line.strip().lstrip("-• ").strip()
        for line in section.splitlines()
        if line.strip().lstrip("-• ").strip()
    )
    return () if len(items) == 1 and items[0].lower() in {"none", "n/a"} else items


def parse_cot_response(raw_text: str) -> tuple[str, ReasoningTrace]:
    """Parse a CoT response defensively; malformed responses still yield an answer."""
    answer = _section(raw_text, "answer")
    if answer is None:
        # A non-conforming model response is still useful as a direct answer.
        answer = raw_text.strip()
    confidence_text = _section(raw_text, "confidence")
    confidence_match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", confidence_text or "")
    confidence = float(confidence_match.group(0)) if confidence_match else None
    return answer.strip(), ReasoningTrace(
        steps=_items(_section(raw_text, "reasoning")),
        assumptions=_items(_section(raw_text, "assumptions")),
        self_confidence=confidence,
    )


def _normalise_answer(answer: str) -> str:
    return " ".join(answer.casefold().split())


def self_consistency(
    backend: AnswerBackend,
    document: str,
    question: str,
    n_samples: int = 5,
    **gen_kwargs: Any,
) -> Answer:
    """Sample multiple CoT answers and select the answer with most agreement."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    prompt = build_cot_prompt(document, question)
    parsed = [parse_cot_response(backend.answer(document, prompt, **gen_kwargs).text) for _ in range(n_samples)]
    normalised = [_normalise_answer(answer) for answer, _ in parsed]
    counts = Counter(normalised)
    winner_key = max(
        counts, key=lambda normalised_answer: counts[normalised_answer]
    )  # Ties deliberately retain first-seen order.
    winner_index = normalised.index(winner_key)
    winner_answer, winner_trace = parsed[winner_index]
    weak = (__import__("mamv_model.verifier", fromlist=["LexicalVerifier"]).LexicalVerifier()
            .verify(winner_answer, [document]).label == "not_enough_information")
    notable = counts[winner_key] > 1 and weak
    reason = "Independent samples agreed on content not found in supplied evidence; review for confident hallucination." if notable else None
    return Answer(
        text=winner_answer,
        confidence=counts[winner_key] / n_samples,
        reasoning=ReasoningTrace(
            steps=winner_trace.steps,
            assumptions=winner_trace.assumptions,
            critiques=winner_trace.critiques,
            self_confidence=winner_trace.self_confidence,
            sample_traces=tuple(trace for _, trace in parsed),
            coherence_score=_sample_coherence(parsed),
            notable_convergence=notable,
            notable_convergence_reason=reason,
        ),
        notable_convergence=notable, notable_convergence_reason=reason,
    )


def _sample_coherence(parsed: list[tuple[str, ReasoningTrace]]) -> float | None:
    """Use sample-answer lexical vectors only when no hidden states are available."""
    hidden = [t.coherence_score for _, t in parsed if t.coherence_score is not None]
    return sum(hidden) / len(hidden) if hidden else None


def self_refine(
    backend: AnswerBackend,
    document: str,
    question: str,
    max_iterations: int = 2,
    critique_prompt_fn: Callable[[str, str, str], str] | None = None,
    **gen_kwargs: Any,
) -> Answer:
    """Iteratively ask a backend to find and correct unsupported answer content."""
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")
    candidate = backend.answer(document, question, **gen_kwargs)
    critiques: list[str] = []
    steps = list(candidate.reasoning.steps if candidate.reasoning else ())
    if candidate.text:
        steps.append(candidate.text)
    for _ in range(max_iterations):
        critique_prompt = (
            critique_prompt_fn(document, question, candidate.text)
            if critique_prompt_fn
            else (
                "What is wrong, unsupported, or missing in this answer, given the document?\n\n"
                f"Document:\n{document}\n\nQuestion: {question}\n\nAnswer to critique:\n{candidate.text}"
            )
        )
        critique = backend.answer(document, critique_prompt, **gen_kwargs).text.strip()
        critiques.append(critique)
        if re.search(r"\bno issues?(?: found)?\b", critique, re.IGNORECASE):
            break
        revision_prompt = (
            f"Revise the answer using this critique and only the document evidence.\n\n"
            f"Document:\n{document}\n\nQuestion: {question}\n\n"
            f"Previous answer:\n{candidate.text}\n\nCritique:\n{critique}\n\nRevised answer:"
        )
        candidate = backend.answer(document, revision_prompt, **gen_kwargs)
        if candidate.text:
            steps.append(candidate.text)
    prior = candidate.reasoning
    return Answer(
        text=candidate.text,
        confidence=candidate.confidence,
        sources=candidate.sources,
        reasoning=ReasoningTrace(
            steps=tuple(steps),
            assumptions=prior.assumptions if prior else (),
            critiques=tuple(critiques),
            self_confidence=prior.self_confidence if prior else None,
            sample_traces=prior.sample_traces if prior else (),
        ),
    )


def critique_claim(
    backend: AnswerBackend,
    document: str,
    claim: str,
    max_iterations: int = 2,
    **gen_kwargs: Any,
) -> ReasoningTrace:
    """Give evidence feedback on student writing, without ever revising it."""
    if max_iterations < 0:
        raise ValueError("max_iterations must not be negative")
    from .verifier import LexicalVerifier

    verification = LexicalVerifier().verify(claim, [document])
    feedback_label = {
        "supported": "supported",
        "refuted": "contradicted",
        "not_enough_information": "unsupported",
    }[verification.label]
    critiques: list[str] = []
    for _ in range(max_iterations):
        prompt = (
            "Identify unsupported, contradicted, or missing-evidence parts of this student claim. "
            "Give feedback only; do not rewrite the claim.\n\n"
            f"Document:\n{document}\n\nStudent claim:\n{claim}"
        )
        critique = backend.answer(document, prompt, **gen_kwargs).text.strip()
        critiques.append(critique)
        if re.search(r"\bno issues?(?: found)?\b", critique, re.IGNORECASE):
            break
    return ReasoningTrace(
        steps=("Student submission retained without revision.",),
        critiques=tuple(critiques),
        self_confidence=verification.confidence,
        verification_label=feedback_label,
    )
