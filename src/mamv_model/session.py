"""In-memory, bounded conversation context for a single document."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .document_qa import Answer
from .reasoning import ReasoningTrace
from .retrieval import IntegrationBudget

if TYPE_CHECKING:
    from . import MAMVModel


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: Answer
    reasoning: ReasoningTrace | None
    timestamp: datetime


class ConversationSession:
    """Temporary path-dependent prompt history; it neither persists nor learns."""
    def __init__(self, model: "MAMVModel", document: str, *, max_tokens: int = 512) -> None:
        self.model, self.document, self.max_tokens = model, document, max_tokens
        self.turns: list[ConversationTurn] = []

    def ask(self, question: str, **kwargs: object) -> Answer:
        history = [f"Q: {turn.question}\nA: {turn.answer.text}" for turn in self.turns]
        kept: list[str] = []
        used = 0
        for entry in reversed(history):
            size = len(entry.split())
            if used + size <= self.max_tokens:
                kept.append(entry)
                used += size
        kept.reverse()
        dropped = len(history) - len(kept)
        prompt_document = self.document + ("\n\nPrior conversation:\n" + "\n\n".join(kept) if kept else "")
        answer = self.model.answer(prompt_document, question, **kwargs)
        budget = IntegrationBudget(self.max_tokens, used, len(kept), dropped, bool(dropped))
        categories = [critique.casefold() for turn in self.turns for critique in (turn.reasoning.critiques if turn.reasoning else ())]
        repeated = next((category for category in set(categories) if categories.count(category) >= 2), None)
        trace = answer.reasoning or ReasoningTrace()
        if repeated:
            trace = replace(trace, session_pattern_note=f"Recurring session critique (informational only): {repeated}")
        answer = replace(answer, reasoning=trace, integration_budget=budget)
        self.turns.append(ConversationTurn(question, answer, trace, datetime.now(timezone.utc)))
        return answer
