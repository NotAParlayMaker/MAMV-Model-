"""In-memory, bounded conversation context for a single document."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .document_qa import Answer
from .model_result import FrameWarning, build_inference_frame, make_frame_transition
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
        source_frame = self.turns[-1].answer.inference_frame if self.turns else None
        base = answer.inference_frame
        session_info = {"session_id": f"session-{id(self):x}", "prior_turn_count": len(self.turns), "included_turn_ids": tuple(str(i) for i in range(len(self.turns) - len(kept), len(self.turns))), "dropped_turn_ids": tuple(str(i) for i in range(dropped)), "history_token_budget": self.max_tokens}
        warnings = (FrameWarning("SESSION_TURNS_DROPPED", "Earlier conversation turns were excluded from the prompt.", "session.dropped_turn_ids", source_ids=tuple(session_info["dropped_turn_ids"])) ,) if dropped else ()
        frame = build_inference_frame(question=question, original_document=self.document, effective_context=prompt_document, selected_sources=(), dropped_sources=(), model_artifacts=base.model_artifacts if base else {}, retrieval_config=base.retrieval if base else {}, generation_config=base.inference if base else {}, reasoning_strategy=base.reasoning_strategy if base else str(kwargs.get("mode", "direct")), integration_mode=base.inference.get("integration_mode", "integrated") if base else "integrated", integration_budget=budget, grounding_config=base.grounding if base else {}, session_context=session_info, parent_frame_id=source_frame.frame_id if source_frame else None, assumptions=base.assumptions if base else (), limitations=base.limitations if base else (), extra_warnings=warnings)
        transition = make_frame_transition(source_frame, frame, "conversation_followup", answer_changed=True, explanation="Conversation history was incorporated into this answer.") if source_frame else None
        answer = replace(answer, reasoning=trace, integration_budget=budget, inference_frame=frame, frame_transition=transition)
        answer = self.model._with_provenance(answer, question, [], [], verifier_completed=False)
        self.turns.append(ConversationTurn(question, answer, trace, datetime.now(timezone.utc)))
        return answer
