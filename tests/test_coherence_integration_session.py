from collections import deque

from mamv_model import MAMVModel
from mamv_model.coherence import HiddenStateTrajectory, compute_coherence_score
from mamv_model.document_qa import Answer
from mamv_model.document_qa import DocumentQABackend
from mamv_model.reasoning import ReasoningTrace
from mamv_model.retrieval import InMemoryRetriever, RetrievedDocument, select_with_budget


class FakeBackend:
    def __init__(self, answers): self.answers, self.calls = deque(answers), []
    def answer(self, document, question, **kwargs):
        self.calls.append((document, question))
        return Answer(self.answers.popleft())


def test_coherence_is_deterministic_for_known_hidden_vectors():
    trajectory = HiddenStateTrajectory(((1, 0), (1, 0), (0, 1)))
    assert compute_coherence_score(trajectory) == 0.75


def test_backend_captures_known_fake_hidden_states():
    class Tokenizer:
        def __call__(self, prompt, return_tensors): return {"input_ids": type("Ids", (list,), {"shape": property(lambda self: (1, len(self)))}) ([1])}
        def decode(self, tokens, skip_special_tokens): return "answer"
    class Output:
        sequences = [[1, 2]]
        hidden_states = [[1, 0], [1, 0]]
    class Model:
        def generate(self, **kwargs): return Output()
    answer = DocumentQABackend(Model(), Tokenizer()).answer("doc", "q")
    assert answer.reasoning and answer.reasoning.coherence_score == 1.0


def test_integration_budget_reports_dropped_chunks():
    chunks = [RetrievedDocument("one two", "a", 1), RetrievedDocument("three four", "b", 0)]
    kept, budget = select_with_budget(chunks, 2)
    assert [item.source_id for item in kept] == ["a"]
    assert budget.chunks_dropped == 1 and budget.truncated


def test_fragmented_answers_surface_disagreement():
    model = MAMVModel(FakeBackend(["Blue", "Red"]), InMemoryRetriever({"a": "blue", "b": "red"}), require_grounding=False)
    answer = model.answer("ignored", "color", integration_mode="fragmented")
    assert "disagree" in answer.text.lower() and "Blue" in answer.text and "Red" in answer.text


def test_session_adds_prior_turn_to_second_prompt_and_notes_repeated_critiques():
    backend = FakeBackend(["first", "second", "third", "fourth"])
    session = MAMVModel(backend).conversation_session("rules", max_tokens=100)
    for question in ("q1", "q2", "q3"):
        result = session.ask(question)
        session.turns[-1] = session.turns[-1].__class__(question, result, ReasoningTrace(critiques=("Overgeneralizing",)), session.turns[-1].timestamp)
    fourth = session.ask("exception mentioned above?")
    assert "Q: q1\nA: first" in backend.calls[-1][0]
    assert fourth.reasoning and fourth.reasoning.session_pattern_note


def test_notable_convergence_stays_downgraded_by_grounding_gate():
    model = MAMVModel(FakeBackend(["Mars has canals", "Mars has canals"]), InMemoryRetriever({"evidence": "blue sky"}))
    answer = model.answer("blue sky", "what?", mode="self_consistency", n_samples=2)
    assert answer.notable_convergence
    assert answer.confidence == 0.0
