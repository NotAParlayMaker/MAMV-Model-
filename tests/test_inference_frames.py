from collections import deque
from mamv_model import MAMVModel
from mamv_model.document_qa import Answer
from mamv_model.model_result import compare_inference_frames
from mamv_model.retrieval import InMemoryRetriever

class Backend:
    def __init__(self, values=("blue",)): self.values=deque(values)
    def answer(self, document, question, **kwargs): return Answer(self.values.popleft())

def test_answer_and_result_have_deterministic_canonical_frames():
    a=MAMVModel(Backend(), InMemoryRetriever({"x":"blue"})).answer("doc", "q")
    b=MAMVModel(Backend(), InMemoryRetriever({"x":"blue"})).answer("doc", "q")
    assert a.inference_frame and a.inference_frame.frame_id == b.inference_frame.frame_id
    assert a.inference_frame.created_at != "" and compare_inference_frames(a.inference_frame,b.inference_frame).identical

def test_context_loss_and_refinement_transition_are_visible():
    model=MAMVModel(Backend(("blue","issue","blue")), InMemoryRetriever({"x":"blue", "y":"red"}), require_grounding=False)
    answer=model.answer("doc", "q", integration_max_tokens=1, mode="self_refine", max_iterations=1)
    assert any(w.code == "CONTEXT_TRUNCATED" for w in answer.inference_frame.warnings)
    assert answer.frame_transition and answer.frame_transition.transition_type == "self_refinement"

def test_conversation_parent_and_dropped_turn_warning():
    session=MAMVModel(Backend(("one","two")), require_grounding=False).conversation_session("doc", max_tokens=1)
    first=session.ask("one")
    second=session.ask("two")
    assert second.inference_frame.session["parent_frame_id"] == first.inference_frame.frame_id
