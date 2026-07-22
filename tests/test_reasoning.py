from __future__ import annotations

from collections import deque

from mamv_model import MAMVModel
from mamv_model.config import ReasoningConfig, reasoning_answer_kwargs
from mamv_model.document_qa import Answer
from mamv_model.document_qa import DocumentQABackend
from mamv_model.reasoning import parse_cot_response, self_consistency, self_refine
from mamv_model.retrieval import InMemoryRetriever
from mamv_model.verifier import VerificationResult


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque(responses)
        self.calls: list[tuple[str, str]] = []

    def answer(self, document: str, question: str, **kwargs: object) -> Answer:
        self.calls.append((document, question))
        return Answer(self.responses.popleft())


class FakeInputIds(list[int]):
    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))


class FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, FakeInputIds]:
        assert return_tensors == "pt"
        self.prompts.append(prompt)
        return {"input_ids": FakeInputIds([1, 2, 3])}

    def decode(self, tokens: object, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "Reasoning:\n- stated evidence\nAnswer: Blue\nConfidence (0-1): 0.9"


class FakeGenerationModel:
    def generate(self, **kwargs: object) -> list[list[int]]:
        assert "input_ids" in kwargs
        return [[1, 2, 3, 4]]


def test_parse_cot_response_extracts_fields_and_handles_malformed_response():
    raw = """Reasoning:
- Find the stated color.
Assumptions:
- The document is authoritative.
Answer: Blue
Confidence (0-1): 0.75
"""
    answer, trace = parse_cot_response(raw)
    assert answer == "Blue"
    assert trace.steps == ("Find the stated color.",)
    assert trace.assumptions == ("The document is authoritative.",)
    assert trace.self_confidence == 0.75

    malformed_answer, malformed_trace = parse_cot_response("Answer: uncertain")
    assert malformed_answer == "uncertain"
    assert malformed_trace.self_confidence is None
    assert malformed_trace.steps == ()


def test_cot_sends_one_complete_prompt_to_the_real_generation_path():
    tokenizer = FakeTokenizer()
    backend = DocumentQABackend(FakeGenerationModel(), tokenizer)
    answer = backend.answer("The color is blue.", "What color?", mode="cot")
    assert answer.text == "Blue"
    assert len(tokenizer.prompts) == 1
    prompt = tokenizer.prompts[0]
    assert prompt.count("Document:\nThe color is blue.") == 1
    assert "Question: Document:" not in prompt


def test_self_consistency_selects_majority_and_retains_all_traces():
    backend = FakeBackend(
        [
            "Reasoning:\n- evidence\nAnswer: Blue\nConfidence (0-1): 0.8",
            "Reasoning:\n- evidence\nAnswer: blue\nConfidence (0-1): 0.7",
            "Reasoning:\n- evidence\nAnswer: Red\nConfidence (0-1): 0.6",
        ]
    )
    answer = self_consistency(backend, "The color is blue.", "What color?", n_samples=3)
    assert answer.text == "Blue"
    assert answer.confidence == 2 / 3
    assert answer.reasoning is not None and len(answer.reasoning.sample_traces) == 3


def test_self_refine_stops_for_no_issues_and_iterates_otherwise():
    early = FakeBackend(["Blue", "No issues found."])
    answer = self_refine(early, "The color is blue.", "What color?", max_iterations=2)
    assert answer.text == "Blue"
    assert len(early.calls) == 2
    assert answer.reasoning is not None and answer.reasoning.critiques == ("No issues found.",)

    iterative = FakeBackend(["Red", "Wrong color.", "Blue", "Missing support.", "Blue"])
    refined = self_refine(iterative, "The color is blue.", "What color?", max_iterations=2)
    assert refined.text == "Blue"
    assert len(iterative.calls) == 5
    assert refined.reasoning is not None and len(refined.reasoning.critiques) == 2


def test_mamv_answer_downgrades_and_records_ungrounded_retrieval_answer():
    model = MAMVModel(FakeBackend(["unsupported statement"]), InMemoryRetriever({"source": "blue"}))
    model.verifier = type(
        "FakeVerifier",
        (),
        {"verify": lambda self, claim, evidence: VerificationResult("not_enough_information", 0.2, ())},
    )()

    answer = model.answer("The original document", "Question")
    assert answer.confidence == 0.2
    assert answer.reasoning is not None
    assert "not well-supported" in answer.reasoning.critiques[0]


def test_reasoning_sample_count_changes_backend_runtime_calls():
    one = FakeBackend(["Answer: blue"])
    one_config = ReasoningConfig(strategy="self_consistency", num_samples=1)
    MAMVModel(one).answer("blue", "color?", **reasoning_answer_kwargs(one_config))
    three = FakeBackend(["Answer: blue", "Answer: blue", "Answer: red"])
    three_config = ReasoningConfig(strategy="self_consistency", num_samples=3)
    MAMVModel(three).answer("blue", "color?", **reasoning_answer_kwargs(three_config))
    assert len(one.calls) == 1
    assert len(three.calls) == 3
