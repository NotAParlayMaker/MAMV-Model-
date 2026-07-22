from __future__ import annotations

from collections import deque
from dataclasses import fields

import pytest

from mamv_model import EducationAnswer, MAMVModel
from mamv_model.document_qa import Answer
from mamv_model.ingestion import IngestionError, chunk_document, ingest_file
from mamv_model.reasoning import critique_claim


class FeedbackBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque(responses)
        self.prompts: list[str] = []

    def answer(self, document: str, question: str, **kwargs: object) -> Answer:
        self.prompts.append(question)
        return Answer(self.responses.popleft())


def _pdf(path):
    # A deliberately tiny text PDF, avoiding a network dependency or a renderer.
    content = b"BT /F1 12 Tf 72 720 Td (The river is blue.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(f"{x:010d} 00000 n \n".encode() for x in offsets)
    data += b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(start).encode() + b"\n%%EOF\n"
    path.write_bytes(data)


def test_ingestion_txt_pdf_chunking_and_clear_errors(tmp_path):
    reading = tmp_path / "reading.txt"
    reading.write_text("First sentence.\n\nSecond paragraph stays together.")
    assert "First sentence" in ingest_file(reading)
    chunks = chunk_document(ingest_file(reading), max_words=10)
    assert [str(c) for c in chunks] == ["First sentence. Second paragraph stays together."]
    pytest.importorskip("pypdf")
    pdf = tmp_path / "reading.pdf"
    _pdf(pdf)
    pdf_chunks = chunk_document(ingest_file(pdf))
    assert str(pdf_chunks[0]) == "The river is blue."
    assert pdf_chunks[0].source_location == "page 1"
    unsupported = tmp_path / "reading.epub"
    unsupported.write_text("x")
    with pytest.raises(IngestionError, match="Supported types"):
        ingest_file(unsupported)


def test_answer_file_supplies_chunk_location_without_a_model(tmp_path):
    reading = tmp_path / "reading.txt"
    reading.write_text("The library opens at nine AM.")
    result = MAMVModel(FeedbackBackend(["Nine AM."])).answer_file(reading, "When does it open?")
    assert result.text == "Nine AM."
    assert result.sources == ("reading.txt",)


def test_student_claim_feedback_does_not_rewrite_and_is_verified():
    backend = FeedbackBackend(["The document does not establish that every student wins."])
    claim = "The river is blue and every student wins."
    trace = critique_claim(backend, "The river is blue.", claim, max_iterations=1)
    assert trace.verification_label == "unsupported"
    assert trace.critiques and claim not in trace.critiques[0]
    assert all(claim not in prompt or "Student claim" in prompt for prompt in backend.prompts)
    supported = critique_claim(FeedbackBackend(["No issues found."]), "The river is blue.", "The river is blue.", max_iterations=1)
    assert supported.verification_label == "supported"


def test_genericity_is_available_for_student_sentences():
    from mamv_model import estimate_genericity
    assert estimate_genericity("Every student finished the reflection.").is_generic


def test_education_shape_has_no_grading_and_always_has_trace_and_citation():
    session = MAMVModel(FeedbackBackend(["Reasoning:\n- evidence\nAnswer: blue\nConfidence (0-1): 0.8"])).education_session()
    result = session.answer("The river is blue.", "What color is the river?", mode="cot")
    names = {field.name for field in fields(EducationAnswer)}
    assert not names & {"grade", "letter_grade", "score", "pass_fail", "points"}
    assert result.reasoning is not None and result.citations
    assert {"stated_confidence", "consensus_confidence", "grounding_confidence"} <= names
    with pytest.raises(ValueError, match="never grades"):
        session.answer("doc", "question", as_grade=True)
