import unittest
from pathlib import Path
import tempfile

from mamv_model import DocumentQA
from mamv_model.documents import ingest_file


class TestEmbeddings:
    def encode(self, values):
        vectors = []
        for value in values:
            lower = value.lower()
            vectors.append([float(lower.count("parental")), float(lower.count("password")), 0.1])
        return vectors


class DocumentQATests(unittest.TestCase):
    def setUp(self):
        self.qa = DocumentQA(embedding_model=TestEmbeddings())
        self.qa.add_documents([
            ("handbook", "Employees receive 12 weeks of parental leave after a birth or adoption. Leave is paid at the regular salary."),
            ("security", "Passwords must be at least 16 characters long and use a password manager."),
        ])

    def test_returns_grounded_sentence_and_citation(self):
        result = self.qa.ask("How many weeks of parental leave are available?")
        self.assertEqual(result.answer, "Employees receive 12 weeks of parental leave after a birth or adoption.")
        self.assertEqual(result.citations[0].document_id, "handbook")
        self.assertGreater(result.confidence, 0)

    def test_returns_none_for_unrelated_question(self):
        result = self.qa.ask("What is the office aquarium schedule?")
        self.assertIsNone(result.answer)

    def test_empty_index_has_no_answer(self):
        self.assertIsNone(DocumentQA(embedding_model=TestEmbeddings()).ask("anything").answer)

    def test_hybrid_retrieval_returns_exact_field_chunk(self):
        result = self.qa.retrieve("What is the password field requirement?", top_k=1)
        self.assertEqual(result[0].document_id, "security")
        self.assertIn("16 characters", result[0].passage)


class IngestionTests(unittest.TestCase):
    def test_extracts_text_and_page_from_sample_pdf(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf is not installed; install requirements.txt")
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            _write_sample_pdf(pdf, "Policy Number: ABC-123")
            parts = ingest_file(pdf)
        self.assertEqual(parts[0].page_number, 1)
        self.assertIn("Policy Number: ABC-123", parts[0].text)


def _write_sample_pdf(path: Path, text: str) -> None:
    """Write a minimal one-page text PDF without a test-only PDF dependency."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, object_data in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + object_data + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    data.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)


if __name__ == "__main__":
    unittest.main()
