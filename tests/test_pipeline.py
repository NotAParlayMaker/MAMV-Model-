import unittest

from mamv_model import DocumentQA


class DocumentQATests(unittest.TestCase):
    def setUp(self):
        self.qa = DocumentQA()
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
        self.assertIsNone(DocumentQA().ask("anything").answer)


if __name__ == "__main__":
    unittest.main()
