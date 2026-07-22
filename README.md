# MAMV-Model

**MAMV** (Multimodal-Aware, Metadata-Verified) is a lightweight, open-source
document question-answering baseline. It retrieves the most relevant passages
from structured and unstructured documents, then returns an extractive answer
with source citations so results stay grounded in the supplied material.

The repository is deliberately small and dependency-light for research,
experimentation, enterprise search prototypes, and document-analysis agents.

## Included weights

The repository ships with `mamv_model/weights/mamv-base.json`, a versioned,
deterministic set of ranking and answer-selection weights used by the default
pipeline. They are loaded automatically and are included in source
distributions. These are baseline heuristic weights—not a claim of a
fine-tuned neural checkpoint—so they are practical to inspect, reproduce, and
tune for a collection.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
mamv ask --doc handbook.txt --question "How long is parental leave?"
```

`mamv ask` accepts **PDF, DOCX, TXT/Markdown, PNG, JPEG, TIFF, and BMP**.
PDF text is extracted per page; DOCX headings and tables are retained as logical
units; image files use Tesseract OCR. Install the system `tesseract` binary for
OCR (for example, `apt install tesseract-ocr`).

## Python API

```python
from mamv_model import DocumentQA

qa = DocumentQA()
qa.add_document(
    "policy",
    "Employees receive 12 weeks of parental leave after a birth or adoption.",
)
answer = qa.ask("How much parental leave is available?")

print(answer.answer)
print(answer.citations[0].document_id, answer.citations[0].passage)
```

`ask()` returns an `Answer` with a confidence score, supporting passages, and
the exact answer span when one is found. When the evidence is weak, it returns
`None` rather than inventing an answer.

## Design

1. **Ingestion and chunking:** files are parsed into page/heading/table-aware
   parts, then split at paragraph and row boundaries with source metadata.
2. **Indexing and retrieval:** local `sentence-transformers` embeddings are
   stored in a local FAISS similarity index, then fused with BM25 keyword scores
   using reciprocal-rank fusion. Supply any object
   implementing `encode(list[str])` to `DocumentQA(embedding_model=...)` to use
   an API-backed embedding provider; read its API key from environment variables.
3. **Grounded reading:** the reader selects the most relevant sentence/span
   from retrieved evidence and exposes that evidence as a citation.

This is an extractive baseline: it deliberately says no answer when retrieved
evidence has no exact content-term overlap. There is no LLM answer generator,
so it never sends documents or API keys to a hosted service.

> **Known limitation / TODO:** scanned PDF pages need a PDF rendering OCR
> adapter. Image scans work today; a scanned PDF currently returns an actionable
> error rather than silently producing an ungrounded answer.

## Development

```bash
python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
