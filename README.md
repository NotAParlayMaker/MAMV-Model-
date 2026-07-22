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
python -m mamv_model --document handbook.txt --question "How long is parental leave?"
```

For PDF input, install the optional PDF reader:

```bash
pip install -e '.[pdf]'
python -m mamv_model --document contract.pdf --question "When does this agreement end?"
```

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

1. **Ingestion:** plain text, Markdown, HTML-like text, and optional PDFs are
   converted to normalized passages while retaining document and passage IDs.
2. **Retrieval:** weighted BM25-style lexical matching plus character n-gram
   similarity finds relevant passages without external services.
3. **Grounded reading:** the reader selects the most relevant sentence/span
   from retrieved evidence and exposes that evidence as a citation.

This is an extractive baseline. Future work can swap in dense retrieval and a
fine-tuned generative reader while retaining the same citation-oriented API.

## Development

```bash
python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
