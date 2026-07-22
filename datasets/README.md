# Datasets

The checked-in JSONL file is a tiny schema example, not training data. Each row may contain `document`, `question`, `answer`, `claims` (with `text`, `label`, and optional `evidence`), `genericity_label`, and `quantifier`. Use data with documented provenance, licenses, consent, and a held-out split. Do not place confidential source documents in the repository.

## Reproducible open training data

`scripts/prepare_dataset.py` downloads the [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)
extractive question-answering corpus through Hugging Face Datasets and writes ignored
`squad_train.jsonl` and `squad_validation.jsonl` files. It retains records with a non-empty
first annotated answer span and maps `context`, `question`, and that span into MAMV's
`document`, `question`, and `answer` fields. Claims and genericity fields are omitted because
SQuAD does not supply them. SQuAD is provided under **CC BY-SA 4.0**; see the dataset's
[license statement](https://rajpurkar.github.io/SQuAD-explorer/). The generated files are real
training data and intentionally ignored, unlike `sample_dataset.jsonl`.

```bash
python scripts/prepare_dataset.py --train-limit 128 --validation-limit 32
```
