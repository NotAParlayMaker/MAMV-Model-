---
language: en
license: mit
library_name: transformers
tags: [document-question-answering, retrieval-augmented-generation, verification]
---
# MAMV Model

## Purpose
MAMV models support evidence-grounded document QA, retrieval-augmented QA, generic-statement and quantifier classification, claim verification, confidence scoring, and source attribution for the MAMV verification stack.

## Architecture and training
This repository is backend-neutral and uses Hugging Face `AutoModelForCausalLM`/`AutoTokenizer`; it does not bundle a base model, weights, or tokenizer. Train a licensed base checkpoint with `scripts/train.py`, optionally using PEFT LoRA or 4-bit QLoRA, Accelerate-compatible `Trainer`, mixed precision, gradient checkpointing, and resumable checkpoints. Dataset records and provenance must be documented for each released model.

## Evaluation
Report document QA exact match/F1, retrieval accuracy, calibration, verification accuracy, genericity and quantifier accuracy, hallucination rate, and slice-level results. `scripts/evaluate.py` writes JSON reports; do not claim benchmark values until measured on a disclosed evaluation set.

## Limitations and responsible AI
Outputs can be incorrect, omit context, reflect training-data bias, or mis-handle nuanced legal, medical, educational, and high-stakes claims. Require human review for consequential decisions; retain source evidence, protect private documents, and test disparate error rates. Retrieval and lexical verification are reference components, not proof engines.

## Citation
```bibtex
@software{mamv_model_2026, title={MAMV Model}, author={MAMV contributors}, year={2026}}
```

## License
MIT; every released checkpoint may impose additional base-model and dataset terms.

## Inference
```python
from mamv_model import MAMVModel
model = MAMVModel.load("org/mamv-v1")
answer = model.answer(document="...", question="...")
```
