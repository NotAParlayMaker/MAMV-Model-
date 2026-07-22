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
This repository does not bundle a base model, weights, or tokenizer. The reproducible document-QA
configuration uses [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
licensed under Apache-2.0, through Hugging Face `AutoModelForCausalLM`/`AutoTokenizer`. It trains
PEFT LoRA (rank 16, alpha 32, dropout 0.05) for one epoch with batch size 1, accumulation 1, and
256-token examples. `scripts/prepare_dataset.py` obtains SQuAD 1.1 from its published source;
SQuAD is CC BY-SA 4.0 and is converted from context/question/first answer span into the MAMV
document/question/answer schema.

`outputs/mamv/` is intentionally ignored and is an adapter-only checkpoint: adapter weights,
`adapter_config.json`, and tokenizer artifacts are saved, while the Apache-2.0 Qwen base remains
an external dependency. This keeps generated artifacts small and makes the exact base model
explicit. Use `scripts/export.py --merge-adapter` only when a standalone full-weight export is
required and storage/licensing review permits it.

## Evaluation
No checkpoint or evaluation report is committed, so this card intentionally reports no invented
exact-match or F1 value. `scripts/evaluate.py` writes a JSON report against the generated held-out
SQuAD split; record its exact-match/F1 alongside any exported checkpoint. Manual checks should
also probe unsupported questions: generation can still hallucinate, and the lexical grounding gate
only operates when a retriever supplies evidence, so it is not a factuality guarantee.

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
