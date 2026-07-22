# MAMV Model

The official model-development repository for **MAMV**'s document understanding and verification stack. It provides configuration-driven training, evaluation, inference, export, and publication tooling—not invented artifacts. No model weights, checkpoints, or tokenizer files are committed here.

## MAMV integration
`MAMVModel` is the application-facing API for document QA, optional retrieval context and source IDs, genericity/quantifier analysis, and claim verification. Model backends use Hugging Face AutoModel interfaces, so deployments are not tied to any one model family.

## Layout
- `src/mamv_model/`: typed inference, retrieval, verification, genericity, metrics, and config modules.
- `configs/`: base and task-specific training configuration.
- `datasets/`: documented JSONL schema example only.
- `scripts/`: train, evaluate, inference, export, and Hub publication entry points.
- `evals/`: small checked-in evaluation fixtures and benchmark runner.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[training,publish,dev]'
```

## Train the first real MAMV model
1. Select a licensed Hugging Face base model and set `model.base_model` in a copied config (it is intentionally `null`).
2. Prepare licensed, de-identified JSONL data following `datasets/README.md`, then point `data.train_file` and `data.validation_file` at it.
3. Train with LoRA (or set `adapter.method: qlora`):
```bash
python scripts/train.py --config configs/document_qa.yaml
# Continue a run:
python scripts/train.py --config configs/document_qa.yaml --resume-from-checkpoint outputs/mamv/checkpoint-500
```
4. The Trainer writes real adapter/checkpoint artifacts to `training.output_dir` (default `outputs/mamv/`); its epoch checkpoints are under `outputs/mamv/checkpoint-*`. The final `save_model` and `tokenizer.save_pretrained` output also appears in `outputs/mamv/`. These ignored files are the first real weights/tokenizer produced by training.

## Evaluate and infer
```bash
python scripts/evaluate.py --model outputs/mamv --data evals/document_qa.jsonl --output outputs/evaluation.json
python scripts/inference.py --model outputs/mamv --document 'The office opens at 9 AM.' --question 'When does it open?'
python evals/benchmark.py
```
Evaluation emits JSON with QA exact match and F1. Extend reports with retrieval accuracy, calibration, claim verification, genericity/quantifier accuracy, and hallucination rate before release.

```python
from mamv_model import MAMVModel
model = MAMVModel.load("outputs/mamv")
answer = model.answer(document="...", question="...")
```

## Reasoning strategies
`DocumentQABackend.answer()` supports `mode="direct"` for the unchanged single-pass
path, `"cot"` for a parsed step-by-step trace, `"self_consistency"` for majority
voting across sampled traces, and `"self_refine"` for document-grounded
self-critique and revision. Select defaults in `configs/reasoning.yaml`:
`reasoning.strategy`, `reasoning.num_samples`, and
`reasoning.max_refine_iterations`. When a retriever is present,
`reasoning.require_grounding` enables a lexical evidence gate; unsupported answers
are assigned lower confidence and receive a visible critique in their reasoning
trace rather than being silently treated as grounded.

## Publish version 1.0
After validating a real export, ensure `outputs/mamv/config.json` and a model card (`outputs/mamv/README.md` or root `MODEL_CARD.md`) exist. Log in via `huggingface-cli login` or set `HF_TOKEN`, then:
```bash
python scripts/publish_to_huggingface.py --model-dir outputs/mamv --repo-id YOUR_ORG/mamv-v1 --revision v1.0
```
The script authenticates, validates required release material, creates the repo if needed, and uploads that folder at tag/revision `v1.0`. The manual GitHub workflow performs the same protected-token flow.

## Roadmap
Vision-language models, OCR, long-context models, agent verification, multi-document reasoning, education verification, evidence grounding, and genericity verification.

## License
[MIT](LICENSE). See [MODEL_CARD.md](MODEL_CARD.md) for model-use guidance.
