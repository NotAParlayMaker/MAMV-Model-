# MAMV Model

The official model-development repository for **MAMV**'s document understanding and verification stack. It provides configuration-driven training, evaluation, inference, export, and publication tooling—not invented artifacts. No model weights, checkpoints, or tokenizer files are committed here.

## MAMV integration
`MAMVModel` is the application-facing API for document QA, optional retrieval context and source IDs, genericity/quantifier analysis, and claim verification. Model backends use Hugging Face AutoModel interfaces, so deployments are not tied to any one model family.

## Layout
- `src/mamv_model/`: typed inference, retrieval, verification, genericity, metrics, and config modules.
- `configs/`: base and task-specific training configuration.
- `datasets/`: schema fixture plus an ignored, reproducible open-data preparation path.
- `scripts/`: train, evaluate, inference, export, and Hub publication entry points.
- `evals/`: small checked-in evaluation fixtures and benchmark runner.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[training,publish,dev]'
```

## Train the first real MAMV model
1. `configs/document_qa.yaml` selects `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) while
   `configs/base.yaml` deliberately remains unset as the safe default.
2. Prepare the documented SQuAD 1.1 (CC BY-SA 4.0) subset; the generated JSONL files are
   ignored and never committed:
```bash
python scripts/prepare_dataset.py --train-limit 128 --validation-limit 32
```
3. Train with LoRA (or set `adapter.method: qlora`):
```bash
python scripts/train.py --config configs/document_qa.yaml
# Continue a run:
python scripts/train.py --config configs/document_qa.yaml --resume-from-checkpoint outputs/mamv/checkpoint-500
```
4. The Trainer writes a **LoRA adapter-only** checkpoint to `training.output_dir` (default
   `outputs/mamv/`); its epoch checkpoints are under `outputs/mamv/checkpoint-*`. The final
   folder contains `adapter_config.json`, adapter weights, and tokenizer files. It requires the
   declared Qwen base model at inference time; `MAMVModel.load` reconstructs that pairing.

## Evaluate and infer
```bash
python scripts/evaluate.py --model outputs/mamv --data datasets/squad_validation.jsonl --output outputs/evaluation.json
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

Both inference commands accept `--reasoning-config configs/reasoning.yaml`; its strategy,
sample count, refinement limit, and grounding setting are applied at runtime. There are still no
bundled weights or claimed benchmark scores: run the commands above and retain the generated
`outputs/evaluation.json` with the checkpoint that produced it.

## Coherence, context, and session telemetry
`ReasoningTrace.coherence_score` is a deterministic geometric score computed from available
generation hidden states. It is a proxy for output consistency, not a statement about model
consciousness, self-awareness, or physical dynamics; unavailable hidden states leave it `None`.
`IntegrationBudget` reports the configured context token budget, used tokens, included and
dropped chunks, so context loss is inspectable rather than silently truncated. Retrieval can use
`integration_mode="integrated"` (one merged context) or `"fragmented"` (one answer per chunk);
the latter visibly reports disagreement rather than selecting a hidden winner.

Conversation sessions are in memory only and carry bounded prior turns, dropping older turns
first. Their repeated-critique pattern notes are informational and do not modify weights or
bypass grounding. Permanent reorganization is only LoRA/QLoRA fine-tuning through
`scripts/train.py`, including its `--resume-from-checkpoint` flow. A notable-convergence flag
means independent samples agreed despite weak evidence: it is a human-review signal for possible
confident hallucination, never a correctness or confidence signal. Grounding remains the only
route that can validate evidence support.

## Publish version 1.0
After validating a real export, ensure `outputs/mamv/config.json` and a model card (`outputs/mamv/README.md` or root `MODEL_CARD.md`) exist. Log in via `huggingface-cli login` or set `HF_TOKEN`, then:
```bash
python scripts/publish_to_huggingface.py --model-dir outputs/mamv --repo-id YOUR_ORG/mamv-v1 --revision v1.0
```
The script authenticates, validates required release material, creates the repo if needed, and uploads that folder at tag/revision `v1.0`. The manual GitHub workflow performs the same protected-token flow.

## Roadmap
Implemented: v1 local reading ingestion for text, Markdown, text-extractable PDF, and DOCX;
paragraph/sentence-aware chunking with PDF page locations; and feedback-only education verification
with source traces and separate confidence fields. See [Education use](docs/EDUCATION_USE.md).

Still planned: OCR (scanned/image-only PDFs are rejected), vision-language models, long-context
models, agent verification, and multi-document reasoning.

## License
[MIT](LICENSE). See [MODEL_CARD.md](MODEL_CARD.md) for model-use guidance.

## Portable, frame-relative results

`produce_result()` emits a portable non-verdict artifact with a canonical inference frame, claim/evidence/relation candidates, typed context-loss warnings, and separate model, consensus, grounding, retrieval-coverage, and coherence signals. MAMV and MAMV-IR adapters preserve frame metadata for downstream work but make no verification verdict or workflow-completion decision. See `docs/INFERENCE_FRAMES.md` and `docs/MODEL_RESULT_SCHEMA.md`.
