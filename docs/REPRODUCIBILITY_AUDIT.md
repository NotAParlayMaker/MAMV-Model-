# Reproducibility audit

## Before this change

Configuration was loaded from YAML but neither config inputs nor resolved values were recorded. SQuAD used an unpinned default revision; model and tokenizer revisions were unpinned. The training seed existed in configuration but was not applied or recorded in run evidence. No dataset, training-run, or checkpoint manifests existed. Evaluation reported only Exact Match and token F1, without an evaluation frame. Export/publish did not validate evidence, and CI combined checks in one job.

## Current evidence

Dataset preparation writes a versioned manifest with upstream revision, field mapping, preprocessing configuration, output hashes, row/rejection counts, and a content-derived ID. Training writes `training_run.json` and `checkpoint_manifest.json`; adapter checkpoints explicitly require a base model. Evaluation embeds an immutable frame in every report and marks metrics with unavailable labels as `not_evaluated`, rather than zero.

## CI root cause

The observed smoke-test failure is Python 3.14's changed pickle private API: older `datasets`/`dill` serialization calls `_batch_setitems` with an obsolete signature. The supported matrix is Python 3.10 and 3.11, and dependencies pin `datasets<3` with `dill>=0.3.7,<0.3.9`. The local environment uses Python 3.14 and is intentionally outside that matrix.

## Remaining limitations

GPU operation remains nondeterministic in some kernels. Hallucination values are observable evidence/citation proxies, not general factuality judgments. Retrieval, calibration, genericity, and contradiction metrics require task-specific labels and are reported as not evaluated when absent.
