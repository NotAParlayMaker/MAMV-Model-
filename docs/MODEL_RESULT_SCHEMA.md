# Portable result schema

New exports use `mamv-model-result/v4` and `schemas/mamv-model-result-v4.json`. Loaders retain `v1`, `v2`, and `v3` support. Legacy results migrate with `decision_provenance: null`, empty operation records, and an explicit unavailable limitation rather than fabricated history. Unknown major versions fail clearly. Candidate frame IDs always refer to the canonical result frame.

## Candidate boundary

Every claim and evidence candidate declares `derivation`: `retrieved`, `extracted`, or `generated`. Generated candidates also include `evidence_density`, a lexical measure of available scoped context, which is not evidence support or a truth signal. Candidate claims remain `unverified`, candidate evidence remains non-authoritative, and model-proposed relations remain insufficient until MAMV independently evaluates them.

`confidence_signals.model_stated_confidence` is model-reported certainty—not verification confidence. `coherence_score` is internal output consistency only and must not be interpreted as correctness. The exported generation strategy records the computation that ran; the backend's historical `cot` mode is serialized as `structured_reasoning`.

With `integration_mode: fragmented`, exports retain a separate claim candidate and source scope for each labelled chunk answer. No synthesized winner is created by the export layer.
