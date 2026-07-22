# Portable result schema

New exports use `mamv-model-result/v3` and `schemas/mamv-model-result-v3.json`. Loaders retain `v1` and `v2` support; legacy results migrate with `decision_provenance: null`, empty operation records, and an explicit unavailable limitation rather than fabricated history. Unknown major versions fail clearly. Candidate frame IDs always refer to the canonical result frame.
