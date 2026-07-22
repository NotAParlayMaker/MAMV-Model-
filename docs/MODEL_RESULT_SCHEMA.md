# Portable result schema

New exports use `mamv-model-result/v2` and `schemas/mamv-model-result-v2.json`. Loaders retain `v1` support and migrate it with an explicit limitation: v1 did not record complete context or artifact revisions. Unknown versions fail clearly. Candidate frame IDs always refer to the canonical result frame.
