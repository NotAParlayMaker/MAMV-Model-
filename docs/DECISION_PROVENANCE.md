# Decision provenance

Decision provenance records observable system operations and data relationships. It does not reveal hidden cognition, faithfully reconstruct token-level reasoning, or establish why a neural model internally produced a token.

`decision-provenance/v1` is immutable and content-addressed: timestamps never affect a graph ID. It records hashed question/answer artifacts, selected and excluded chunks, exported claim and evidence candidates, completed verifier calls, and operation records. It intentionally excludes prompts, source text, credentials, hidden states, and `ReasoningTrace.steps`.

An operation is recorded only when the software executed it. A skipped verifier is recorded as `skipped`, rather than being inferred from generated prose. Graph relationships remain non-authoritative proposals; no graph is a MAMV verdict or a MAMV-IR workflow decision.

## Architecture audit (before this implementation)

1. Frames already recorded selected/dropped chunks, retrieval scores, budgets, artifacts, settings, warnings, and frame transitions.
2. Retrieval, generation, verification, contradiction, and synthesis decisions were scattered among frames, answer fields, traces, and candidates.
3. `answer`, file/collection, and session paths carried a frame but no unified operation/result relationship graph.
4. Self-refinement retained only final answer text; trace steps could obscure prior revisions and are deliberately not exported.
5. Stated confidence, consensus confidence, and coherence could be visually confused with grounding despite separate fields.
6. Self-consistency agreement was a stability signal, not independent support, but lacked explicit lineage metadata.
7. Verifier and deterministic conflict checks returned results without an operation-level summary.
8. Candidate limitations existed, while answer-level limitations did not link each limitation to a claim transformation.
9. Prompts, raw trace steps, hidden states, credentials, and any claimed token-level rationale must remain excluded.
10. Adding result provenance requires a new `mamv-model-result/v3`; v1/v2 load with unavailable provenance rather than fabricated history.

Later focused changes will add verification depth, presentation firewall, revision/claim transformations, embodiment boundaries, and consensus lineage.
