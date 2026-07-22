# Verification stack

`mamv_model.verifier` supplies evidence-bounded **model-layer signals**, not truth
or a MAMV verification decision. `LexicalVerifier` is the dependency-free default
and retains deterministic overlap behaviour. `EntailmentVerifier` accepts an
injected NLI backend; it has no required model dependency and reports a limitation
when unavailable. `CompositeVerifier` preserves every component result and applies
one explicit policy: `conservative`, `lexical_only`, `entailment_only`, or
`require_agreement`.

The conservative policy treats an authorized contradiction as blocking support,
never lets lexical support override entailment contradiction, and returns
`ambiguous` for component disagreement. Evidence is restricted to the evidence
passed to `verify`; a result must not be read beyond that scope.

The result includes verifier identity/version, frame ID when supplied, claim and
evidence IDs, three confidence dimensions, limitations, components, explanation,
and coverage (`fully_supported`, `partially_supported`, `unsupported`,
`contradicted`, or `mixed`). `verify_atomic_claims` and `decompose_claims` keep
coordinated claims separate so support for one clause is never silently assigned to
the entire answer.

Deterministic guards flag negation, number/date, quantifier, modal, and
generic-versus-episodic mismatches. They are intentionally conservative signals;
they do not resolve context, scope, or world truth.

## Distillation watermark capability

`DistillationWatermarkVerifier` (`distillation-watermark-v1`) is an opt-in adapter
for an injected `watermark_statistical_detection` implementation. Its declared
capability permits only `model_provenance` candidates. A detection is a bounded
statistical observation on the supplied output sample and exposes only the
configured detector output. If no detector is configured, it returns
`insufficient_evidence` with an explicit limitation rather than fabricating a
detection result. As with every verifier here, this remains a model-layer signal
and never a MAMV or MAMV-IR decision.

Verifier strategy, identities, optional model identity, threshold, checks, and
selected-context evidence scope are recorded in `InferenceFrame.grounding`, so a
configuration change produces a new frame identity.
