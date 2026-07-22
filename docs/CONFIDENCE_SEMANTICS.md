# Confidence semantics and authority boundaries

MAMV-Model confidence fields are bounded, frame-relative model signals. Lexical
confidence is overlap; entailment confidence is supplied by an optional local or
injected backend; composite confidence is the explicit policy result. None is a
probability of truth, an authorization, or a final MAMV trust verdict.

`supported` means only that the configured verifier found bounded support in the
provided evidence. `contradicted`, `insufficient_evidence`, and `ambiguous` are
likewise evidence-relative signals. Coverage reports whether atomic claims were
fully, partially, not, or mixedly supported and must not be compressed into a
single affirmative claim.

Downstream MAMV verification remains responsible for normalization, authority,
provenance, policy, and final trust decisions. Missing optional semantic models
produce limitations rather than fabricated semantic outputs.
