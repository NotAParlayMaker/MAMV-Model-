from mamv_model.model_result import build_inference_frame
from mamv_model.verifier import (
    CompositeVerifier, EntailmentVerifier, LexicalVerifier, VerificationResult,
    coverage_for_results, verify_atomic_claims,
)


def test_lexical_default_and_conflict_guards() -> None:
    verifier = LexicalVerifier()
    assert verifier.verify("The sky is blue", ["The sky is blue."]).label == "supported"
    assert verifier.verify("The service is not available", ["The service is available"]).label == "contradicted"
    assert verifier.verify("Capacity is 10", ["Capacity is 12"]).label == "contradicted"
    assert verifier.verify("The event is in 2024", ["The event is in 2025"]).label == "contradicted"
    assert verifier.verify("All students passed", ["Some students passed"]).label == "contradicted"
    assert verifier.verify("The system may fail", ["The system fails"]).label == "contradicted"
    assert verifier.verify("Dogs are friendly", ["Yesterday one dog was friendly"]).label == "contradicted"


def test_entailment_is_injected_and_composite_preserves_disagreement() -> None:
    entailment = EntailmentVerifier(lambda claim, text: ("supported", 0.9))
    semantic = entailment.verify("The office begins at nine", ["The office opens at 9."])
    assert semantic.label == "supported" and semantic.verifier_id == "entailment"
    result = CompositeVerifier((LexicalVerifier(), EntailmentVerifier(lambda c, e: ("contradicted", .8))), policy="conservative").verify("The office opens at nine", ["The office opens at nine"])
    assert result.label == "contradicted" and len(result.component_results) == 2


def test_atomic_coverage_missing_optional_and_frame_identity() -> None:
    parts = verify_atomic_claims(LexicalVerifier(), "The office opens at 9 and closes at 5", ["The office opens at 9"])
    assert len(parts) == 2
    assert coverage_for_results(parts) == "partially_supported"
    assert EntailmentVerifier().verify("x", ["x"]).limitations
    one = build_inference_frame(question="q", original_document="d", effective_context="d", grounding_config={"verifier_strategy": "lexical_only"})
    two = build_inference_frame(question="q", original_document="d", effective_context="d", grounding_config={"verifier_strategy": "conservative"})
    assert one.frame_id != two.frame_id
    assert VerificationResult("supported", .9, ()).label == "supported"  # signal only, never a MAMV verdict
