import json

import pytest

from mamv_model import MAMVModel
from mamv_model.document_qa import Answer
from mamv_model.integrations import to_mamv_ir_workflow_input, to_mamv_verification_request
from mamv_model.model_result import model_result_from_json, model_result_to_json
from mamv_model.retrieval import InMemoryRetriever


class Backend:
    def answer(self, document: str, question: str, **kwargs: object) -> Answer:
        return Answer("The sky is blue.", confidence=0.8)


def _result():
    return MAMVModel(Backend(), InMemoryRetriever({"blue": "The sky is blue.", "red": "Roses are red."})).produce_result(
        "fallback", "what color is the sky", include_claim_candidates=True,
        include_evidence_candidates=True, include_relation_candidates=True,
    )


def test_portable_result_round_trip_and_candidate_labels() -> None:
    result = _result()
    data = json.loads(model_result_to_json(result))
    assert data["schema_version"] == "mamv-model-result/v4"
    assert model_result_from_json(json.dumps(data)) == result
    assert result.claim_candidates[0].status == "unverified"
    assert all(item.status == "model_proposed" for item in result.proposed_relations)
    assert all(item.relation == "insufficient" for item in result.proposed_relations)
    assert result.decision_provenance is not None
    assert result.decision_provenance.graph_id.startswith("graph-")
    assert any(record.operation_type == "export_result" for record in result.operation_records)


def test_provenance_is_deterministic_and_has_no_dangling_or_private_content() -> None:
    first, second = _result(), _result()
    assert first.decision_provenance.graph_id == second.decision_provenance.graph_id
    graph = first.decision_provenance
    ids = {node.node_id for node in graph.nodes}
    assert all(edge.source_node_id in ids and edge.target_node_id in ids for edge in graph.edges)
    serialized = model_result_to_json(first).casefold()
    assert "hidden_states" not in serialized and "prompt" not in serialized


def test_frame_propagates_and_adapters_make_no_decisions() -> None:
    result = _result()
    frame_id = result.inference_frame.frame_id
    assert all(x.frame_id == frame_id for x in result.claim_candidates + result.evidence_candidates)
    request = to_mamv_verification_request(result)
    assert request["source_ids"] == list(result.source_ids)
    assert "verdict" not in request and "trust_receipt" not in request
    workflow = to_mamv_ir_workflow_input(result)
    assert workflow["completion_decision"] is None


def test_unknown_version_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported model result schema version"):
        model_result_from_json('{"schema_version":"future/v9"}')


def test_generated_candidates_record_derivation_and_evidence_density() -> None:
    result = _result()
    claim = result.claim_candidates[0]
    assert claim.derivation == "generated"
    assert claim.evidence_density is not None
    assert all(item.derivation == "retrieved" for item in result.evidence_candidates)
    assert result.confidence_signals.model_stated_confidence == 0.8


def test_fragmented_export_keeps_chunk_scopes_separate() -> None:
    class FragmentedBackend:
        def __init__(self): self.answers = iter(["Blue", "Red"])
        def answer(self, document: str, question: str, **kwargs: object) -> Answer:
            return Answer(next(self.answers))
    result = MAMVModel(FragmentedBackend(), InMemoryRetriever({"a": "blue", "b": "red"}), require_grounding=False).produce_result(
        "fallback", "what color", integration_mode="fragmented", include_claim_candidates=True,
        include_evidence_candidates=True, include_relation_candidates=True,
    )
    assert [claim.source_ids for claim in result.claim_candidates] == [("a",), ("b",)]
    assert {relation.claim_id for relation in result.proposed_relations} == {"claim-1", "claim-2"}


def test_v3_result_migrates_to_cautious_candidate_metadata() -> None:
    payload = json.loads(model_result_to_json(_result()))
    payload["schema_version"] = "mamv-model-result/v3"
    payload["confidence_signals"] = {"model_confidence": 0.8, "self_confidence": 0.7}
    del payload["claim_candidates"][0]["derivation"]
    del payload["claim_candidates"][0]["evidence_density"]
    del payload["evidence_candidates"][0]["derivation"]
    migrated = model_result_from_json(json.dumps(payload))
    assert migrated.schema_version == "mamv-model-result/v4"
    assert migrated.confidence_signals.model_stated_confidence == 0.7
    assert migrated.claim_candidates[0].derivation == "generated"
    assert migrated.evidence_candidates[0].derivation == "retrieved"


def test_cot_strategy_is_exported_as_structured_reasoning() -> None:
    result = MAMVModel(Backend()).produce_result("The sky is blue.", "what color", mode="cot")
    assert result.inference_frame.reasoning_strategy == "structured_reasoning"
