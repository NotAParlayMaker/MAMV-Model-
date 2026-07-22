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
    assert data["schema_version"] == "mamv-model-result/v3"
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
