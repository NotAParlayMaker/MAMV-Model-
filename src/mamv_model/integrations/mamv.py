"""Transform artifacts into MAMV verification requests without verdicts."""
from __future__ import annotations
from typing import Any
from ..model_result import MAMVModelResult


def to_mamv_verification_request(result: MAMVModelResult) -> dict[str, Any]:
    return {
        "request_type": "claim_verification_request",
        "result_id": result.result_id,
        "frame_id": result.inference_frame.frame_id,
        "source_ids": list(result.source_ids),
        "claim_candidates": [{
            "claim_id": candidate.claim_id, "text": candidate.text,
            "source_ids": list(candidate.source_ids), "frame_id": candidate.frame_id,
            "status": "unverified", "limitations": list(candidate.limitations),
        } for candidate in result.claim_candidates],
        "evidence_candidates": [{
            "evidence_id": item.evidence_id, "source_id": item.source_id,
            "frame_id": item.frame_id, "limitations": list(item.limitations),
        } for item in result.evidence_candidates],
        "contradiction_candidates": [{
            "contradiction_id": item.contradiction_id, "claim_a_id": item.claim_a_id,
            "claim_b_id": item.claim_b_id, "source_a_ids": list(item.source_a_ids),
            "source_b_ids": list(item.source_b_ids), "relation": item.relation,
            "frame_id": item.frame_id, "limitations": list(item.limitations),
        } for item in result.contradiction_candidates],
        "warnings": [w.__dict__ for w in result.warnings],
        "model_artifacts": result.inference_frame.model_artifacts,
        "decision_provenance": result.decision_provenance,
        "operation_records": result.operation_records,
        "provenance_status": "observable_non_authoritative" if result.decision_provenance else "unavailable",
        "limitations": list(result.limitations),
    }
