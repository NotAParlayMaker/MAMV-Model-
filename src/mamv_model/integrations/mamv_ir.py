"""Transform artifacts into MAMV-IR workflow inputs without decisions."""
from __future__ import annotations
from typing import Any
from ..model_result import MAMVModelResult


def to_mamv_ir_workflow_input(result: MAMVModelResult) -> dict[str, Any]:
    return {
        "input_type": "model_artifact",
        "result_id": result.result_id,
        "frame_id": result.inference_frame.frame_id,
        "source_ids": list(result.source_ids),
        "answer": result.answer,
        "limitations": list(result.limitations),
        "workflow_state": "unassigned",
        "completion_decision": None,
    }
