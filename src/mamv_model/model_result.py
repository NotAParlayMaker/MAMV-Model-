"""Portable, non-authoritative artifacts emitted by MAMV-Model.

These types intentionally describe what the model saw and proposed, not what is
true.  Verification, verdicts, receipts, and workflow decisions remain
downstream responsibilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from .genericity import GenericityResult
from .reasoning import ReasoningTrace
from .retrieval import IntegrationBudget
from .verifier import VerificationResult

MODEL_RESULT_SCHEMA_VERSION = "mamv-model-result/v1"


@dataclass(frozen=True)
class InferenceFrame:
    """Inspectable context and strategy metadata for one model inference."""

    frame_id: str
    model_id: str
    visible_source_ids: tuple[str, ...]
    excluded_source_ids: tuple[str, ...]
    reasoning_strategy: str
    grounding_required: bool
    answer_revised: bool
    multiple_samples_agreed: bool
    context_hash: str


@dataclass(frozen=True)
class ConfidenceSignals:
    """Model and heuristic signals; none are evidentiary confidence."""

    model_confidence: float | None = None
    self_confidence: float | None = None
    consensus_confidence: float | None = None
    grounding_heuristic_confidence: float | None = None


@dataclass(frozen=True)
class ClaimCandidate:
    claim_id: str
    text: str
    source_span: str | None
    claim_type: str
    literal_or_implied: str
    hedge: str | None
    hypothetical: bool
    source_ids: tuple[str, ...]
    frame_id: str
    extraction_confidence: float | None
    limitations: tuple[str, ...]
    status: Literal["unverified"] = "unverified"


@dataclass(frozen=True)
class EvidenceCandidate:
    evidence_id: str
    source_id: str
    text_hash: str
    source_location: str | None
    retrieval_score: float | None
    selected: bool
    dropped_reason: str | None
    frame_id: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ProposedEvidenceRelation:
    claim_id: str
    evidence_id: str
    relation: Literal["supports", "contradicts", "qualifies", "insufficient", "unrelated"]
    confidence: float | None
    rationale_summary: str
    limitations: tuple[str, ...]
    status: Literal["model_proposed"] = "model_proposed"


@dataclass(frozen=True)
class MAMVModelResult:
    schema_version: str
    result_id: str
    answer: str
    source_ids: tuple[str, ...]
    inference_frame: InferenceFrame
    reasoning_summary: ReasoningTrace | None
    confidence_signals: ConfidenceSignals
    grounding: VerificationResult | None
    genericity: GenericityResult | None
    integration_budget: IntegrationBudget | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    created_at: str
    claim_candidates: tuple[ClaimCandidate, ...] = ()
    evidence_candidates: tuple[EvidenceCandidate, ...] = ()
    proposed_relations: tuple[ProposedEvidenceRelation, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def model_result_to_json(result: MAMVModelResult, *, compact: bool = False) -> str:
    """Serialize a result without model prompts, credentials, or hidden CoT."""
    if result.schema_version != MODEL_RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model result schema version: {result.schema_version}")
    return json.dumps(_plain(result), sort_keys=True, separators=(",", ":") if compact else None)


def _construct(cls: type[Any], data: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


def model_result_from_json(payload: str) -> MAMVModelResult:
    """Deserialize a supported portable artifact, rejecting unknown versions."""
    data = json.loads(payload)
    version = data.get("schema_version")
    if version != MODEL_RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model result schema version: {version!r}")
    for key in ("source_ids", "warnings", "limitations"):
        data[key] = tuple(data.get(key, ()))
    data["inference_frame"] = _construct(InferenceFrame, {
        **data["inference_frame"],
        "visible_source_ids": tuple(data["inference_frame"].get("visible_source_ids", ())),
        "excluded_source_ids": tuple(data["inference_frame"].get("excluded_source_ids", ())),
    })
    data["confidence_signals"] = _construct(ConfidenceSignals, data["confidence_signals"])
    for key, cls in (("grounding", VerificationResult), ("genericity", GenericityResult),
                     ("integration_budget", IntegrationBudget), ("reasoning_summary", ReasoningTrace)):
        if data.get(key) is not None:
            item = data[key]
            for tuple_key in ("evidence", "steps", "assumptions", "critiques", "sample_traces"):
                if tuple_key in item:
                    item[tuple_key] = tuple(item[tuple_key])
            data[key] = _construct(cls, item)
    for key, cls in (("claim_candidates", ClaimCandidate), ("evidence_candidates", EvidenceCandidate),
                     ("proposed_relations", ProposedEvidenceRelation)):
        converted = []
        for item in data.get(key, []):
            for tuple_key in ("source_ids", "limitations"):
                if tuple_key in item:
                    item[tuple_key] = tuple(item[tuple_key])
            converted.append(_construct(cls, item))
        data[key] = tuple(converted)
    return _construct(MAMVModelResult, data)


def save_model_result(result: MAMVModelResult, path: str | Path, *, compact: bool = False) -> None:
    Path(path).write_text(model_result_to_json(result, compact=compact), encoding="utf-8")


def load_model_result(path: str | Path) -> MAMVModelResult:
    return model_result_from_json(Path(path).read_text(encoding="utf-8"))
