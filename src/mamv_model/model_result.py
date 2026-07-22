"""Portable, frame-relative inference artifacts (never verification verdicts)."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from typing import Any, Literal, Mapping
from .genericity import GenericityResult
from .reasoning import ReasoningTrace
from .retrieval import IntegrationBudget
from .verifier import VerificationResult

MODEL_RESULT_SCHEMA_VERSION = "mamv-model-result/v2"
LEGACY_SCHEMA_VERSION = "mamv-model-result/v1"

def utc_now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def canonical_json(value: Any) -> str:
    """Stable, secret-free JSON suitable for content identity hashing."""
    def clean(v: Any) -> Any:
        if hasattr(v, "__dataclass_fields__"): return {name: clean(getattr(v, name)) for name in v.__dataclass_fields__}
        if isinstance(v, Mapping): return {str(k): clean(x) for k, x in sorted(v.items()) if str(k).lower() not in {"api_token", "access_token", "secret", "password", "prompt", "hidden_states"}}
        if isinstance(v, (tuple, list)): return [clean(x) for x in v]
        return v
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def content_hash(value: Any) -> str: return hashlib.sha256(canonical_json(value).encode()).hexdigest()

@dataclass(frozen=True)
class FrameWarning:
    code: Literal["CONTEXT_TRUNCATED", "SOURCES_DROPPED", "SESSION_TURNS_DROPPED", "RETRIEVAL_EMPTY", "FRAGMENTED_DISAGREEMENT", "MODEL_REVISION_UNPINNED", "TOKENIZER_REVISION_UNPINNED", "ADAPTER_REVISION_UNPINNED", "ANSWER_RELATIVE_TO_SELECTED_CONTEXT", "GROUNDING_SCOPE_LIMITED"]
    message: str; affected_field: str; severity: Literal["info", "warning", "error"] = "warning"; source_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class InferenceFrame:
    frame_id: str; created_at: str; question_hash: str; original_document_hash: str; effective_context_hash: str
    model_artifacts: dict[str, Any]; context: dict[str, Any]; inference: dict[str, Any]; retrieval: dict[str, Any]
    grounding: dict[str, Any]; session: dict[str, Any] | None; assumptions: tuple[str, ...]; limitations: tuple[str, ...]; warnings: tuple[FrameWarning, ...]
    # Legacy convenience views retained for callers of v1.
    model_id: str = ""; visible_source_ids: tuple[str, ...] = (); excluded_source_ids: tuple[str, ...] = (); reasoning_strategy: str = "direct"; grounding_required: bool = True; answer_revised: bool = False; multiple_samples_agreed: bool = False; context_hash: str = ""
    collection_id: str | None = None; document_ids: tuple[str, ...] = (); selected_chunks_per_document: dict[str, tuple[str, ...]] | None = None; dropped_chunks_per_document: dict[str, tuple[str, ...]] | None = None; retrieval_diversity_settings: dict[str, Any] | None = None; synthesis_mode: str | None = None; contradiction_candidates: tuple[Any, ...] = (); temporal_relations: tuple[Any, ...] = (); collection_limitations: tuple[str, ...] = ()

@dataclass(frozen=True)
class InferenceFrameTransition:
    transition_id: str; source_frame_id: str; target_frame_id: str; transition_type: str; changed_fields: tuple[str, ...]; preserved_source_ids: tuple[str, ...]; added_source_ids: tuple[str, ...]; removed_source_ids: tuple[str, ...]; answer_changed: bool; grounding_changed: bool; explanation: str

@dataclass(frozen=True)
class FrameCompatibility:
    identical: bool; context_compatible: bool; artifact_compatible: bool; retrieval_compatible: bool; reasoning_compatible: bool; grounding_compatible: bool; directly_comparable: bool; requires_reverification: bool; differences: tuple[str, ...]; explanation: str

def _source(item: Any) -> dict[str, Any]:
    return {"source_id": getattr(item, "source_id", str(item)), "chunk_hash": content_hash(getattr(item, "text", "")), "source_location": getattr(item, "source_location", None), "score": getattr(item, "score", None)}
def build_inference_frame(*, question: str, original_document: str, effective_context: str, selected_sources: tuple[Any, ...] | list[Any] = (), dropped_sources: tuple[Any, ...] | list[Any] = (), model_artifacts: dict[str, Any] | None = None, retrieval_config: dict[str, Any] | None = None, generation_config: dict[str, Any] | None = None, reasoning_strategy: str = "direct", integration_mode: str = "integrated", integration_budget: IntegrationBudget | None = None, grounding_config: dict[str, Any] | None = None, session_context: dict[str, Any] | None = None, parent_frame_id: str | None = None, assumptions: tuple[str, ...] = (), limitations: tuple[str, ...] = (), extra_warnings: tuple[FrameWarning, ...] = (), collection_id: str | None = None, document_ids: tuple[str, ...] = (), synthesis_mode: str | None = None, contradiction_candidates: tuple[Any, ...] = (), temporal_relations: tuple[Any, ...] = (), collection_limitations: tuple[str, ...] = ()) -> InferenceFrame:
    selected, dropped = tuple(_source(x) for x in selected_sources), tuple(_source(x) for x in dropped_sources)
    artifacts, retrieval, generation, grounding = model_artifacts or {}, retrieval_config or {}, generation_config or {}, grounding_config or {}
    warnings = list(extra_warnings)
    if integration_budget and integration_budget.truncated: warnings += [FrameWarning("CONTEXT_TRUNCATED", "Context token budget excluded chunks.", "context.truncation_status"), FrameWarning("SOURCES_DROPPED", "Retrieved sources were excluded from context.", "context.dropped_chunks", source_ids=tuple(x["source_id"] for x in dropped))]
    if not selected_sources and retrieval.get("retriever_type"): warnings.append(FrameWarning("RETRIEVAL_EMPTY", "Retriever returned no selected source.", "retrieval.selected_source_ids"))
    for field, code in (("requested_revision", "MODEL_REVISION_UNPINNED"), ("tokenizer_revision", "TOKENIZER_REVISION_UNPINNED"), ("adapter_revision", "ADAPTER_REVISION_UNPINNED")):
        if artifacts.get(field) is None and (field != "adapter_revision" or artifacts.get("adapter_id")): warnings.append(FrameWarning(code, f"{field.replace('_', ' ').title()} is not pinned.", f"model_artifacts.{field}"))
    context = {"source_ids": tuple(x["source_id"] for x in selected), "included_chunks": selected, "dropped_chunks": dropped, "truncation_status": bool(integration_budget and integration_budget.truncated), "document_type": retrieval.get("document_type"), "source_locations": {x["source_id"]: x["source_location"] for x in selected + dropped if x["source_location"]}}
    session = ({**session_context, "parent_frame_id": parent_frame_id} if session_context or parent_frame_id else None)
    selected_by_doc = {d: tuple(x["source_id"] for x in selected if getattr(next((i for i in selected_sources if getattr(i, "source_id", str(i)) == x["source_id"]), None), "document_id", None) == d) for d in document_ids}
    dropped_by_doc = {d: tuple(x["source_id"] for x in dropped if getattr(next((i for i in dropped_sources if getattr(i, "source_id", str(i)) == x["source_id"]), None), "document_id", None) == d) for d in document_ids}
    identity = {"question_hash": content_hash(question), "original_document_hash": content_hash(original_document), "effective_context_hash": content_hash(effective_context), "model_artifacts": artifacts, "context": context, "inference": {"reasoning_strategy": reasoning_strategy, "integration_mode": integration_mode, **generation}, "retrieval": retrieval, "grounding": grounding, "session": session, "assumptions": assumptions, "limitations": limitations, "collection_id": collection_id, "document_ids": document_ids, "synthesis_mode": synthesis_mode, "contradiction_candidates": contradiction_candidates, "temporal_relations": temporal_relations}
    fid = "frame-" + content_hash(identity)[:24]
    return InferenceFrame(fid, utc_now(), identity["question_hash"], identity["original_document_hash"], identity["effective_context_hash"], artifacts, context, identity["inference"], retrieval, grounding, session, assumptions, limitations, tuple(warnings), artifacts.get("base_model_id", "unknown"), context["source_ids"], tuple(x["source_id"] for x in dropped), reasoning_strategy, bool(grounding.get("required", True)), reasoning_strategy == "self_refine", reasoning_strategy == "self_consistency", identity["effective_context_hash"], collection_id, document_ids, selected_by_doc, dropped_by_doc, retrieval.get("diversity_settings"), synthesis_mode, contradiction_candidates, temporal_relations, collection_limitations)

def make_frame_transition(source: InferenceFrame, target: InferenceFrame, transition_type: str, *, answer_changed=False, grounding_changed=False, explanation="") -> InferenceFrameTransition:
    a, b = set(source.visible_source_ids), set(target.visible_source_ids); changed = tuple(k for k in ("model_artifacts", "context", "inference", "retrieval", "grounding", "session") if getattr(source, k) != getattr(target, k))
    return InferenceFrameTransition("transition-" + content_hash((source.frame_id, target.frame_id, transition_type))[:24], source.frame_id, target.frame_id, transition_type, changed, tuple(sorted(a & b)), tuple(sorted(b-a)), tuple(sorted(a-b)), answer_changed, grounding_changed, explanation)
def compare_inference_frames(a: InferenceFrame, b: InferenceFrame) -> FrameCompatibility:
    artifact = a.model_artifacts == b.model_artifacts; context = (a.effective_context_hash == b.effective_context_hash and a.visible_source_ids == b.visible_source_ids and a.excluded_source_ids == b.excluded_source_ids); retrieval = a.retrieval == b.retrieval; reasoning = a.reasoning_strategy == b.reasoning_strategy; grounding = a.grounding == b.grounding
    diffs = tuple(k for k, ok in (("artifacts", artifact), ("context", context), ("retrieval", retrieval), ("reasoning", reasoning), ("grounding", grounding)) if not ok); identical = a.frame_id == b.frame_id
    return FrameCompatibility(identical, context, artifact, retrieval, reasoning, grounding, artifact and context, not (artifact and context and grounding), diffs, "Frames are identical." if identical else "Changed artifacts require re-evaluation; changed context requires re-grounding.")

@dataclass(frozen=True)
class ConfidenceSignals:
    model_confidence: float | None = None; self_confidence: float | None = None; consensus_confidence: float | None = None; grounding_heuristic_confidence: float | None = None; retrieval_coverage: float | None = None; coherence_score: float | None = None
    def __post_init__(self):
        for value in (self.model_confidence, self.self_confidence, self.consensus_confidence, self.grounding_heuristic_confidence, self.retrieval_coverage, self.coherence_score):
            if value is not None and not 0 <= value <= 1: raise ValueError("Confidence signals must be within [0, 1].")
@dataclass(frozen=True)
class ClaimCandidate: claim_id: str; text: str; source_span: str | None; claim_type: str; literal_or_implied: str; hedge: str | None; hypothetical: bool; source_ids: tuple[str, ...]; frame_id: str; extraction_confidence: float | None; limitations: tuple[str, ...]; status: Literal["unverified"] = "unverified"
@dataclass(frozen=True)
class EvidenceCandidate: evidence_id: str; source_id: str; text_hash: str; source_location: str | None; retrieval_score: float | None; selected: bool; dropped_reason: str | None; frame_id: str; limitations: tuple[str, ...]
@dataclass(frozen=True)
class ProposedEvidenceRelation: claim_id: str; evidence_id: str; relation: Literal["supports", "contradicts", "qualifies", "insufficient", "unrelated"]; confidence: float | None; rationale_summary: str; limitations: tuple[str, ...]; status: Literal["model_proposed"] = "model_proposed"
@dataclass(frozen=True)
class ContradictionCandidate:
    contradiction_id: str; claim_a_id: str; claim_b_id: str; source_a_ids: tuple[str, ...]; source_b_ids: tuple[str, ...]; relation: Literal["contradicts", "temporally_distinct", "scope_distinct", "qualifies", "potentially_conflicting", "incomparable"]; confidence: float | None; explanation_summary: str; frame_id: str; limitations: tuple[str, ...]
@dataclass(frozen=True)
class MAMVModelResult:
    schema_version: str; result_id: str; answer: str; source_ids: tuple[str, ...]; inference_frame: InferenceFrame; reasoning_summary: ReasoningTrace | None; confidence_signals: ConfidenceSignals; grounding: VerificationResult | None; genericity: GenericityResult | None; integration_budget: IntegrationBudget | None; warnings: tuple[FrameWarning, ...]; limitations: tuple[str, ...]; created_at: str; claim_candidates: tuple[ClaimCandidate, ...] = (); evidence_candidates: tuple[EvidenceCandidate, ...] = (); proposed_relations: tuple[ProposedEvidenceRelation, ...] = (); frame_transition: InferenceFrameTransition | None = None; contradiction_candidates: tuple[ContradictionCandidate, ...] = (); document_sources: tuple[Any, ...] = (); source_agreement_summary: str | None = None; synthesis_mode: str | None = None
def _plain(value: Any) -> Any: return json.loads(canonical_json(value))
def model_result_to_json(result: MAMVModelResult, *, compact=False) -> str:
    if result.schema_version != MODEL_RESULT_SCHEMA_VERSION: raise ValueError(f"Unsupported model result schema version: {result.schema_version}")
    return json.dumps(_plain(result), sort_keys=True, separators=(",", ":") if compact else None)
def _construct(cls, data): return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})
def model_result_from_json(payload: str) -> MAMVModelResult:
    d=json.loads(payload); version=d.get("schema_version")
    if version not in (LEGACY_SCHEMA_VERSION, MODEL_RESULT_SCHEMA_VERSION): raise ValueError(f"Unsupported model result schema version: {version!r}")
    if version == LEGACY_SCHEMA_VERSION:
        f=d["inference_frame"]; d["inference_frame"]=asdict(build_inference_frame(question="legacy", original_document="", effective_context="", selected_sources=tuple(), dropped_sources=tuple(), model_artifacts={"base_model_id":f.get("model_id", "unknown")}, reasoning_strategy=f.get("reasoning_strategy", "direct"), grounding_config={"required":f.get("grounding_required", True)}, limitations=("Migrated from v1; original context and artifact revisions were not recorded.",))) | {"frame_id": f["frame_id"], "context_hash": f.get("context_hash", "")}; d["schema_version"]=MODEL_RESULT_SCHEMA_VERSION
    for k in ("source_ids", "limitations"): d[k]=tuple(d.get(k,()))
    fd=d["inference_frame"]
    for k in ("visible_source_ids","excluded_source_ids","assumptions","limitations","warnings","document_ids","contradiction_candidates","temporal_relations","collection_limitations"): fd[k]=tuple(fd.get(k,()))
    fd["warnings"]=tuple(FrameWarning(**({**x,"source_ids":tuple(x.get("source_ids",()))} if isinstance(x,dict) else {"code":"GROUNDING_SCOPE_LIMITED","message":x,"affected_field":"legacy","severity":"warning"})) for x in fd["warnings"])
    for section, keys in (("context", ("source_ids", "included_chunks", "dropped_chunks")), ("retrieval", ("selected_source_ids", "excluded_source_ids")), ("session", ("included_turn_ids", "dropped_turn_ids"))):
        if fd.get(section):
            for key in keys:
                if key in fd[section]: fd[section][key] = tuple(fd[section][key])
    if fd.get("grounding") and "limitations" in fd["grounding"]: fd["grounding"]["limitations"] = tuple(fd["grounding"]["limitations"])
    d["inference_frame"]=_construct(InferenceFrame,fd); d["warnings"]=tuple(FrameWarning(**({**x,"source_ids":tuple(x.get("source_ids",()))} if isinstance(x,dict) else {"code":"GROUNDING_SCOPE_LIMITED","message":x,"affected_field":"legacy","severity":"warning"})) for x in d.get("warnings",())); d["document_sources"] = tuple(d.get("document_sources", ()))
    d["confidence_signals"]=_construct(ConfidenceSignals,d["confidence_signals"])
    for k,c in (("grounding",VerificationResult),("genericity",GenericityResult),("integration_budget",IntegrationBudget),("reasoning_summary",ReasoningTrace)):
        if d.get(k) is not None:
            if k == "grounding" and "evidence" in d[k]: d[k]["evidence"] = tuple(d[k]["evidence"])
            d[k]=_construct(c,d[k])
    for k,c in (("claim_candidates",ClaimCandidate),("evidence_candidates",EvidenceCandidate),("proposed_relations",ProposedEvidenceRelation),("contradiction_candidates",ContradictionCandidate)):
        d[k]=tuple(_construct(c,{**x, **{z:tuple(x[z]) for z in ("source_ids","limitations") if z in x}}) for x in d.get(k,()))
    if d.get("frame_transition"): d["frame_transition"]=_construct(InferenceFrameTransition,d["frame_transition"])
    return _construct(MAMVModelResult,d)
def save_model_result(result,path,*,compact=False): Path(path).write_text(model_result_to_json(result,compact=compact),encoding="utf-8")
def load_model_result(path): return model_result_from_json(Path(path).read_text(encoding="utf-8"))
