"""Observable decision provenance, deliberately excluding private reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .model_result import InferenceFrame, content_hash

DECISION_PROVENANCE_SCHEMA_VERSION = "decision-provenance/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str; operation_type: str; frame_id: str; implementation_id: str
    implementation_version: str; input_ids: tuple[str, ...]; output_ids: tuple[str, ...]
    status: str; deterministic: bool; configuration_hash: str; warnings: tuple[str, ...]
    limitations: tuple[str, ...]; started_at: str | None = None; completed_at: str | None = None


def operation_record(*, operation_type: str, frame_id: str, implementation_id: str,
                     implementation_version: str = "1", input_ids: Sequence[str] = (),
                     output_ids: Sequence[str] = (), status: str = "completed",
                     deterministic: bool = True, configuration: Mapping[str, Any] | None = None,
                     warnings: Sequence[str] = (), limitations: Sequence[str] = ()) -> OperationRecord:
    """Create an operation identity from observable, time-independent content."""
    identity = {"type": operation_type, "frame": frame_id, "implementation": implementation_id,
                "version": implementation_version, "inputs": tuple(input_ids),
                "outputs": tuple(output_ids), "status": status, "configuration": configuration or {}}
    return OperationRecord("operation-" + content_hash(identity)[:24], operation_type, frame_id,
        implementation_id, implementation_version, tuple(input_ids), tuple(output_ids), status,
        deterministic, content_hash(configuration or {}), tuple(warnings), tuple(limitations))


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str; node_type: str; frame_id: str; label: str; artifact_id: str | None
    source_ids: tuple[str, ...]; claim_ids: tuple[str, ...]; evidence_ids: tuple[str, ...]
    attributes: Mapping[str, Any]; limitations: tuple[str, ...]; created_at: str


@dataclass(frozen=True)
class ProvenanceEdge:
    edge_id: str; source_node_id: str; target_node_id: str; relation: str
    operation_id: str | None; explanation_summary: str; limitations: tuple[str, ...]


@dataclass(frozen=True)
class DecisionProvenanceGraph:
    schema_version: str; graph_id: str; frame_id: str; nodes: tuple[ProvenanceNode, ...]
    edges: tuple[ProvenanceEdge, ...]; unresolved_items: tuple[str, ...]
    limitations: tuple[str, ...]; created_at: str


def build_decision_provenance(*, frame: InferenceFrame, question: str,
                              source_documents: Sequence[Any] = (), retrieved_chunks: Sequence[Any] = (),
                              dropped_chunks: Sequence[Any] = (), answer: Any,
                              claim_candidates: Sequence[Any] = (), evidence_candidates: Sequence[Any] = (),
                              verifier_results: Sequence[Any] = (), contradiction_candidates: Sequence[Any] = (),
                              revision_records: Sequence[Any] = (), operation_records: Sequence[OperationRecord] = (),
                              limitations: Sequence[str] = ()) -> DecisionProvenanceGraph:
    """Construct a canonical graph of software-observable relationships only.

    Text is represented by hashes and bounded labels, never prompts, hidden states, or traces.
    """
    nodes: list[ProvenanceNode] = []
    edges: list[ProvenanceEdge] = []
    def node(kind: str, key: str, label: str, **kwargs: Any) -> str:
        nid = "node-" + content_hash((frame.frame_id, kind, key))[:24]
        nodes.append(ProvenanceNode(nid, kind, frame.frame_id, label, kwargs.get("artifact_id"),
            tuple(kwargs.get("source_ids", ())), tuple(kwargs.get("claim_ids", ())),
            tuple(kwargs.get("evidence_ids", ())), kwargs.get("attributes", {}),
            tuple(kwargs.get("limitations", ())), _now()))
        return nid
    def edge(source: str, target: str, relation: str, operation_id: str | None = None,
             explanation: str = "", limitations_: Sequence[str] = ()) -> None:
        eid = "edge-" + content_hash((source, target, relation, operation_id, explanation, tuple(limitations_)))[:24]
        edges.append(ProvenanceEdge(eid, source, target, relation, operation_id, explanation, tuple(limitations_)))

    question_node = node("question", content_hash(question), "question", attributes={"question_hash": content_hash(question)})
    answer_node = node("generated_answer", content_hash(getattr(answer, "text", "")), "generated answer",
                       attributes={"answer_hash": content_hash(getattr(answer, "text", ""))})
    for item in source_documents:
        sid = getattr(item, "document_id", None) or getattr(item, "source_id", str(item))
        source = node("document", sid, "source document", artifact_id=sid, source_ids=(sid,))
        edge(source, question_node, "selected_for", explanation="Document was available to this inference frame.")
    chunk_nodes: dict[str, str] = {}
    for item, kind, relation in [(x, "selected_context", "selected_for") for x in retrieved_chunks] + [(x, "excluded_context", "excluded_from") for x in dropped_chunks]:
        sid = getattr(item, "source_id", str(item)); chunk_nodes[sid] = node(kind, sid, sid, artifact_id=sid, source_ids=(sid,), attributes={"content_hash": content_hash(getattr(item, "text", "")), "retrieval_score": getattr(item, "score", None)})
        edge(chunk_nodes[sid], answer_node, relation, explanation="Context selection recorded by the retrieval path.")
    for claim in claim_candidates:
        cid = getattr(claim, "claim_id"); cnode = node("claim_candidate", cid, cid, claim_ids=(cid,), source_ids=getattr(claim, "source_ids", ()), limitations=getattr(claim, "limitations", ()))
        edge(answer_node, cnode, "decomposed_into", explanation="Claim candidate was exported from the generated answer.")
    for evidence in evidence_candidates:
        eid = getattr(evidence, "evidence_id"); sid = getattr(evidence, "source_id", "")
        enode = node("evidence_candidate", eid, eid, evidence_ids=(eid,), source_ids=(sid,), limitations=getattr(evidence, "limitations", ()))
        if sid in chunk_nodes: edge(chunk_nodes[sid], enode, "derived_from", explanation="Evidence candidate references this retrieved chunk.")
    for result in verifier_results:
        key = content_hash((getattr(result, "claim", ""), getattr(result, "verifier_id", ""), getattr(result, "evidence_ids", ())))
        vnode = node("verifier_result", key, getattr(result, "verifier_id", "verifier"), evidence_ids=getattr(result, "evidence_ids", ()), limitations=getattr(result, "limitations", ()))
        edge(answer_node, vnode, "checked_by", explanation="Verifier call was completed; its result remains model-layer evidence bounded.")
    unresolved = tuple(sorted({warning for record in operation_records if record.status != "completed" for warning in record.warnings + record.limitations}))
    nodes.sort(key=lambda x: x.node_id); edges.sort(key=lambda x: x.edge_id)
    valid = {x.node_id for x in nodes}
    if any(x.source_node_id not in valid or x.target_node_id not in valid for x in edges): raise ValueError("Decision provenance graph contains dangling edges.")
    identity = {"version": DECISION_PROVENANCE_SCHEMA_VERSION, "frame": frame.frame_id,
                "nodes": [{k: v for k, v in n.__dict__.items() if k != "created_at"} for n in nodes],
                "edges": [x.__dict__ for x in edges], "unresolved": unresolved, "limitations": tuple(limitations)}
    return DecisionProvenanceGraph(DECISION_PROVENANCE_SCHEMA_VERSION, "graph-" + content_hash(identity)[:24], frame.frame_id,
        tuple(nodes), tuple(edges), unresolved, tuple(limitations), _now())
