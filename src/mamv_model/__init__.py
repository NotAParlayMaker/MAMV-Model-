"""Public MAMV model API."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from .document_qa import Answer, DocumentQABackend
from .genericity import GenericityResult, estimate_genericity
from .reasoning import ReasoningTrace
from .retrieval import IntegrationBudget, Retriever, select_with_budget, InMemoryRetriever, RetrievalDiversitySettings, select_diverse, RetrievedDocument
from .verifier import EvidenceVerifier, LexicalVerifier, VerificationResult
from .model_result import (
    MODEL_RESULT_SCHEMA_VERSION, ClaimCandidate, ConfidenceSignals, EvidenceCandidate,
    InferenceFrame, MAMVModelResult, ProposedEvidenceRelation, utc_now, build_inference_frame,
    content_hash, FrameWarning, compare_inference_frames, make_frame_transition, ContradictionCandidate,
)

if TYPE_CHECKING:
    from .session import ConversationSession


@dataclass(frozen=True)
class EducationAnswer:
    """Feedback-only answer shape; deliberately contains no grading fields."""

    text: str
    reasoning: ReasoningTrace
    citations: tuple[str, ...]
    stated_confidence: float | None
    consensus_confidence: float | None
    grounding_confidence: float | None
    grounding_status: str
    inference_frame: InferenceFrame | None = None


class EducationSession:
    """A non-grading classroom façade which always exposes evidence and trace."""

    def __init__(self, model: "MAMVModel") -> None:
        self.model = model

    @staticmethod
    def _reject_grading_kwargs(kwargs: dict[str, Any]) -> None:
        forbidden = {"as_grade", "grade", "score", "pass_fail"} & set(kwargs)
        if forbidden:
            raise ValueError("EducationSession never grades student work; request feedback instead.")

    def answer(self, document: str, question: str, **kwargs: Any) -> EducationAnswer:
        self._reject_grading_kwargs(kwargs)
        answer = self.model.answer(document, question, **kwargs)
        trace = answer.reasoning or ReasoningTrace()
        verification = self.model.verifier.verify(answer.text, [document])
        consensus = answer.confidence if kwargs.get("mode") == "self_consistency" else None
        return EducationAnswer(
            text=answer.text,
            reasoning=trace,
            citations=answer.sources or ("provided document",),
            stated_confidence=trace.self_confidence,
            consensus_confidence=consensus,
            grounding_confidence=verification.confidence,
            grounding_status=verification.label,
            inference_frame=answer.inference_frame,
        )

    def answer_file(self, path: str | Path, question: str, **kwargs: Any) -> EducationAnswer:
        from .ingestion import chunk_document, ingest_file

        self._reject_grading_kwargs(kwargs)
        text = ingest_file(path)
        result = self.model.answer_file(path, question, **kwargs)
        trace = result.reasoning or ReasoningTrace()
        verification = self.model.verifier.verify(result.text, [str(c) for c in chunk_document(text)])
        return EducationAnswer(
            result.text, trace, result.sources or (str(path),), trace.self_confidence,
            result.confidence if kwargs.get("mode") == "self_consistency" else None,
            verification.confidence, verification.label, result.inference_frame,
        )


class MAMVModel:
    """Composable MAMV inference façade backed by a real HF checkpoint."""

    def __init__(
        self,
        qa: DocumentQABackend,
        retriever: Retriever | None = None,
        require_grounding: bool = True,
        verifier: EvidenceVerifier | None = None,
    ) -> None:
        self.qa, self.retriever, self.verifier = qa, retriever, verifier or LexicalVerifier()
        self.require_grounding = require_grounding

    @classmethod
    def load(cls, model_id_or_path: str | None = None, **kwargs: Any) -> "MAMVModel":
        if not model_id_or_path:
            raise ValueError(
                "model_id_or_path is required; train or download a real checkpoint first."
            )
        require_grounding = kwargs.pop("require_grounding", True)
        return cls(
            DocumentQABackend.from_pretrained(model_id_or_path, **kwargs),
            require_grounding=require_grounding,
        )

    def answer(
        self, document: str, question: str, *, integration_mode: Literal["fragmented", "integrated"] = "integrated",
        integration_max_tokens: int | None = None, **kwargs: Any,
    ) -> Answer:
        sources = self.retriever.retrieve(question) if self.retriever else []
        selected = sources
        budget = None
        if sources and integration_max_tokens is not None:
            selected, budget = select_with_budget(sources, integration_max_tokens)
        context = "\n\n".join(item.text for item in selected) or document
        fragmented_disagreement = False
        if integration_mode == "fragmented" and selected:
            parts = [self._ask(item.text, question, **kwargs) for item in selected]
            labels = [self.verifier.verify(part.text, [other.text for other in selected if other is not item]).label for item, part in zip(selected, parts)]
            disagreement = len({_normalise(part.text) for part in parts}) > 1 or "insufficient_evidence" in labels
            fragmented_disagreement = disagreement
            text = "\n".join(f"[{item.source_id}] {part.text}" for item, part in zip(selected, parts))
            if disagreement:
                text = "Per-chunk answers disagree; human review required:\n" + text
            answer = Answer(text, confidence=None, reasoning=ReasoningTrace(critiques=("Per-chunk answers disagree" if disagreement else "Per-chunk answers are lexically consistent",)))
        elif integration_mode == "integrated":
            answer = self._ask(context, question, **kwargs)
        else:
            raise ValueError(f"Unsupported integration mode: {integration_mode}")
        reasoning = answer.reasoning
        confidence = answer.confidence
        if self.retriever and self.require_grounding:
            verification = self.verifier.verify(answer.text, [item.text for item in selected])
            if verification.label == "insufficient_evidence":
                confidence = min(confidence if confidence is not None else 1.0, verification.confidence)
                critique = "Answer not well-supported by retrieved evidence"
                reasoning = (
                    replace(reasoning, critiques=reasoning.critiques + (critique,))
                    if reasoning
                    else ReasoningTrace(critiques=(critique,))
                )
        dropped = [item for item in sources if item not in selected]
        warnings = ([FrameWarning("FRAGMENTED_DISAGREEMENT", "Per-chunk answers disagree.", "inference.integration_mode")] if fragmented_disagreement else [])
        mode = str(kwargs.get("mode", "direct"))
        frame = self._frame(document, question, context, selected, dropped, mode=mode, integration_mode=integration_mode, budget=budget, kwargs=kwargs, warnings=tuple(warnings))
        transition = None
        if mode == "self_refine":
            initial = self._frame(document, question, context, selected, dropped, mode="direct", integration_mode=integration_mode, budget=budget, kwargs=kwargs)
            transition = make_frame_transition(initial, frame, "self_refinement", answer_changed=True, explanation="The answer was revised after model self-critique.")
        return Answer(answer.text, confidence, tuple(item.source_id for item in selected), reasoning, budget,
                      answer.notable_convergence, answer.notable_convergence_reason, frame, transition)

    def answer_file(self, path: str | Path, question: str, **kwargs: Any) -> Answer:
        """Answer from a local course reading with chunk locations as sources."""
        from .ingestion import chunk_document, ingest_file
        from .retrieval import InMemoryRetriever

        reading = ingest_file(path)
        chunks = chunk_document(reading)
        if not chunks:
            raise ValueError("The reading contains no usable text.")
        chunk_map = {
            f"{Path(path).name}{(': ' + chunk.source_location) if chunk.source_location else ''}": str(chunk)
            for chunk in chunks
        }
        # A caller-provided retriever may already index this reading; otherwise use
        # the lightweight local retriever so every file works without setup.
        selected = (
            self.retriever.retrieve(question, top_k=3)
            if self.retriever
            else InMemoryRetriever(chunk_map).retrieve(question, top_k=min(3, len(chunk_map)))
        )
        if not selected:
            selected = InMemoryRetriever(chunk_map).retrieve(question, top_k=min(3, len(chunk_map)))
        context = "\n\n".join(item.text for item in selected)
        answer = self._ask(context, question, **kwargs)
        verification = self.verifier.verify(answer.text, [item.text for item in selected])
        reasoning = answer.reasoning
        confidence = answer.confidence
        if self.require_grounding and verification.label == "insufficient_evidence":
            confidence = min(confidence if confidence is not None else 1.0, verification.confidence)
            critique = "Answer not well-supported by selected reading passages"
            reasoning = replace(reasoning, critiques=reasoning.critiques + (critique,)) if reasoning else ReasoningTrace(critiques=(critique,))
        frame = self._frame(reading, question, context, selected, [], mode=str(kwargs.get("mode", "direct")), integration_mode="integrated", budget=None, kwargs=kwargs, document_type=Path(path).suffix.lstrip(".") or "text")
        return Answer(answer.text, confidence, tuple(item.source_id for item in selected), reasoning, None, answer.notable_convergence, answer.notable_convergence_reason, frame)

    def answer_files(self, paths: list[str | Path] | tuple[str | Path, ...], question: str, *, synthesis_mode: Literal["source_separated", "cautious_synthesis", "consensus_only", "contradiction_first"] = "cautious_synthesis", max_chunks_per_document: int = 3, min_documents: int = 2, top_k: int = 6, **kwargs: Any) -> Answer:
        """Answer an ordered reading collection without collapsing source provenance."""
        from .ingestion import ingest_documents, chunk_documents
        collection, texts = ingest_documents(paths)
        chunks = chunk_documents(collection, texts)
        refs = {d.document_id: d for d in collection.documents}
        candidates = [RetrievedDocument(str(c), c.chunk_id or f"{c.document_id}-chunk-{i}", 0.0, c.document_id, refs[c.document_id].media_type, refs[c.document_id].modified_at, c.content_hash, c.source_location) for i, c in enumerate(chunks, 1)]
        ranked = InMemoryRetriever({x.source_id: x.text for x in candidates}).retrieve(question, top_k=len(candidates))
        candidates = [replace(x, score=next(r.score for r in ranked if r.source_id == x.source_id)) for x in candidates]
        settings = RetrievalDiversitySettings(top_k, max_chunks_per_document, min_documents, True)
        selected, dropped, decisions = select_diverse(candidates, settings)
        if not selected: selected = candidates[:top_k]
        per_source: list[tuple[RetrievedDocument, Answer]] = [(item, self._ask(item.text, question, **kwargs)) for item in selected]
        contradictions = _contradictions(per_source, "pending")
        if synthesis_mode == "source_separated": text = "\n\n".join(f"[{refs[i.document_id].name}] {a.text}" for i, a in per_source)
        elif synthesis_mode == "consensus_only":
            normalized = {}; [normalized.setdefault(_normalise(a.text), []).append((i,a)) for i,a in per_source]
            agreed = [v for v in normalized.values() if len({i.document_id for i,_ in v}) >= min_documents]
            text = "\n".join(a.text for group in agreed for _, a in group[:1]) or "No claim met the configured cross-source agreement threshold."
        else:
            synthesized = self._ask("\n\n".join(i.text for i in selected), question, **kwargs).text
            conflict = "Source disagreements require review: " + "; ".join(c.explanation_summary for c in contradictions) if contradictions else "No bounded contradiction candidate was detected among selected passages."
            text = (conflict + "\n\n" + synthesized) if synthesis_mode == "contradiction_first" else (synthesized + "\n\n" + conflict)
        # Rebind proposal candidates to the immutable frame identity.
        frame = self._frame("\n\n".join(str(t) for t in texts.values()), question, "\n\n".join(i.text for i in selected), selected, dropped, mode=str(kwargs.get("mode", "direct")), integration_mode="integrated", budget=None, kwargs=kwargs, document_type="collection")
        contradictions = tuple(replace(c, frame_id=frame.frame_id) for c in contradictions)
        frame = replace(frame, collection_id=collection.collection_id, document_ids=tuple(refs), selected_chunks_per_document={d: tuple(i.source_id for i in selected if i.document_id == d) for d in refs}, dropped_chunks_per_document={d: tuple(i.source_id for i in dropped if i.document_id == d) for d in refs}, retrieval_diversity_settings={"top_k":top_k,"max_chunks_per_document":max_chunks_per_document,"min_documents":min_documents,"deduplicate":True,"decisions":decisions}, synthesis_mode=synthesis_mode, contradiction_candidates=contradictions, collection_limitations=collection.limitations)
        agreement = "Conflicting source-specific answers were detected." if contradictions else "Selected source-specific answers did not yield a bounded conflict candidate."
        return Answer(text, None, tuple(i.source_id for i in selected), ReasoningTrace(critiques=(agreement,)), None, False, None, frame, None, tuple(refs.values()), contradictions, agreement, synthesis_mode)

    def _frame(self, original: str, question: str, context: str, selected: list[Any], dropped: list[Any], *, mode: str, integration_mode: str, budget: IntegrationBudget | None, kwargs: dict[str, Any], warnings: tuple[FrameWarning, ...] = (), document_type: str | None = None) -> InferenceFrame:
        artifacts = {"base_model_id": getattr(self.qa, "model_id", type(self.qa).__name__), "requested_revision": getattr(self.qa, "requested_revision", None), "resolved_revision": getattr(self.qa, "resolved_revision", None), "adapter_id": getattr(self.qa, "adapter_id", None), "adapter_revision": getattr(self.qa, "adapter_revision", None), "tokenizer_id": getattr(getattr(self.qa, "tokenizer", None), "name_or_path", type(getattr(self.qa, "tokenizer", None)).__name__), "tokenizer_revision": getattr(self.qa, "tokenizer_revision", None), "local_config_hash": content_hash(getattr(getattr(self.qa, "model", None), "config", {}))}
        generation = {k: kwargs.get(k) for k in ("temperature", "top_p", "max_new_tokens", "seed") if k in kwargs} | {"number_of_samples": kwargs.get("n_samples", 1), "refinement_limit": kwargs.get("max_iterations", 0)}
        retrieval = {"retriever_type": type(self.retriever).__name__ if self.retriever else None, "query": question, "top_k": kwargs.get("top_k", 5), "selected_source_ids": tuple(x.source_id for x in selected), "excluded_source_ids": tuple(x.source_id for x in dropped), "retrieval_scores": {x.source_id:x.score for x in selected + dropped}, "token_budget": budget.max_tokens if budget else None, "tokens_used": budget.tokens_used if budget else None, "document_type": document_type}
        verification_config = {"verifier_strategy": getattr(self.verifier, "policy", "lexical_only"), "verifier_id": getattr(self.verifier, "verifier_id", type(self.verifier).__name__), "verifier_version": getattr(self.verifier, "verifier_version", "unknown"), "required":self.require_grounding, "evidence_scope":"selected_context", "enabled_deterministic_checks": {"numeric": getattr(self.verifier, "detect_numeric_conflicts", None), "negation": getattr(self.verifier, "detect_negation_conflicts", None), "quantifier": getattr(self.verifier, "detect_quantifier_conflicts", None)}, "optional_model_identity": {"model_id": getattr(self.verifier, "model_id", None), "revision": getattr(self.verifier, "revision", None)}, "thresholds": {"lexical_overlap": 0.5}}
        return build_inference_frame(question=question, original_document=original, effective_context=context, selected_sources=selected, dropped_sources=dropped, model_artifacts=artifacts, retrieval_config=retrieval, generation_config=generation, reasoning_strategy=mode, integration_mode=integration_mode, integration_budget=budget, grounding_config=verification_config, assumptions=(), limitations=("This artifact does not assert truth or create a verification verdict.",), extra_warnings=warnings)

    def education_session(self) -> EducationSession:
        return EducationSession(self)

    def conversation_session(self, document: str, *, max_tokens: int = 512) -> "ConversationSession":
        from .session import ConversationSession
        return ConversationSession(self, document, max_tokens=max_tokens)

    def _ask(self, document: str, question: str, **kwargs: Any) -> Answer:
        """Apply reasoning strategies for minimal backend test doubles as well as HF backends."""
        mode = kwargs.get("mode", "direct")
        if not isinstance(self.qa, DocumentQABackend) and mode in {"self_consistency", "self_refine"}:
            from .reasoning import self_consistency, self_refine
            copied = dict(kwargs)
            copied.pop("mode", None)
            return (self_consistency(self.qa, document, question, **copied)
                    if mode == "self_consistency" else self_refine(self.qa, document, question, **copied))
        return self.qa.answer(document, question, **kwargs)

    def verify_claim(self, claim: str, evidence: list[str]) -> VerificationResult:
        return self.verifier.verify(claim, evidence)

    def produce_result(
        self, document: str, question: str, *, include_claim_candidates: bool = False,
        include_evidence_candidates: bool = False, include_relation_candidates: bool = False,
        semantic_analyzers: tuple[Any, ...] = (), **kwargs: Any,
    ) -> MAMVModelResult:
        """Produce a portable artifact with proposals explicitly kept non-authoritative."""
        answer = self.answer(document, question, **kwargs)
        retrieved = self.retriever.retrieve(question) if self.retriever else []
        selected_ids = answer.sources
        selected = [item for item in retrieved if item.source_id in selected_ids]
        excluded = [item for item in retrieved if item.source_id not in selected_ids]
        mode = str(kwargs.get("mode", "direct"))
        frame = answer.inference_frame or self._frame(document, question, document, selected, excluded, mode=mode, integration_mode=str(kwargs.get("integration_mode", "integrated")), budget=answer.integration_budget, kwargs=kwargs)
        grounding = self.verifier.verify(answer.text, [item.text for item in selected]) if selected else None
        # Do not export trace steps: they can contain private model scratchwork.
        trace = answer.reasoning
        summary = (ReasoningTrace(assumptions=trace.assumptions, critiques=trace.critiques,
                   self_confidence=trace.self_confidence, verification_label=trace.verification_label,
                   coherence_score=trace.coherence_score, session_pattern_note=trace.session_pattern_note,
                   notable_convergence=trace.notable_convergence,
                   notable_convergence_reason=trace.notable_convergence_reason) if trace else None)
        limitations = ["This artifact does not assert truth or create a verification verdict."]
        claims: tuple[ClaimCandidate, ...] = ()
        evidence: tuple[EvidenceCandidate, ...] = ()
        relations: tuple[ProposedEvidenceRelation, ...] = ()
        if include_claim_candidates and answer.text.strip():
            claims = (ClaimCandidate("claim-1", answer.text.strip(), None, "assertion", "literal", None,
                      False, selected_ids, frame.frame_id, answer.confidence,
                      ("Candidate only; MAMV must normalize and verify it.",)),)
        if include_evidence_candidates:
            all_items = [(item, True, None) for item in selected] + [
                (item, False, "excluded_from_context") for item in excluded]
            evidence = tuple(EvidenceCandidate(
                f"evidence-{index}", item.source_id, content_hash(item.text), getattr(item, "source_location", None),
                item.score, selected_flag, dropped, frame.frame_id,
                ("Retrieval score is not evidence support.",),
            ) for index, (item, selected_flag, dropped) in enumerate(all_items, 1))
        if include_relation_candidates and claims and evidence:
            relations = tuple(ProposedEvidenceRelation(
                claims[0].claim_id, item.evidence_id, "insufficient", None,
                "Model export does not determine evidentiary support.",
                ("Model-proposed only; MAMV verification is required.",),
            ) for item in evidence)
        # Analyzers are deliberately optional and their outputs are warning-only metadata.
        warnings = frame.warnings + tuple(FrameWarning("GROUNDING_SCOPE_LIMITED", f"Semantic analyzer available: {analyzer.__class__.__name__}", "semantic_analyzers", "info") for analyzer in semantic_analyzers)
        return MAMVModelResult(
            MODEL_RESULT_SCHEMA_VERSION, "result-" + content_hash({"frame_id": frame.frame_id, "answer": answer.text})[:24],
            answer.text, selected_ids, frame, summary,
            ConfidenceSignals(answer.confidence, trace.self_confidence if trace else None,
                              answer.confidence if mode == "self_consistency" else None,
                              grounding.confidence if grounding else None),
            grounding, estimate_genericity(answer.text), answer.integration_budget,
            warnings, tuple(limitations), utc_now(), claims, evidence, relations, answer.frame_transition,
        )


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _contradictions(items: list[tuple[RetrievedDocument, Answer]], frame_id: str) -> tuple[ContradictionCandidate, ...]:
    """Small, inspectable candidate detector; it proposes conflicts, never resolves them."""
    import re
    result = []
    for n, (left, a) in enumerate(items):
        for right, b in items[n + 1:]:
            if left.document_id == right.document_id: continue
            x, y = _normalise(a.text), _normalise(b.text)
            nums_x, nums_y = set(re.findall(r"\b\d+(?:\.\d+)?\b", x)), set(re.findall(r"\b\d+(?:\.\d+)?\b", y))
            dates_x, dates_y = set(re.findall(r"\b(?:19|20)\d{2}\b", x)), set(re.findall(r"\b(?:19|20)\d{2}\b", y))
            polarity = (" not " in f" {x} " ) != (" not " in f" {y} ")
            relation = None; summary = ""
            if dates_x and dates_y and dates_x != dates_y:
                relation, summary = "temporally_distinct", "Different dates may describe different times rather than a contradiction."
            elif nums_x and nums_y and nums_x != nums_y:
                relation, summary = "potentially_conflicting", "Different numeric values were proposed by separate sources."
            elif polarity:
                relation, summary = "potentially_conflicting", "Opposite polarity appears in source-specific answers."
            elif x != y and set(x.split()) & set(y.split()):
                relation, summary = "incomparable", "Answers overlap but the bounded heuristic cannot determine whether their scopes differ."
            if relation:
                result.append(ContradictionCandidate(f"contradiction-{len(result)+1}", f"claim-{n+1}", f"claim-{n+2}", (left.source_id,), (right.source_id,), relation, None, summary, frame_id, ("Heuristic/model-proposed candidate only; MAMV retains verification authority.",)))
    return tuple(result)


__all__ = ["Answer", "ClaimCandidate", "ConfidenceSignals", "EducationAnswer", "EducationSession", "EvidenceCandidate", "GenericityResult", "InferenceFrame", "IntegrationBudget", "MAMVModel", "MAMVModelResult", "ProposedEvidenceRelation", "ReasoningTrace", "build_inference_frame", "compare_inference_frames", "estimate_genericity"]
from .provenance import compare_evaluation_reports
