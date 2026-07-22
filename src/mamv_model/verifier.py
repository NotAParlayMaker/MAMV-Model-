"""Evidence-bounded, non-authoritative claim-verification signals.

These verifiers describe relations between a claim and the supplied evidence only.
They never establish a MAMV trust verdict.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, Sequence

from .genericity import estimate_genericity

if TYPE_CHECKING:
    from .model_result import EvidenceCandidate, InferenceFrame

VerificationLabel = Literal["supported", "contradicted", "insufficient_evidence", "ambiguous"]
EvidenceCoverage = Literal["fully_supported", "partially_supported", "unsupported", "contradicted", "mixed"]


@dataclass(frozen=True)
class VerifierCapability:
    """Declared boundary for a verifier's candidate, model-layer output."""

    allowed_claim_types: tuple[str, ...]
    allowed_methods: tuple[str, ...]
    prohibited_assertions: tuple[str, ...]


@dataclass(frozen=True)
class VerificationResult:
    """A model-layer signal, with legacy ``label/confidence/evidence`` conveniences."""
    label: VerificationLabel | Literal["refuted", "not_enough_information"] = "insufficient_evidence"
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    verifier_id: str = "lexical"
    verifier_version: str = "1"
    frame_id: str | None = None
    claim: str = ""
    evidence_ids: tuple[str, ...] = ()
    support_label: VerificationLabel | None = None
    support_confidence: float | None = None
    contradiction_confidence: float = 0.0
    insufficient_evidence_confidence: float = 0.0
    limitations: tuple[str, ...] = ()
    component_results: tuple["VerificationResult", ...] = ()
    explanation_summary: str = ""
    coverage: EvidenceCoverage = "unsupported"

    def __post_init__(self) -> None:
        # Accept old callers while ensuring all newly produced labels use the new vocabulary.
        normalized = {"refuted": "contradicted", "not_enough_information": "insufficient_evidence"}.get(self.label, self.label)
        object.__setattr__(self, "label", normalized)
        object.__setattr__(self, "support_label", self.support_label or normalized)
        object.__setattr__(self, "support_confidence", self.confidence if self.support_confidence is None else self.support_confidence)
        if not self.evidence_ids:
            object.__setattr__(self, "evidence_ids", self.evidence)


class EvidenceVerifier(Protocol):
    verifier_id: str
    verifier_version: str

    def verify(self, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None) -> VerificationResult: ...


# Alias retained for integrations which imported the old protocol.
Verifier = EvidenceVerifier


def _evidence(evidence: Sequence["EvidenceCandidate | str"]) -> tuple[list[str], tuple[str, ...]]:
    texts, ids = [], []
    for index, item in enumerate(evidence, 1):
        if isinstance(item, str):
            texts.append(item); ids.append(f"evidence-{index}")
        else:
            text = getattr(item, "text", None)
            if text is not None: texts.append(str(text))
            ids.append(getattr(item, "evidence_id", f"evidence-{index}"))
    return texts, tuple(ids)


def _tokens(text: str) -> set[str]: return set(re.findall(r"\b[\w'-]+\b", text.casefold()))
def _numbers(text: str) -> set[str]: return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
def _dates(text: str) -> set[str]: return set(re.findall(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|(?:19|20)\d{2})\b", text))
def _negated(text: str) -> bool: return bool(re.search(r"\b(?:no|not|never|none|cannot|can't|won't|isn't|aren't|doesn't|don't|didn't)\b", text.casefold()))
def _modal(text: str) -> bool: return bool(re.search(r"\b(?:may|might|could|can|possibly|perhaps)\b", text.casefold()))
def _episodic(text: str) -> bool: return bool(re.search(r"\b(?:yesterday|today|last|on \w+ \d|in \d{4}|this \w+)\b", text.casefold()))


def deterministic_conflicts(claim: str, evidence: str, *, numeric: bool = True, negation: bool = True, quantifier: bool = True) -> tuple[str, ...]:
    """Conservative checks: a mismatch is a conflict signal, not a truth decision."""
    found = []
    if negation and _negated(claim) != _negated(evidence): found.append("negation mismatch")
    verbs = {"open", "opens", "close", "closes", "start", "starts", "end", "ends"}
    claim_verbs, evidence_verbs = _tokens(claim) & verbs, _tokens(evidence) & verbs
    comparable_predicate = not (claim_verbs and evidence_verbs and claim_verbs != evidence_verbs)
    if numeric and comparable_predicate and _numbers(claim) and _numbers(evidence) and _numbers(claim) != _numbers(evidence): found.append("numeric mismatch")
    if numeric and _dates(claim) and _dates(evidence) and _dates(claim) != _dates(evidence): found.append("date mismatch")
    if quantifier:
        cq, eq = estimate_genericity(claim).quantifier, estimate_genericity(evidence).quantifier
        if cq in {"all", "every"} and eq in {"some", "many"}: found.append("universal/existential mismatch")
        if _modal(claim) != _modal(evidence): found.append("modal mismatch")
        if estimate_genericity(claim).is_generic and _episodic(evidence): found.append("generic/episodic mismatch")
    return tuple(found)


class LexicalVerifier:
    """The original deterministic overlap baseline, now with bounded conflict checks."""
    verifier_id = "lexical"
    verifier_version = "2"

    def __init__(self, *, detect_numeric_conflicts: bool = True, detect_negation_conflicts: bool = True, detect_quantifier_conflicts: bool = True) -> None:
        self.detect_numeric_conflicts, self.detect_negation_conflicts, self.detect_quantifier_conflicts = detect_numeric_conflicts, detect_negation_conflicts, detect_quantifier_conflicts

    def verify(self, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None) -> VerificationResult:
        texts, ids = _evidence(evidence)
        best = max(texts, key=lambda text: len(_tokens(claim) & _tokens(text)), default="")
        overlap = len(_tokens(claim) & _tokens(best)) / max(len(_tokens(claim)), 1)
        conflicts = deterministic_conflicts(claim, best, numeric=self.detect_numeric_conflicts, negation=self.detect_negation_conflicts, quantifier=self.detect_quantifier_conflicts) if best else ()
        if conflicts:
            label, confidence, coverage = "contradicted", 0.9, "contradicted"
            explanation = "; ".join(conflicts)
        # A universal claim needs universal evidence; shared surface words alone are
        # not coverage of its scope.
        elif _predicate_mismatch(claim, best):
            label, confidence, coverage = "insufficient_evidence", overlap, "unsupported"
            explanation = "Shared subject terms do not establish a different predicate."
        elif estimate_genericity(claim).quantifier in {"all", "every"} and estimate_genericity(best).quantifier not in {"all", "every"}:
            label, confidence, coverage = "insufficient_evidence", overlap, "partially_supported"
            explanation = "Lexical overlap does not establish the claim's universal scope."
        elif overlap >= .5:
            label, confidence, coverage = "supported", overlap, "fully_supported"
            explanation = "Deterministic lexical term overlap met the 0.5 threshold."
        else:
            label, confidence, coverage = "insufficient_evidence", overlap, "unsupported"
            explanation = "Lexical overlap did not meet the 0.5 threshold."
        return VerificationResult(label, confidence, (best,) if best else (), self.verifier_id, self.verifier_version, getattr(frame, "frame_id", None), claim, ids, label, confidence if label == "supported" else 0.0, confidence if label == "contradicted" else 0.0, confidence if label == "insufficient_evidence" else 0.0, (), (), explanation, coverage)


class EntailmentVerifier:
    """Optional NLI adapter.  A backend is injected so base installs need no model."""
    verifier_id = "entailment"
    verifier_version = "1"

    def __init__(self, backend: Callable[[str, str], Any] | None = None, *, model_id: str | None = None, revision: str | None = None) -> None:
        self.backend, self.model_id, self.revision = backend, model_id, revision

    def verify(self, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None) -> VerificationResult:
        texts, ids = _evidence(evidence)
        if self.backend is None:
            return VerificationResult("insufficient_evidence", 0.0, (), self.verifier_id, self.verifier_version, getattr(frame, "frame_id", None), claim, ids, "insufficient_evidence", 0.0, 0.0, 1.0, ("Optional entailment backend is unavailable; no semantic result was fabricated.",), (), "No optional NLI backend is configured.", "unsupported")
        best, raw = "", None
        for text in texts:
            candidate = self.backend(claim, text)
            if isinstance(candidate, VerificationResult): raw, best = candidate, text; break
            label, confidence = candidate if isinstance(candidate, tuple) else (candidate.get("label"), candidate.get("confidence", 0.0))
            trial = VerificationResult(label, float(confidence), (text,))
            if raw is None or trial.confidence > raw.confidence: raw, best = trial, text
        if raw is None: raw = VerificationResult("insufficient_evidence", 0.0, ())
        return VerificationResult(raw.label, raw.confidence, (best,) if best else (), self.verifier_id, self.verifier_version, getattr(frame, "frame_id", None), claim, ids, raw.label, raw.confidence if raw.label == "supported" else 0.0, raw.confidence if raw.label == "contradicted" else 0.0, raw.confidence if raw.label == "insufficient_evidence" else 0.0, raw.limitations, (raw,), "Injected entailment backend result.", raw.coverage)


class DistillationWatermarkVerifier:
    """Adapter for bounded watermark-statistical detection observations.

    The injected detector must inspect the supplied output sample only and return
    ``(detected, confidence)`` or a mapping with matching keys. Its output is
    limited to the configured detector's observation.
    """

    verifier_id = "distillation-watermark-v1"
    verifier_version = "1"
    capability = VerifierCapability(
        allowed_claim_types=("model_provenance",),
        allowed_methods=("watermark_statistical_detection",),
        prohibited_assertions=(
            "cannot prove direct training on our outputs vs. incidental corpus overlap",
            "cannot establish intent or authorization",
            "cannot prove the absence of distillation, only its statistical absence in this sample",
        ),
    )

    def __init__(self, detector: Callable[[str], Any] | None = None) -> None:
        self.detector = detector

    def verify(
        self, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None,
    ) -> VerificationResult:
        texts, ids = _evidence(evidence)
        sample = "\n".join(texts)
        common = (
            "Watermark detection is a statistical observation on the supplied sample, not a MAMV or MAMV-IR decision.",
            "The result is limited to the configured detector output and supplied sample.",
        )
        if self.detector is None:
            return VerificationResult(
                "insufficient_evidence", 0.0, (), self.verifier_id, self.verifier_version,
                getattr(frame, "frame_id", None), claim, ids, "insufficient_evidence", 0.0, 0.0, 1.0,
                common + ("No watermark statistical detector was configured; no result was fabricated.",), (),
                "No watermark-statistical observation was produced.", "unsupported",
            )
        raw = self.detector(sample)
        if isinstance(raw, tuple):
            detected, confidence = raw
        elif isinstance(raw, dict):
            detected, confidence = raw.get("detected", False), raw.get("confidence", 0.0)
        else:
            raise TypeError("Watermark detector must return (detected, confidence) or a mapping.")
        confidence = max(0.0, min(float(confidence), 1.0))
        if bool(detected):
            label, coverage = "supported", "fully_supported"
            summary = "Configured watermark-statistical detector reported a signal in the supplied sample."
        else:
            label, coverage = "insufficient_evidence", "unsupported"
            summary = "Configured watermark-statistical detector did not report a signal in the supplied sample."
        return VerificationResult(
            label, confidence, tuple(texts), self.verifier_id, self.verifier_version,
            getattr(frame, "frame_id", None), claim, ids, label,
            confidence if label == "supported" else 0.0, 0.0,
            confidence if label == "insufficient_evidence" else 0.0, common, (), summary, coverage,
        )


class CompositeVerifier:
    verifier_id = "composite"
    verifier_version = "1"
    def __init__(self, components: Sequence[EvidenceVerifier], *, policy: Literal["conservative", "lexical_only", "entailment_only", "require_agreement"] = "conservative") -> None: self.components, self.policy = tuple(components), policy
    def verify(self, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None) -> VerificationResult:
        results = tuple(c.verify(claim, evidence, frame=frame) for c in self.components)
        picked = tuple(r for r in results if (self.policy != "lexical_only" or r.verifier_id == "lexical") and (self.policy != "entailment_only" or r.verifier_id == "entailment"))
        limitations = tuple(x for r in results for x in r.limitations)
        active = tuple(r for r in picked if not r.limitations or r.support_confidence or r.contradiction_confidence)
        if not active: label, confidence, coverage, summary = "insufficient_evidence", 0.0, "unsupported", "No selected verifier produced evidence-bounded output."
        elif any(r.label == "contradicted" for r in active): label, confidence, coverage, summary = "contradicted", max(r.contradiction_confidence for r in active), "contradicted", "An authorized component reported contradiction."
        elif self.policy == "require_agreement" and len({r.label for r in active}) != 1: label, confidence, coverage, summary = "ambiguous", 0.0, "mixed", "Components disagree."
        elif self.policy == "conservative" and len({r.label for r in active}) > 1: label, confidence, coverage, summary = "ambiguous", 0.0, "mixed", "Components disagree."
        else:
            winner = max(active, key=lambda r: r.confidence); label, confidence, coverage, summary = winner.label, winner.confidence, winner.coverage, "Composite policy selected component signal."
        return VerificationResult(label, confidence, tuple(x for r in results for x in r.evidence), self.verifier_id, self.verifier_version, getattr(frame, "frame_id", None), claim, tuple(x for r in results for x in r.evidence_ids), label, confidence if label == "supported" else 0.0, confidence if label == "contradicted" else 0.0, confidence if label == "insufficient_evidence" else 0.0, limitations, results, summary, coverage)


def decompose_claims(text: str) -> tuple[str, ...]:
    """Small transparent splitter for coordinated clauses; callers retain each result."""
    parts = [p.strip(" ,.;") for p in re.split(r"\s+(?:and|but)\s+", text) if p.strip(" ,.;")]
    if len(parts) == 2 and re.search(r"\b(?:opens|starts|is)\b", parts[0]) and re.match(r"^(?:closes|ends|is)\b", parts[1]):
        subject = parts[0].split(" opens", 1)[0] if " opens" in parts[0] else ""
        parts[1] = f"{subject} {parts[1]}".strip()
    return tuple(parts) or (text,)


def verify_atomic_claims(verifier: EvidenceVerifier, claim: str, evidence: Sequence["EvidenceCandidate | str"], *, frame: "InferenceFrame | None" = None) -> tuple[VerificationResult, ...]:
    return tuple(verifier.verify(part, evidence, frame=frame) for part in decompose_claims(claim))


def _predicate_mismatch(claim: str, evidence: str) -> bool:
    verbs = {"open", "opens", "close", "closes", "start", "starts", "end", "ends"}
    left, right = _tokens(claim) & verbs, _tokens(evidence) & verbs
    return bool(left and right and left != right)


def coverage_for_results(results: Sequence[VerificationResult]) -> EvidenceCoverage:
    """Preserve partial atomic support instead of promoting it to ``supported``."""
    labels = {result.label for result in results}
    if not labels or labels == {"insufficient_evidence"}:
        return "unsupported"
    if labels == {"supported"}:
        return "fully_supported"
    if labels == {"contradicted"}:
        return "contradicted"
    if "contradicted" in labels:
        return "mixed"
    if "supported" in labels and "insufficient_evidence" in labels:
        return "partially_supported"
    return "mixed"
