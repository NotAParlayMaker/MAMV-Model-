"""Claim-verification interfaces and evidence-preserving result types."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol

VerificationLabel = Literal["supported", "refuted", "not_enough_information"]


@dataclass(frozen=True)
class VerificationResult:
    label: VerificationLabel
    confidence: float
    evidence: tuple[str, ...]


class Verifier(Protocol):
    def verify(self, claim: str, evidence: list[str]) -> VerificationResult: ...


class LexicalVerifier:
    def verify(self, claim: str, evidence: list[str]) -> VerificationResult:
        claim_terms = set(claim.lower().split())
        best = max(
            evidence, key=lambda text: len(claim_terms & set(text.lower().split())), default=""
        )
        overlap = len(claim_terms & set(best.lower().split())) / max(len(claim_terms), 1)
        return VerificationResult(
            "supported" if overlap >= 0.5 else "not_enough_information",
            overlap,
            (best,) if best else (),
        )
