"""Generic-statement and quantifier utilities."""

from __future__ import annotations
import re
from dataclasses import dataclass

QUANTIFIERS = ("all", "every", "most", "many", "some", "none", "no")


@dataclass(frozen=True)
class GenericityResult:
    is_generic: bool
    quantifier: str | None
    confidence: float


def estimate_genericity(text: str) -> GenericityResult:
    """Conservative lexical baseline; replace with a trained classification backend in production."""
    match = re.search(r"\b(" + "|".join(QUANTIFIERS) + r")\b", text.lower())
    return GenericityResult(
        match is not None, match.group(1) if match else None, 0.6 if match else 0.4
    )
