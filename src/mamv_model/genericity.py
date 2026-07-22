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
    # Bare plural subjects are a useful deterministic genericity cue ("Dogs bark").
    bare_plural = bool(re.match(r"^\s*(?:the\s+)?[a-z]+s\b", text.lower()))
    return GenericityResult(
        match is not None or bare_plural, match.group(1) if match else None,
        0.6 if match is not None else (0.5 if bare_plural else 0.4)
    )
