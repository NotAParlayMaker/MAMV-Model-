"""Dependency-light metrics used by evaluation and benchmark scripts."""

from __future__ import annotations
from collections import Counter
import re


def _tokens(value: str) -> list[str]:
    """Normalize punctuation and English articles for span metrics."""
    return [token for token in re.findall(r"\w+", value.lower()) if token not in {"a", "an", "the"}]


def exact_match(prediction: str, reference: str) -> float:
    return float(_tokens(prediction) == _tokens(reference))


def f1(prediction: str, reference: str) -> float:
    a, b = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((a & b).values())
    return 0.0 if not overlap else 2 * overlap / (sum(a.values()) + sum(b.values()))


def accuracy(predictions: list[str], references: list[str]) -> float:
    return sum(p == r for p, r in zip(predictions, references)) / max(len(references), 1)


def calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    return (
        sum(
            abs(
                sum(c for c, ok in zip(confidences, correct) if int(min(c, 0.999) * bins) == bucket)
                / max(sum(int(min(c, 0.999) * bins) == bucket for c in confidences), 1)
                - sum(
                    ok for c, ok in zip(confidences, correct) if int(min(c, 0.999) * bins) == bucket
                )
                / max(sum(int(min(c, 0.999) * bins) == bucket for c in confidences), 1)
            )
            for bucket in range(bins)
        )
        / bins
    )
