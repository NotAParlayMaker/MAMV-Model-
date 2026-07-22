"""Deterministic hidden-state consistency telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence


@dataclass(frozen=True)
class HiddenStateTrajectory:
    """Hidden-state vectors collected per generation step or per sample."""

    states: Sequence[Sequence[float]]


def compute_coherence_score(trajectory: HiddenStateTrajectory) -> float:
    """Return mean adjacent cosine similarity mapped to ``[0, 1]``.

    This is a lexical/geometric proxy for output consistency, not evidence of
    correctness and not a claim about phenomenology or literal attractor dynamics.
    Zero vectors contribute neutral similarity (0.5 after normalization).
    """
    vectors = [tuple(float(x) for x in state) for state in trajectory.states]
    if len(vectors) < 2:
        return 1.0 if vectors else 0.0
    scores = []
    for left, right in zip(vectors, vectors[1:]):
        if len(left) != len(right):
            raise ValueError("all hidden-state vectors must have the same dimension")
        denom = sqrt(sum(x * x for x in left) * sum(x * x for x in right))
        cosine = sum(x * y for x, y in zip(left, right)) / denom if denom else 0.0
        scores.append(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))
    return sum(scores) / len(scores)
