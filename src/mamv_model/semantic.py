"""Optional, non-authoritative semantic-analysis extension contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .model_result import InferenceFrame


@dataclass(frozen=True)
class SemanticAnalysisResult:
    analyzer_name: str
    findings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ("Model-proposed semantic analysis is not authoritative.",)


class SemanticAnalyzer(Protocol):
    def analyze(self, text: str, context: str, frame: InferenceFrame) -> SemanticAnalysisResult: ...
