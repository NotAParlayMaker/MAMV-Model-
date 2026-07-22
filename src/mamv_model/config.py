"""Typed configuration loading for MAMV training and inference."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml


@dataclass(frozen=True)
class ModelConfig:
    base_model: str | None = None
    task: str = "document_qa"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class AdapterConfig:
    enabled: bool = True
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "outputs/mamv"
    seed: int = 42
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False
    resume_from_checkpoint: str | None = None


@dataclass(frozen=True)
class ReasoningConfig:
    strategy: Literal["direct", "cot", "self_consistency", "self_refine"] = "direct"
    num_samples: int = 5
    max_refine_iterations: int = 2
    require_grounding: bool = True


@dataclass(frozen=True)
class MAMVConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: dict[str, Any] = field(default_factory=dict)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)


def load_config(path: str | Path) -> MAMVConfig:
    """Load YAML, resolving a single `_base` file relative to the child config."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text()) or {}
    if base := raw.pop("_base", None):
        parent = yaml.safe_load((source.parent / base).read_text()) or {}
        raw = {
            **parent,
            **raw,
            **{
                key: {**parent.get(key, {}), **raw[key]}
                for key in raw
                if isinstance(raw[key], dict)
            },
        }
    return MAMVConfig(
        model=ModelConfig(**raw.get("model", {})),
        adapter=AdapterConfig(**raw.get("adapter", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=raw.get("data", {}),
        reasoning=ReasoningConfig(**raw.get("reasoning", {})),
    )
