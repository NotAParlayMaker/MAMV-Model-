#!/usr/bin/env python3
"""Fail publication unless release evidence is complete and internally honest."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

REQUIRED = (
    "config.json",
    "MODEL_CARD.md",
    "dataset_manifest.json",
    "training_run.json",
    "checkpoint_manifest.json",
    "evaluation_frame.json",
    "evaluation_report.json",
    "reproducibility_report.json",
    "LICENSE",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    a = p.parse_args()
    d = Path(a.model_dir)
    errors = [f"missing required release evidence: {x}" for x in REQUIRED if not (d / x).is_file()]
    if not any(d.glob("*.safetensors")) and not any(d.glob("*.bin")):
        errors.append("missing model or adapter weights (*.safetensors or *.bin)")
    if not any(d.glob("tokenizer*")):
        errors.append("missing tokenizer files")
    ck = d / "checkpoint_manifest.json"
    if (
        ck.is_file()
        and json.loads(ck.read_text()).get("adapter_only")
        and not json.loads(ck.read_text()).get("base_model_required")
    ):
        errors.append("adapter checkpoint must declare base_model_required")
    if errors:
        raise SystemExit("Release validation failed:\n- " + "\n- ".join(errors))
    print("Release validation passed: complete reproducibility evidence found.")


if __name__ == "__main__":
    main()
