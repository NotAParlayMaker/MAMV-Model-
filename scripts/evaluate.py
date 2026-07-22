#!/usr/bin/env python3
"""Evaluate QA outputs in an explicit, portable evaluation frame."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.metrics import exact_match, f1
from mamv_model.provenance import (
    environment,
    file_hash,
    read_json,
    timestamp,
    unavailable,
    write_json,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="outputs/evaluation_report.json")
    p.add_argument("--reasoning-config", default="configs/reasoning.yaml")
    p.add_argument("--checkpoint-manifest")
    p.add_argument("--dataset-manifest")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    from mamv_model.config import load_config, reasoning_answer_kwargs

    reasoning = load_config(a.reasoning_config).reasoning
    model = MAMVModel.load(a.model, require_grounding=reasoning.require_grounding)
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x]
    results = [
        model.answer(r["document"], r["question"], **reasoning_answer_kwargs(reasoning))
        for r in rows
    ]
    predictions = [x.text for x in results]
    dm = read_json(a.dataset_manifest) if a.dataset_manifest else {}
    cm = read_json(a.checkpoint_manifest) if a.checkpoint_manifest else {}
    frame = {
        "schema_version": "v1",
        "checkpoint_manifest_id": cm.get("checkpoint_id"),
        "dataset_manifest_id": dm.get("manifest_id"),
        "split_fingerprint": file_hash(a.data),
        "preprocessing_version": dm.get("preprocessing_version"),
        "prompt_template_version": "document-qa-v1",
        "reasoning_mode": reasoning.strategy,
        "retrieval_configuration": None,
        "integration_mode": reasoning.integration_mode,
        "grounding_configuration": {"required": reasoning.require_grounding},
        "genericity_configuration": None,
        "generation_configuration": {},
        "seed": a.seed,
        "metric_implementation_versions": {"qa": "v1", "hallucination_proxy": "v1"},
        "environment_metadata": environment(),
        "evaluation_timestamp": timestamp(),
    }
    n = max(len(rows), 1)
    em = sum(exact_match(x, r["answer"]) for x, r in zip(predictions, rows)) / n
    f = sum(f1(x, r["answer"]) for x, r in zip(predictions, rows)) / n
    unsupported = [not bool(getattr(result, "citations", [])) for result in results]
    summary = {
        "exact_match": em,
        "token_f1": f,
        "answer_length_mean": sum(len(x.split()) for x in predictions) / n,
        "unsupported_generation_rate": sum(unsupported) / n,
        "evidence_contradiction_rate": unavailable(),
        "unsupported_specificity_rate": unavailable(),
        "abstention_rate": unavailable(),
        "grounding_support_rate": unavailable(),
        "source_citation_accuracy": unavailable(),
        "retrieval_recall_at_k": unavailable(),
        "retrieval_precision_at_k": unavailable(),
        "calibration_error": unavailable(),
        "brier_score": unavailable(),
    }
    report = {
        "schema_version": "v1",
        "frame": frame,
        "summary_metrics": summary,
        "slice_metrics": {
            "answerable": {"sample_count": len(rows), "exact_match": em},
            "grounded": {
                "sample_count": sum(not x for x in unsupported),
                "warning": "tiny slices must not be ranked",
            },
        },
        "sample_counts": {"total": len(rows)},
        "abstentions": 0,
        "errors": [],
        "unevaluated_metrics": {k: v for k, v in summary.items() if isinstance(v, dict)},
        "limitations": [
            "Unsupported generation is a transparent proxy based on available citations, not a universal hallucination score.",
            "Contradiction and specificity require evidence annotations.",
        ],
    }
    out = Path(a.output)
    write_json(out, report)
    write_json(out.parent / "evaluation_frame.json", frame)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
