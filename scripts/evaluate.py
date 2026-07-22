#!/usr/bin/env python3
"""Evaluate QA outputs and write a portable JSON report."""

import argparse
import json
from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.metrics import exact_match, f1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="outputs/evaluation.json")
    p.add_argument("--reasoning-config", default="configs/reasoning.yaml")
    a = p.parse_args()
    from mamv_model.config import load_config, reasoning_answer_kwargs

    reasoning = load_config(a.reasoning_config).reasoning
    model = MAMVModel.load(a.model, require_grounding=reasoning.require_grounding)
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x]
    predictions = [
        model.answer(r["document"], r["question"], **reasoning_answer_kwargs(reasoning)).text
        for r in rows
    ]
    report = {
        "count": len(rows),
        "exact_match": sum(exact_match(x, r["answer"]) for x, r in zip(predictions, rows))
        / max(len(rows), 1),
        "f1": sum(f1(x, r["answer"]) for x, r in zip(predictions, rows)) / max(len(rows), 1),
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
