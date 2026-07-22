#!/usr/bin/env python3
"""Update only the generated evidence block of a model card."""

import argparse
import json
from pathlib import Path

MARK = "<!-- MAMV GENERATED EVIDENCE -->"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    a = p.parse_args()
    d = Path(a.model_dir)
    t = d / "training_run.json"
    e = d / "evaluation_report.json"
    train = json.loads(t.read_text())
    report = json.loads(e.read_text())
    metrics = report["summary_metrics"]
    body = f"{MARK}\n## Reproducibility evidence\n\n- Base model: `{train['base_model']['id']}`\n- Adapter: `{train['adapter']['method']}`\n- Dataset manifest: `{train['dataset_manifest_id']}`\n- Evaluation frame: `{report['frame']['checkpoint_manifest_id']}`\n- Metrics: `{json.dumps(metrics, sort_keys=True)}`\n- Confidence is evaluated only when confidence labels are supplied; outputs are frame-relative.\n- Limitations: {report['limitations'][0]}\n"
    card = d / "MODEL_CARD.md"
    old = card.read_text() if card.exists() else "# Model Card\n\n"
    card.write_text(old.split(MARK)[0].rstrip() + "\n\n" + body)


if __name__ == "__main__":
    main()
