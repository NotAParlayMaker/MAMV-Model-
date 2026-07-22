#!/usr/bin/env python3
"""Run an inspectable, feedback-only demonstration against a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mamv_model import MAMVModel, estimate_genericity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/mamv", help="Trained adapter/checkpoint path")
    parser.add_argument("--reading", default="datasets/classroom_demo_reading.txt")
    args = parser.parse_args()
    checkpoint = Path(args.model)
    if not checkpoint.exists():
        raise SystemExit(
            f"No trained checkpoint at {checkpoint}. Run `python scripts/train.py --config configs/document_qa.yaml` first."
        )
    model = MAMVModel.load(str(checkpoint))
    session = model.education_session()
    reading = Path(args.reading).read_text(encoding="utf-8")
    for question in ("Which bed dried first?", "What will the club add before planting beans?"):
        output = session.answer(reading, question, mode="cot")
        print(f"\nQuestion: {question}\nAnswer: {output.text}\nCitations: {', '.join(output.citations)}")
        print(f"Reasoning steps: {output.reasoning.steps}\nAssumptions: {output.reasoning.assumptions}")
        print(f"Model-stated confidence: {output.stated_confidence}")
        print(f"Consensus confidence: {output.consensus_confidence}")
        print(f"Grounding gate: {output.grounding_status} ({output.grounding_confidence})")
    claim = "Every raised bed held water for the longest time."
    generic = estimate_genericity(claim)
    print(f"\nOverreaching claim: {claim}\nGenericity flag: {generic.is_generic} ({generic.quantifier})")


if __name__ == "__main__":
    main()
