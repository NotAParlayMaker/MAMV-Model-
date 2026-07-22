#!/usr/bin/env python3
"""Download and convert the openly licensed SQuAD extractive-QA corpus to MAMV JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def records(split: Any, limit: int) -> list[dict[str, str]]:
    """Keep only questions with a first, non-empty span answer."""
    result: list[dict[str, str]] = []
    for row in split:
        answers = row["answers"]["text"]
        if answers and answers[0].strip():
            result.append(
                {
                    "document": row["context"],
                    "question": row["question"],
                    "answer": answers[0].strip(),
                }
            )
        if len(result) >= limit:
            break
    return result


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=32)
    args = parser.parse_args()
    if args.train_limit < 1 or args.validation_limit < 1:
        raise SystemExit("limits must be positive")
    from datasets import load_dataset

    squad = load_dataset("squad")
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "squad_train.jsonl", records(squad["train"], args.train_limit))
    write_jsonl(
        output_dir / "squad_validation.jsonl", records(squad["validation"], args.validation_limit)
    )


if __name__ == "__main__":
    main()
