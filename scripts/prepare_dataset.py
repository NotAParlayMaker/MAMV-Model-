#!/usr/bin/env python3
"""Download and convert the openly licensed SQuAD extractive-QA corpus to MAMV JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mamv_model.provenance import dataset_manifest, file_hash, timestamp, write_json


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
    parser.add_argument("--revision", default="plain_text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_limit < 1 or args.validation_limit < 1:
        raise SystemExit("limits must be positive")
    from datasets import load_dataset

    squad = load_dataset("squad", revision=args.revision)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "squad_train.jsonl", records(squad["train"], args.train_limit))
    write_jsonl(
        output_dir / "squad_validation.jsonl", records(squad["validation"], args.validation_limit)
    )
    train, validation = output_dir / "squad_train.jsonl", output_dir / "squad_validation.jsonl"
    preprocessing = {
        "version": "squad-first-answer-v1",
        "field_mappings": {
            "context": "document",
            "question": "question",
            "answers.text[0]": "answer",
        },
        "filters": ["non-empty first answer"],
        "sample_limits": {"train": args.train_limit, "validation": args.validation_limit},
        "shuffle": None,
        "seed": args.seed,
    }
    manifest = dataset_manifest(
        upstream_dataset="squad",
        upstream_revision=args.revision,
        license="CC BY-SA 4.0",
        source_url_identifier="https://huggingface.co/datasets/squad",
        split=["train", "validation"],
        filters=preprocessing["filters"],
        sample_limits=preprocessing["sample_limits"],
        shuffle_operation=None,
        seed=args.seed,
        field_mappings=preprocessing["field_mappings"],
        preprocessing_configuration=preprocessing,
        preprocessing_version="squad-first-answer-v1",
        output_hashes={"train": file_hash(train), "validation": file_hash(validation)},
        row_counts={
            "train": len(records(squad["train"], args.train_limit)),
            "validation": len(records(squad["validation"], args.validation_limit)),
        },
        rejected_row_counts={"empty_first_answer": 0},
        rejection_reasons={"empty_first_answer": "first answer missing or blank"},
        generated_timestamp=timestamp(),
    )
    write_json(output_dir / "dataset_manifest.json", manifest)


if __name__ == "__main__":
    main()
