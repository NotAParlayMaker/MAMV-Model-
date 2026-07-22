#!/usr/bin/env python3
"""Validate and publish an existing, real checkpoint to Hugging Face Hub."""

from __future__ import annotations
import argparse
from pathlib import Path

REQUIRED = ("config.json",)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", default="v1.0")
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    missing = [name for name in REQUIRED if not (a.model_dir / name).is_file()]
    if not (a.model_dir / "README.md").is_file() and not Path("MODEL_CARD.md").is_file():
        missing.append("README.md or repository MODEL_CARD.md")
    if missing:
        raise SystemExit("Refusing to publish; missing: " + ", ".join(missing))
    from huggingface_hub import HfApi

    api = HfApi()
    api.whoami()
    api.create_repo(a.repo_id, private=a.private, exist_ok=True)
    api.upload_folder(
        repo_id=a.repo_id,
        folder_path=str(a.model_dir),
        revision=a.revision,
        commit_message=f"Publish MAMV {a.revision}",
    )


if __name__ == "__main__":
    main()
