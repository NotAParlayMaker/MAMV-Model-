"""Versioned, credential-free evidence contracts for training and evaluation."""

from __future__ import annotations
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical(value).encode())


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("datasets", "torch", "transformers", "peft", "accelerate", "dill"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dataset_manifest(**values: Any) -> dict[str, Any]:
    body = {"schema_version": SCHEMA_VERSION, **values}
    body["manifest_id"] = (
        "dataset-" + fingerprint({k: v for k, v in body.items() if k != "manifest_id"})[:16]
    )
    return body


def checkpoint_manifest(**values: Any) -> dict[str, Any]:
    body = {"schema_version": SCHEMA_VERSION, **values}
    body["checkpoint_id"] = (
        "checkpoint-" + fingerprint({k: v for k, v in body.items() if k != "checkpoint_id"})[:16]
    )
    return body


def unavailable(reason: str = "required labels unavailable") -> dict[str, str]:
    return {"status": "not_evaluated", "reason": reason}


def compare_evaluation_reports(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    fa, fb = a["frame"], b["frame"]
    keys = (
        "dataset_manifest_id",
        "split_fingerprint",
        "preprocessing_version",
        "metric_implementation_versions",
        "checkpoint_manifest_id",
        "reasoning_mode",
        "retrieval_configuration",
        "generation_configuration",
    )
    mismatches = [key for key in keys if fa.get(key) != fb.get(key)]
    status = (
        "directly_comparable"
        if not mismatches
        else (
            "partially_comparable"
            if fa.get("dataset_manifest_id") == fb.get("dataset_manifest_id")
            else "not_comparable"
        )
    )
    deltas = (
        {}
        if status == "not_comparable"
        else {
            k: b.get("summary_metrics", {}).get(k) - a.get("summary_metrics", {}).get(k)
            for k in set(a.get("summary_metrics", {})) & set(b.get("summary_metrics", {}))
            if isinstance(a["summary_metrics"][k], (int, float))
            and isinstance(b["summary_metrics"][k], (int, float))
        }
    )
    return {"classification": status, "incompatible_fields": mismatches, "metric_deltas": deltas}
