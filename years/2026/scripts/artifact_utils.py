#!/usr/bin/env python3
"""Small helpers for stamping and verifying generated planner artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_id() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path | str) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {
            "path": str(artifact_path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    return {
        "path": str(artifact_path),
        "exists": True,
        "size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
    }


def build_artifact_manifest(
    run_id: str,
    inputs: list[Path | str],
    outputs: list[Path | str],
    command: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "generated_at": utc_now_id(),
        "command": command,
        "inputs": [artifact_record(path) for path in inputs],
        "outputs": [artifact_record(path) for path in outputs],
        "metadata": metadata or {},
    }
    manifest["verification"] = verify_artifact_manifest(manifest)
    return manifest


def verify_artifact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    missing = []
    checked = 0
    for artifact in list(manifest.get("inputs") or []) + list(manifest.get("outputs") or []):
        path = Path(str(artifact.get("path")))
        if not path.exists():
            missing.append({"path": str(path), "reason": "missing"})
            continue
        checked += 1
        current = artifact_record(path)
        if current.get("sha256") != artifact.get("sha256") or current.get("size_bytes") != artifact.get("size_bytes"):
            mismatches.append(
                {
                    "path": str(path),
                    "expected_sha256": artifact.get("sha256"),
                    "actual_sha256": current.get("sha256"),
                    "expected_size_bytes": artifact.get("size_bytes"),
                    "actual_size_bytes": current.get("size_bytes"),
                }
            )
    return {
        "valid": not missing and not mismatches,
        "checked_artifacts": checked,
        "missing": missing,
        "mismatches": mismatches,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
