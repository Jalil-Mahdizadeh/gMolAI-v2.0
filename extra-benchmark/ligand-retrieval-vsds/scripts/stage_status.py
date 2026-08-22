#!/usr/bin/env python3
"""Validate a completed benchmark-stage manifest for safe runner reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    load_json,
    load_protocol,
    protocol_digest,
    sha256_file,
)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repository_path = REPOSITORY_ROOT / path
    if repository_path.exists():
        return repository_path
    return BENCHMARK_DIR / path


def verify_file(value: str, expected: str) -> None:
    path = resolve_path(value)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or unsafe stage artifact: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Stage artifact changed: {path}: expected {expected}, observed {observed}"
        )


def verify_tree(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            verify_tree(item)
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
        verify_file(value["path"], value["sha256"])
    for key, expected in value.items():
        if not key.endswith("_sha256") or not isinstance(expected, str):
            continue
        stem = key[: -len("_sha256")]
        path_value = value.get(stem)
        if not isinstance(path_value, str):
            path_value = value.get(stem + "_path")
        if isinstance(path_value, str):
            verify_file(path_value, expected)
    for key, item in value.items():
        if key != "verified_files":
            verify_tree(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    path = args.manifest
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    manifest = load_json(path)
    if manifest.get("status") not in {"ok", "frozen", "complete"}:
        raise RuntimeError(f"Incomplete stage manifest: {path}")
    verified_files = manifest.get("verified_files", {})
    if isinstance(verified_files, dict):
        for filename, expected in verified_files.items():
            verify_file(str(filename), str(expected))
    verify_tree(manifest)
    if path.name == "preflight.json":
        protocol = load_protocol()
        if manifest.get("protocol_sha256") != protocol_digest(protocol):
            raise RuntimeError("Preflight protocol digest changed")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if manifest.get("repository_head") != head:
            raise RuntimeError("Preflight repository HEAD changed")
    if path.name == "COMPLETE.json":
        checksum_path = resolve_path(str(manifest["sha256_manifest"]))
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            verify_file(str(BENCHMARK_DIR / relative), expected)
    print(json.dumps({"status": "ok", "reused": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
