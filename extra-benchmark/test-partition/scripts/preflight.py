#!/usr/bin/env python3
"""Validate every frozen input and container before benchmark execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

from benchmark_io import (
    BENCHMARK_DIR,
    PROTOCOL_PATH,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    protocol_digest,
    require_hash,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-container-hashes",
        action="store_true",
        help="Developer-only fast check; production jobs must not use this flag.",
    )
    return parser.parse_args()


def command_output(*arguments: str) -> str:
    return subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if Path(protocol["repository"]["project_root"]).resolve() != REPOSITORY_ROOT:
        raise RuntimeError("Protocol repository root differs from this checkout")

    head = command_output("git", "rev-parse", "--short=7", "HEAD")
    if head != protocol["repository"]["expected_head"]:
        raise RuntimeError(
            f"Repository HEAD changed: expected {protocol['repository']['expected_head']}, "
            f"observed {head}"
        )
    status_lines = [
        line
        for line in command_output("git", "status", "--porcelain=v1").splitlines()
        if line
    ]
    outside_changes = [
        line for line in status_lines if "extra-benchmark/" not in line[3:]
    ]
    if outside_changes:
        raise RuntimeError(
            "Working tree has changes outside extra-benchmark: "
            + json.dumps(outside_changes)
        )

    verified: dict[str, str] = {}
    for name, specification in protocol["authoritative_panels"].items():
        path_value = specification.get("path")
        expected = specification.get("sha256")
        if path_value and expected:
            target = REPOSITORY_ROOT / path_value
            verified[f"authoritative_panels.{name}"] = require_hash(target, expected)

    if not args.skip_container_hashes:
        for name, specification in protocol["comparators"].items():
            path_value = specification.get("container")
            expected = specification.get("container_sha256")
            if path_value and expected:
                verified[f"comparators.{name}.container"] = require_hash(
                    path_value, expected
                )

    source_hashes: dict[str, str] = {}
    for path in sorted((BENCHMARK_DIR / "scripts").glob("*")):
        if path.is_file() and not path.is_symlink():
            source_hashes[str(path.relative_to(REPOSITORY_ROOT))] = sha256_file(path)
    source_hashes[str(PROTOCOL_PATH.relative_to(REPOSITORY_ROOT))] = sha256_file(
        PROTOCOL_PATH
    )

    report = {
        "schema_version": 1,
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "inference_only",
        "repository_head": head,
        "working_tree_changes": status_lines,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_semantic_digest": protocol_digest(protocol),
        "verified_artifacts": verified,
        "benchmark_source_sha256": source_hashes,
        "host": platform.node(),
        "architecture": platform.machine(),
        "container_hashes_skipped": bool(args.skip_container_hashes),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "preflight.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
