#!/usr/bin/env python3
"""Validate the frozen panel, identities, model sources and containers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
    read_panel_tsv,
    require_hash,
    sha256_file,
    sha256_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-container-hashes",
        action="store_true",
        help="Developer-only syntax check; production execution must not use it.",
    )
    parser.add_argument(
        "--allow-existing-results",
        action="store_true",
        help="Validation-only mode; production execution starts with no raw results.",
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
    if Path(protocol["repository"]["root"]).resolve() != REPOSITORY_ROOT:
        raise RuntimeError("Protocol repository root differs from this checkout")

    head = command_output("git", "rev-parse", "HEAD")
    if head != protocol["repository"]["expected_head"]:
        raise RuntimeError(
            f"Repository HEAD changed: expected {protocol['repository']['expected_head']}, "
            f"observed {head}"
        )

    panel = protocol["panel"]
    source_path = REPOSITORY_ROOT / panel["source_path"]
    local_path = REPOSITORY_ROOT / panel["local_path"]
    require_hash(source_path, panel["tsv_sha256"])
    require_hash(local_path, panel["tsv_sha256"])
    rows = read_panel_tsv(local_path)
    if len(rows) != int(panel["rows"]):
        raise RuntimeError(f"Panel row mismatch: {len(rows)} != {panel['rows']}")
    identities = [row["molecule_hash"] for row in rows]
    if len(set(identities)) != int(panel["unique_identities"]):
        raise RuntimeError("Panel identities are not unique")
    if sha256_lines(identities) != panel["ordered_identity_sha256"]:
        raise RuntimeError("Ordered identity digest differs from the frozen panel")
    for index, row in enumerate(rows):
        observed = hashlib.sha256(row["canonical_smiles"].encode("utf-8")).hexdigest()
        if observed != row["molecule_hash"]:
            raise RuntimeError(f"Collision-safe identity verification failed at row {index}")

    verified: dict[str, str] = {
        "panel.source": sha256_file(source_path),
        "panel.local": sha256_file(local_path),
    }
    for name, specification in protocol["gmolai_sources"].items():
        relative = specification.get("path")
        if relative:
            verified[f"gmolai_sources.{name}"] = require_hash(
                REPOSITORY_ROOT / relative, specification["sha256"]
            )
    if not args.skip_container_hashes:
        seen: set[tuple[str, str]] = set()
        for name in protocol["model_order"]:
            specification = protocol["models"][name]
            key = (specification["container"], specification["container_sha256"])
            if key in seen:
                continue
            seen.add(key)
            verified[f"container.{name}"] = require_hash(*key)

    raw_results = sorted((BENCHMARK_DIR / "outputs" / "raw").glob("*.json"))
    if raw_results and not args.allow_existing_results:
        raise RuntimeError(
            "Raw result directory is not empty; refusing to mix timing jobs: "
            + json.dumps([path.name for path in raw_results])
        )

    source_hashes: dict[str, str] = {}
    for path in sorted((BENCHMARK_DIR / "scripts").glob("*.py")):
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
        "working_tree_status": command_output("git", "status", "--porcelain=v1").splitlines(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "protocol_semantic_digest": protocol_digest(protocol),
        "panel_rows": len(rows),
        "ordered_identity_sha256": sha256_lines(identities),
        "collision_safe_identity_verification": True,
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

