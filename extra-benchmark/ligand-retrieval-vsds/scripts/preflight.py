#!/usr/bin/env python3
"""Verify every frozen source, container, checkpoint, and implementation hash."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
import subprocess

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    md5_file,
    protocol_digest,
    require_regular_file,
)


def main() -> None:
    protocol = load_protocol()
    verified: dict[str, str] = {}
    for item in protocol["data"].values():
        if "path" not in item:
            continue
        path = BENCHMARK_DIR / item["path"]
        verified[str(path)] = require_regular_file(
            path, item["sha256"], int(item["bytes"])
        )
        if "md5" in item and md5_file(path) != item["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path}")
    seen_containers: set[str] = set()
    for model in protocol["models"]["primary_order"]:
        item = protocol["models"][model]
        path = item["container"]
        if path not in seen_containers:
            verified[path] = require_regular_file(path, item["container_sha256"])
            seen_containers.add(path)
    for key in ("config", "training_plan", "checkpoint", "calibrator"):
        item = protocol["gmolai"][key]
        path = REPOSITORY_ROOT / item["path"]
        verified[str(path)] = require_regular_file(
            path, item["sha256"], int(item["bytes"])
        )
    for item in protocol["implementation_sources"].values():
        path = REPOSITORY_ROOT / item["path"]
        verified[str(path)] = require_regular_file(path, item["sha256"])
    for item in protocol["exposure_audit"]["corpus_artifacts"].values():
        path = REPOSITORY_ROOT / item["path"]
        verified[str(path)] = require_regular_file(
            path, item["sha256"], int(item["bytes"])
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if head != protocol["repository"]["head_at_freeze"]:
        raise RuntimeError(
            f"Repository HEAD differs from protocol freeze: {head} != "
            f"{protocol['repository']['head_at_freeze']}"
        )
    result = {
        "schema_version": 1,
        "status": "ok",
        "protocol_sha256": protocol_digest(protocol),
        "repository_head": head,
        "verified_files": verified,
        "python": platform.python_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = BENCHMARK_DIR / "audits" / "preflight.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

