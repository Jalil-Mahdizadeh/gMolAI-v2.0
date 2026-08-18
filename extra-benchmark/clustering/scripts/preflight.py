#!/usr/bin/env python3
"""Verify every frozen source, model container, checkpoint, and scaler hash."""

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
    protocol_digest,
    require_regular_file,
)


def main() -> None:
    protocol = load_protocol()
    verified: dict[str, str] = {}
    for source in protocol["data"].values():
        path = BENCHMARK_DIR / source["path"]
        verified[str(path)] = require_regular_file(path, source["sha256"])
        if path.stat().st_size != int(source["bytes"]):
            raise RuntimeError(f"Byte-size mismatch for {path}")
    for model in protocol["models"].values():
        if not isinstance(model, dict):
            continue
        path = model.get("container")
        if path:
            verified[path] = require_regular_file(path, model["container_sha256"])
    for key in ("config", "training_plan", "checkpoint", "calibrator"):
        item = protocol["gmolai"][key]
        path = REPOSITORY_ROOT / item["path"]
        verified[str(path)] = require_regular_file(path, item["sha256"])
    scaler = protocol["descriptor_diagnostic"]["scaler"]
    scaler_path = REPOSITORY_ROOT / scaler["path"]
    verified[str(scaler_path)] = require_regular_file(scaler_path, scaler["sha256"])
    for source in protocol["implementation_sources"].values():
        path = REPOSITORY_ROOT / source["path"]
        verified[str(path)] = require_regular_file(path, source["sha256"])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True,
        text=True, capture_output=True
    ).stdout.strip()
    if head != protocol["repository"]["head_at_freeze"]:
        raise RuntimeError(f"Repository HEAD changed: {head}")
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
    output = BENCHMARK_DIR / "audit" / "preflight.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
