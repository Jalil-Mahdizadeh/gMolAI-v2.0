#!/usr/bin/env python3
"""Seal a successful no-training benchmark with hashes and COMPLETE state."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    sha256_file,
)


def main() -> None:
    protocol = load_protocol()
    summary_path = BENCHMARK_DIR / "outputs" / "test_partition_summary.json"
    summary = load_json(summary_path)
    if summary.get("status") != "ok" or summary.get("execution") != "inference_only":
        raise RuntimeError("Benchmark summary is incomplete or not inference-only")

    checksum_paths = []
    for directory_name in ("inputs", "outputs", "state"):
        directory = BENCHMARK_DIR / directory_name
        for path in directory.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name in {".gitkeep", "SHA256SUMS", "COMPLETE.json", "status.json"}:
                continue
            if path.name.endswith(".partial"):
                raise RuntimeError(f"Incomplete temporary artifact remains: {path}")
            checksum_paths.append(path)
    checksum_paths.sort(key=lambda value: str(value.relative_to(BENCHMARK_DIR)))
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(BENCHMARK_DIR)}\n"
        for path in checksum_paths
    )
    checksum_path = BENCHMARK_DIR / "outputs" / "SHA256SUMS"
    atomic_write_text(checksum_path, checksum_text)

    complete = {
        "schema_version": 1,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "inference_only",
        "training_performed": False,
        "model_weights_modified": False,
        "gmolai_checkpoint_or_calibrator_modified": False,
        "moleculenet_executed": False,
        "protocol_sha256": sha256_file(BENCHMARK_DIR / "protocol.json"),
        "summary_sha256": sha256_file(summary_path),
        "sha256sums_sha256": sha256_file(checksum_path),
        "checksummed_files": len(checksum_paths),
        "common_train_rows": summary["common_train_probe_rows"],
        "common_test_rows": summary["common_test_rows"],
        "comparators": sorted(protocol["comparators"]),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
