#!/usr/bin/env python3
"""Fail closed unless every phase-two source and executable is frozen."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_json,
    load_protocol,
    protocol_digest,
    require_hash,
)


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def main() -> None:
    protocol = load_protocol()
    if (BENCHMARK_DIR / "state" / "COMPLETE.json").exists():
        raise RuntimeError("Benchmark is already complete; refusing to rerun it")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if head != protocol["repository"]["expected_head"]:
        raise RuntimeError(
            f"Repository HEAD changed: {head} != {protocol['repository']['expected_head']}"
        )

    verified = []
    for category in ("files", "containers"):
        for item in protocol["integrity"][category]:
            target = resolve(item["path"])
            require_hash(target, item["sha256"])
            verified.append({"category": category, "path": str(target), "sha256": item["sha256"]})

    development = load_json(
        resolve(protocol["references"]["development"]["path"])
    )
    hiv = load_json(resolve(protocol["references"]["hiv"]["path"]))
    expected_development = {"bace", "bbbp", "esol", "freesolv", "lipophilicity"}
    if set(development.get("datasets", {})) != expected_development:
        raise RuntimeError("Development reference dataset panel changed")
    if set(hiv.get("datasets", {})) != {"hiv"}:
        raise RuntimeError("HIV reference dataset panel changed")
    if set(hiv["datasets"]["hiv"]["feature_results"]) < {
        "molecule_embedding",
        "morgan_radius2_2048",
    }:
        raise RuntimeError("HIV reference lacks gMolAI or Morgan")
    for reference in (development, hiv):
        checkpoint = reference["checkpoint"]
        if int(checkpoint["global_step"]) != 10_000:
            raise RuntimeError("Reference does not bind to selected step 10,000")
        if checkpoint["checkpoint_sha256"] != protocol["sources"]["checkpoint"]["sha256"]:
            raise RuntimeError("Reference checkpoint differs from frozen source")

    qualification = load_json(
        resolve(protocol["references"]["adapter_qualification_complete"]["path"])
    )
    if qualification.get("status") != "complete":
        raise RuntimeError("Prior validation/test adapter qualification is incomplete")
    if set(qualification.get("comparators", [])) != set(protocol["comparators"]["model_order"]):
        raise RuntimeError("Prior qualification comparator set changed")

    report = {
        "schema_version": 1,
        "status": "ok",
        "protocol_sha256": protocol_digest(protocol),
        "repository_head": head,
        "verified_files": len(verified),
        "verified": verified,
        "neural_training_permitted": False,
        "single_gpu_required": True,
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "preflight.json", report)
    print(json.dumps({key: report[key] for key in report if key != "verified"}, sort_keys=True))


if __name__ == "__main__":
    main()
