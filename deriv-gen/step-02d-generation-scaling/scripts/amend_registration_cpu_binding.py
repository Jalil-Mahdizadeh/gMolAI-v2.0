#!/usr/bin/env python3
"""Supersede registration for Arrhenius CPU-affinity launch portability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import STEP_ROOT, atomic_write_json, load_json, sha256_file, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    root = args.step_root.resolve()
    registration_path = root / "state" / "REGISTERED.json"
    superseded_path = root / "state" / "REGISTERED_SUPERSEDED_SLURM_CPU_BINDING.json"
    existing = load_json(registration_path)
    if superseded_path.exists():
        print(json.dumps(existing, sort_keys=True))
        return
    forbidden_seals = list((root / "state").glob("DEVELOPMENT_SHARD_*_COMPLETE.json"))
    forbidden_seals.extend((root / "state").glob("DEVELOPMENT_GENERATION_COMPLETE.json"))
    forbidden_seals.extend((root / "state").glob("DEVELOPMENT_ANALYSIS_COMPLETE.json"))
    forbidden_seals.extend((root / "state").glob("STRATEGY_FROZEN.json"))
    forbidden_seals.extend((root / "state").glob("FINAL_*.json"))
    result_files = [
        path
        for phase in ("development", "final")
        for path in (root / "outputs" / "raw" / phase).glob("*")
        if path.is_file()
    ]
    if forbidden_seals or result_files:
        raise RuntimeError("CPU-binding amendment must occur before any candidate result")
    atomic_write_json(superseded_path, existing, root)
    sources = sorted((root / "scripts").glob("*.py"))
    sources.extend(sorted((root / "scripts").glob("*.sh")))
    sources.extend(sorted((root / "scripts").glob("*.slurm")))
    sources.extend(sorted((root / "config").glob("*.json")))
    sources.extend([root / "PROTOCOL.md", root / "DESIGN.md"])
    amended = dict(existing)
    amended["registered_at"] = utc_now()
    amended["registered_source_sha256"] = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sources
        if path.is_file()
    }
    amendment_history = list(amended.get("pre_result_runtime_amendment_history", []))
    first = amended.pop("pre_result_runtime_amendment", None)
    if first is not None:
        amendment_history.append(first)
    amendment_history.append(
        {
            "reason": "Arrhenius rejected the default CPU affinity mask before task launch",
            "failed_job_id": "1251183",
            "failed_job_state": "FAILED",
            "sealed_candidate_outputs_from_failed_job": 0,
            "unsealed_candidate_outputs_from_failed_job": 0,
            "candidate_contents_inspected": False,
            "scientific_protocol_changed": False,
            "generation_or_analysis_code_changed": False,
            "runtime_change": "add cpu-bind=none; each GPU process still uses OMP_NUM_THREADS=18",
            "superseded_registration_path": superseded_path.relative_to(root).as_posix(),
            "superseded_registration_sha256": sha256_file(superseded_path),
        }
    )
    amended["pre_result_runtime_amendment_history"] = amendment_history
    atomic_write_json(registration_path, amended, root)
    print(json.dumps(amended, sort_keys=True))


if __name__ == "__main__":
    main()
