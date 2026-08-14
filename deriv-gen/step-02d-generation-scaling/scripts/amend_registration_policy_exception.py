#!/usr/bin/env python3
"""Register pre-selection handling for RDKit sanitation exceptions."""

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
    superseded_path = root / "state" / "REGISTERED_SUPERSEDED_POLICY_EXCEPTION.json"
    existing = load_json(registration_path)
    if superseded_path.exists():
        print(json.dumps(existing, sort_keys=True))
        return
    if not (root / "state" / "DEVELOPMENT_GENERATION_COMPLETE.json").is_file():
        raise RuntimeError("Expected sealed development generation")
    forbidden = [
        root / "state" / "DEVELOPMENT_ANALYSIS_COMPLETE.json",
        root / "state" / "STRATEGY_FROZEN.json",
        root / "state" / "FINAL_GENERATION_COMPLETE.json",
        root / "state" / "FINAL_ANALYSIS_COMPLETE.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Policy-exception amendment must precede metrics and selection")
    if list((root / "outputs" / "tables").glob("development_*")):
        raise RuntimeError("Development metric tables already exist")
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
    amended["pre_selection_analysis_amendment"] = {
        "reason": "RDKit AtomValenceException during fragment sanitation was uncaught by the reused Step-2c raw-string auditor",
        "failed_job_id": "1251186",
        "development_generation_sealed_and_preserved": True,
        "development_metrics_produced_before_amendment": False,
        "strategy_selected_before_amendment": False,
        "candidate_contents_inspected": False,
        "handling": "classify parse or sanitation exceptions as RDKit-invalid and policy-rejected",
        "chemistry_policy_changed": False,
        "generation_stream_changed": False,
        "selection_rule_changed": False,
        "superseded_registration_path": superseded_path.relative_to(root).as_posix(),
        "superseded_registration_sha256": sha256_file(superseded_path),
    }
    atomic_write_json(registration_path, amended, root)
    print(json.dumps(amended, sort_keys=True))


if __name__ == "__main__":
    main()
