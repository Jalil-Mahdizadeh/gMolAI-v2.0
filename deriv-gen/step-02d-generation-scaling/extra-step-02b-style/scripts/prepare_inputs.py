#!/usr/bin/env python3
"""Hash immutable inputs and create the aligned unique-molecule encoder CSV."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    atomic_write_csv,
    atomic_write_json,
    input_paths,
    load_config,
    load_json,
    require_analysis_root,
    require_repo_root,
    sha256_file,
    validate_inputs,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def file_record(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": relative_or_absolute(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = require_repo_root(args.repo_root)
    root = require_analysis_root(args.analysis_root)
    cfg = load_config(root)
    paths = input_paths(repo_root)
    validate_inputs(paths)
    step_summary_path = (
        repo_root
        / "deriv-gen"
        / "step-02d-generation-scaling"
        / "outputs"
        / "final_analysis_summary.json"
    )
    if not step_summary_path.is_file():
        raise RuntimeError(f"Missing immutable input: {step_summary_path}")
    for directory in (
        "inputs",
        "intermediate",
        "outputs/tables",
        "outputs/plot-data",
        "outputs/figures",
        "logs",
        "state/tmp",
        "state/matplotlib",
        "state/xdg-cache",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    state_path = root / "state" / "PREPARED.json"
    csv_path = root / "inputs" / "encoder_input.csv"
    manifest_path = root / "inputs" / "manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if not csv_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Prepared state exists but prepared inputs are incomplete")
        if sha256_file(csv_path) != state["encoder_input_sha256"]:
            raise RuntimeError("Prepared encoder input changed after sealing")
        print(json.dumps(state, sort_keys=True))
        return
    if csv_path.exists() or manifest_path.exists():
        raise RuntimeError("Unsealed prepared input exists; refusing to overwrite it")

    started = time.monotonic()
    unique_path = Path(paths["unique_molecules"])
    unique = pd.read_parquet(
        unique_path,
        columns=["structure_index", "molecule_hash", "canonical_smiles"],
    ).sort_values("structure_index", ignore_index=True)
    expected_indices = np.arange(len(unique), dtype=np.int64)
    if not np.array_equal(unique["structure_index"].to_numpy(dtype=np.int64), expected_indices):
        raise RuntimeError("Step 2d structure_index is not contiguous and ordered")
    if unique["molecule_hash"].nunique() != len(unique):
        raise RuntimeError("Step 2d unique-molecule identities are not unique")
    step_summary = load_json(step_summary_path)
    expected_rows = int(step_summary["unique_molecules_fragmented"])
    if len(unique) != expected_rows:
        raise RuntimeError(f"Unique-molecule rows changed: {len(unique)} != {expected_rows}")
    encoder_input = unique[["molecule_hash", "canonical_smiles"]].rename(
        columns={"molecule_hash": "molecule_id", "canonical_smiles": "smiles"}
    )
    atomic_write_csv(csv_path, encoder_input, root)

    verification = load_json(Path(paths["step2d_verification"]))
    expected_hashes = verification["input_sha256"]
    frozen_checks = {
        "encoder_checkpoint": "packaged_checkpoint",
        "encoder_calibrator": "packaged_calibrator",
        "encoder_config": "packaged_resolved_config",
        "encoder_model_source": "gmolai_model_definition",
        "optimized_inference_source": "optimized_inference",
    }
    for local_role, frozen_role in frozen_checks.items():
        observed = sha256_file(Path(paths[local_role]))
        expected = str(expected_hashes[frozen_role])
        if observed != expected:
            raise RuntimeError(
                f"Frozen input hash changed for {local_role}: {observed} != {expected}"
            )

    records: dict[str, Any] = {}
    for role, value in paths.items():
        if isinstance(value, list):
            records[role] = [file_record(path, repo_root) for path in value]
        else:
            records[role] = file_record(value, repo_root)
    records["step2d_final_analysis_summary"] = file_record(
        step_summary_path, repo_root
    )
    records["analysis_config"] = file_record(root / "config" / "analysis.json", repo_root)
    records["analysis_protocol"] = file_record(root / "PROTOCOL.md", repo_root)
    records["encoder_input"] = file_record(csv_path, repo_root)
    manifest = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "created_at": utc_now(),
        "immutable_inputs": records,
        "encoder_input_rows": len(encoder_input),
        "encoder_input_order": "Step 2d final structure_index ascending",
        "reencoding": cfg["reencoding"],
        "writes_allowed_only_under": str(root),
    }
    atomic_write_json(manifest_path, manifest, root)
    state = {
        "schema_version": 1,
        "status": "prepared",
        "study_id": cfg["study_id"],
        "prepared_at": utc_now(),
        "encoder_input_rows": len(encoder_input),
        "encoder_input_sha256": sha256_file(csv_path),
        "manifest_sha256": sha256_file(manifest_path),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
