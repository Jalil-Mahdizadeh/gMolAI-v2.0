#!/usr/bin/env python3
"""Read-only integrity and completeness verifier for the Day-1 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from day1_common import sha256_file


SPACE_SET = {"graph_256", "mean_node_128", "hybrid_384"}
METHOD_SET = {
    "seed_nn",
    "isotropic",
    "global_covariance",
    "local_covariance",
    "mmp_direction",
    "interpolation_0.25",
    "interpolation_0.50",
    "interpolation_0.75",
    "interpolation_1.00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01-latent-geometry-retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    expected_root = (repo_root / "deriv-gen" / step_root.name).resolve()
    if step_root != expected_root:
        raise RuntimeError(f"Unexpected step root: {step_root}")
    required = [
        step_root / "RESULTS.md",
        step_root / "state" / "COMPLETE.json",
        step_root / "state" / "run_metadata.json",
        step_root / "outputs" / "study_summary.json",
        step_root / "outputs" / "SHA256SUMS",
        step_root / "outputs" / "tables" / "global_geometry.csv",
        step_root / "outputs" / "tables" / "local_geometry_summary.csv",
        step_root / "outputs" / "tables" / "distance_chemistry_summary.csv",
        step_root / "outputs" / "tables" / "mmp_alignment_summary.csv",
        step_root / "outputs" / "tables" / "retrieval_summary.csv",
        step_root / "outputs" / "raw" / "retrieval_per_query.parquet",
        step_root / "intermediate" / "molecules.parquet",
        step_root / "outputs" / "figures" / "global_spectrum.png",
        step_root / "outputs" / "figures" / "retrieval_recall10.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")

    manifest = json.loads(
        (step_root / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    for role, record in manifest["files"].items():
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Input hash mismatch for {role}")
        if role != "container" and any(
            token in str(path).lower()
            for token in ("test-partition", "test-standardized", "moleculenet", "hiv")
        ):
            raise RuntimeError(f"Forbidden input in manifest: {path}")

    outputs_root = step_root / "outputs"
    ledger = outputs_root / "SHA256SUMS"
    ledger_rows = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = (outputs_root / relative).resolve()
        if outputs_root.resolve() not in path.parents:
            raise RuntimeError(f"Ledger path escapes outputs: {relative}")
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Output hash mismatch: {relative}")
        ledger_rows += 1
    if ledger_rows < 15:
        raise RuntimeError(f"Suspiciously short output ledger: {ledger_rows}")

    complete = json.loads(
        (step_root / "state" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    if complete.get("status") != "complete" or complete.get("test_rows") != 0:
        raise RuntimeError("Completion seal is not valid")
    if not complete.get("single_gpu"):
        raise RuntimeError("Completion seal does not record single-GPU execution")
    if sha256_file(step_root / "RESULTS.md") != complete["results_sha256"]:
        raise RuntimeError("RESULTS.md identity mismatch")
    if sha256_file(ledger) != complete["output_ledger_sha256"]:
        raise RuntimeError("Output ledger identity mismatch")

    geometry = pd.read_csv(outputs_root / "tables" / "global_geometry.csv")
    retrieval = pd.read_csv(outputs_root / "tables" / "retrieval_summary.csv")
    alignment = pd.read_csv(outputs_root / "tables" / "mmp_alignment_summary.csv")
    if set(geometry["space"]) != SPACE_SET or set(alignment["space"]) != SPACE_SET:
        raise RuntimeError("One or more coordinate spaces are missing")
    for space in SPACE_SET:
        observed_methods = set(retrieval.loc[retrieval["space"] == space, "method"])
        if observed_methods != METHOD_SET:
            raise RuntimeError(
                f"Method set mismatch for {space}: {sorted(observed_methods)}"
            )
    numeric = retrieval.select_dtypes(include="number")
    if numeric.isna().all(axis=None):
        raise RuntimeError("Retrieval summary contains no numeric results")

    retrieval_raw = pd.read_parquet(
        outputs_root / "raw" / "retrieval_per_query.parquet"
    )
    identity_columns = ("seed_hash", "true_target_hash", "top1_candidate_hash")
    for column in identity_columns:
        if column not in retrieval_raw:
            raise RuntimeError(f"Missing retrieval identity column: {column}")
        if not retrieval_raw[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise RuntimeError(f"Invalid or blank retrieval identity: {column}")
    molecules = pd.read_parquet(step_root / "intermediate" / "molecules.parquet")
    validation_molecules = molecules.loc[
        molecules["split"] == "validation", ["molecule_index", "molecule_hash"]
    ]
    if validation_molecules["molecule_index"].duplicated().any():
        raise RuntimeError("Duplicate validation molecule index")
    hash_by_index = validation_molecules.set_index("molecule_index")["molecule_hash"]
    identity_pairs = (
        ("seed_index", "seed_hash"),
        ("true_target_index", "true_target_hash"),
        ("top1_candidate_index", "top1_candidate_hash"),
    )
    for index_column, hash_column in identity_pairs:
        expected = retrieval_raw[index_column].map(hash_by_index)
        observed = retrieval_raw[hash_column].astype(str)
        if expected.isna().any() or not expected.astype(str).reset_index(
            drop=True
        ).equals(observed.reset_index(drop=True)):
            raise RuntimeError(f"Retrieval identity/index mismatch: {hash_column}")

    summary = json.loads(
        (outputs_root / "study_summary.json").read_text(encoding="utf-8")
    )
    if summary["data"]["identity_overlap"] != 0:
        raise RuntimeError("Train/validation overlap is nonzero")
    if summary["data"]["train_rows"] != 100000:
        raise RuntimeError("Train row count mismatch")
    if summary["data"]["validation_rows"] != 50000:
        raise RuntimeError("Validation row count mismatch")

    print(
        json.dumps(
            {
                "status": "verified",
                "ledger_entries": ledger_rows,
                "spaces": sorted(SPACE_SET),
                "methods_per_space": len(METHOD_SET),
                "retrieval_queries": complete["retrieval_queries"],
                "test_rows": complete["test_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

