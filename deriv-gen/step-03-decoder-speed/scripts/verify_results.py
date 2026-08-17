#!/usr/bin/env python3
"""Verify workload cardinality, metric arithmetic, figures, and provenance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import atomic_write_json, load_json, sha256_file, utc_now


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.step_root.resolve()
    config = load_json(root / "config" / "benchmark.json")
    summary = load_json(root / "outputs" / "benchmark_summary.json")
    selected = pd.read_csv(root / "inputs" / "selected_panel.csv")
    conditions = np.load(root / "inputs" / "selected_conditions.npy", allow_pickle=False)
    seeds = pd.read_csv(root / "outputs" / "tables" / "per_seed_metrics.csv")
    batches = pd.read_csv(root / "outputs" / "tables" / "per_batch_timings.csv")
    raw = pq.read_table(root / "outputs" / "raw" / "proposals.parquet").to_pandas()
    checks: dict[str, bool] = {}
    expected_seeds = int(config["selection"]["count"])
    draws = int(config["decoder"]["draws_per_seed"])
    batch_size = int(config["decoder"]["query_batch_size"])
    expected_raw = expected_seeds * draws
    checks["selected_panel_has_100_rows"] = len(selected) == expected_seeds == 100
    checks["selected_conditions_are_100_by_384_float32"] = (
        conditions.shape == (expected_seeds, 384)
        and conditions.dtype == np.float32
        and bool(np.isfinite(conditions).all())
    )
    checks["selected_hashes_are_unique"] = selected["target_hash"].nunique() == expected_seeds
    checks["one_visible_cuda_gpu_recorded"] = (
        summary["execution"]["cuda_visible_device_count"] == 1
    )
    checks["raw_table_has_100000_rows"] = len(raw) == expected_raw == 100000
    per_seed_raw = raw.groupby("benchmark_seed_index").size().sort_index()
    checks["every_seed_has_exactly_1000_draws"] = (
        len(per_seed_raw) == expected_seeds and bool((per_seed_raw == draws).all())
    )
    draw_integrity = raw.groupby("benchmark_seed_index")["draw_index"].agg(
        ["min", "max", "nunique"]
    )
    checks["draw_indices_are_complete"] = bool(
        (draw_integrity["min"] == 1).all()
        and (draw_integrity["max"] == draws).all()
        and (draw_integrity["nunique"] == draws).all()
    )
    checks["seed_table_has_100_rows"] = len(seeds) == expected_seeds
    checks["batch_table_has_50_two_seed_batches"] = (
        len(batches) == expected_seeds // batch_size
        and bool((batches["seed_count"] == batch_size).all())
    )
    raw_rdkit_unique = int(raw["is_first_rdkit_unique_within_seed"].sum())
    raw_policy_unique = int(raw["is_first_policy_unique_within_seed"].sum())
    checks["rdkit_unique_counts_reconcile"] = (
        raw_rdkit_unique
        == int(seeds["rdkit_unique_valid_molecules"].sum())
        == summary["counts"]["per_seed_unique_rdkit_valid_molecules"]
    )
    checks["policy_unique_counts_reconcile"] = (
        raw_policy_unique
        == int(seeds["release_policy_unique_molecules"].sum())
        == summary["counts"]["per_seed_unique_release_policy_molecules"]
    )
    generation_seconds = float(batches["generation_seconds"].sum())
    valid_seconds = float(
        (
            batches["generation_seconds"]
            + batches["token_decode_seconds"]
            + batches["rdkit_validation_seconds"]
        ).sum()
    )
    checks["raw_throughput_arithmetic"] = close(
        summary["metrics"]["raw_proposals_per_second"]["value"],
        expected_raw / generation_seconds,
    )
    checks["valid_unique_throughput_arithmetic"] = close(
        summary["metrics"]["valid_unique_molecules_per_second"]["value"],
        raw_rdkit_unique / valid_seconds,
    )
    plot_headline = pd.read_csv(root / "outputs" / "plot-data" / "headline_throughput.csv")
    checks["headline_plot_has_requested_two_metrics"] = set(plot_headline["metric"]) == {
        "raw_proposals_per_second",
        "valid_unique_molecules_per_second",
    }
    figure_paths = [
        root / "figures" / f"{stem}.{suffix}"
        for stem in (
            "decoder_throughput",
            "batch_throughput_trace",
            "per_seed_valid_unique_yield",
        )
        for suffix in ("png", "svg")
    ]
    checks["all_six_figure_files_are_nonempty"] = all(
        path.is_file() and path.stat().st_size > 1000 for path in figure_paths
    )
    checks["results_document_exists"] = (
        (root / "RESULTS.md").is_file() and (root / "RESULTS.md").stat().st_size > 1000
    )
    required_paths = [
        root / "README.md",
        root / "PROTOCOL.md",
        root / "run_benchmark.sh",
        root / "config" / "benchmark.json",
        root / "scripts" / "common.py",
        root / "scripts" / "prepare_inputs.py",
        root / "scripts" / "run_benchmark.py",
        root / "scripts" / "plot_results.py",
        root / "scripts" / "report_results.py",
        root / "scripts" / "verify_results.py",
        root / "inputs" / "selected_panel.csv",
        root / "inputs" / "selected_conditions.npy",
        root / "inputs" / "selection_metadata.json",
        root / "outputs" / "benchmark_summary.json",
        root / "outputs" / "tables" / "benchmark_summary.csv",
        root / "outputs" / "tables" / "per_batch_timings.csv",
        root / "outputs" / "tables" / "per_seed_metrics.csv",
        root / "outputs" / "raw" / "proposals.parquet",
        root / "outputs" / "plot-data" / "headline_throughput.csv",
        root / "outputs" / "plot-data" / "batch_throughput.csv",
        root / "outputs" / "plot-data" / "per_seed_yield.csv",
        root / "RESULTS.md",
        *figure_paths,
    ]
    checks["all_required_artifacts_exist"] = all(path.is_file() for path in required_paths)
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Verification failed: {failures}")

    manifest_rows = []
    for path in sorted(required_paths):
        manifest_rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = root / "outputs" / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    result = {
        "schema_version": 1,
        "verified_utc": utc_now(),
        "status": "pass",
        "checks": checks,
        "check_count": len(checks),
        "artifact_manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(root / "outputs" / "verification.json", result)
    print(f"PASS: {len(checks)} checks")


if __name__ == "__main__":
    main()
