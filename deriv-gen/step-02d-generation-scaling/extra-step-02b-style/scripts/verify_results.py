#!/usr/bin/env python3
"""Fail-closed verification of the isolated Step 2b-style analysis outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from common import (
    atomic_write_json,
    load_config,
    load_json,
    require_analysis_root,
    require_repo_root,
    sha256_file,
)


PLOT_STEMS = [
    "candidate_quality_and_availability",
    "exact_identity_recovery",
    "top1_structural_fidelity",
    "reranked_latent_metrics",
    "reranking_effect",
]

REQUIRED_METRICS = {
    "raw_token_decode_fraction",
    "raw_rdkit_valid_fraction",
    "raw_policy_acceptance_fraction",
    "unique_identity_yield_per_raw_slot",
    "mean_unique_accepted_identities",
    "candidate_availability_rate",
    "greedy_rdkit_valid_top1_rate",
    "greedy_policy_accepted_top1_rate",
    "greedy_exact_seed_identity_at_1",
    "greedy_seed_scaffold_recovery",
    "greedy_mean_morgan_to_seed",
    "greedy_conditional_mean_morgan_to_seed",
    "generator_order_valid_top1_rate",
    "generator_order_exact_seed_identity_at_1",
    "generator_order_seed_scaffold_recovery",
    "generator_order_mean_morgan_to_seed",
    "generator_order_conditional_mean_morgan_to_seed",
    "exact_seed_oracle_recall_at_budget",
    "reranked_valid_top1_rate",
    "reranked_exact_seed_identity_at_1",
    "rerank_selection_efficiency_given_oracle_presence",
    "reranked_identity_gain_over_greedy",
    "reranked_seed_scaffold_recovery",
    "reranked_mean_morgan_to_seed",
    "reranked_conditional_mean_morgan_to_seed",
    "reranked_mean_latent_l2",
    "reranked_mean_latent_relative_l2",
    "reranked_mean_latent_cosine",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_manifest_path(recorded: str, repo_root: Path) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else repo_root / path


def verify_manifest(root: Path, repo_root: Path) -> int:
    manifest_path = root / "inputs" / "manifest.json"
    manifest = load_json(manifest_path)
    checked = 0
    for role, value in manifest["immutable_inputs"].items():
        records = value if isinstance(value, list) else [value]
        for record in records:
            path = resolve_manifest_path(str(record["path"]), repo_root)
            require(path.is_file(), f"Manifest input is missing for {role}: {path}")
            require(path.stat().st_size == int(record["bytes"]), f"Manifest byte size changed for {role}: {path}")
            require(sha256_file(path) == str(record["sha256"]), f"Manifest hash changed for {role}: {path}")
            checked += 1
    return checked


def verify_ledger(root: Path) -> int:
    ledger = root / "outputs" / "SHA256SUMS"
    require(ledger.is_file(), "Missing output SHA256 ledger")
    checked = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", maxsplit=1)
        path = root / relative
        require(path.is_file(), f"Ledger output is missing: {relative}")
        require(sha256_file(path) == digest, f"Ledger hash mismatch: {relative}")
        checked += 1
    require(checked > 0, "Output SHA256 ledger is empty")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = require_repo_root(args.repo_root)
    root = require_analysis_root(args.analysis_root)
    config = load_config(root)
    verified_state = root / "state" / "VERIFIED.json"
    if verified_state.exists():
        print(json.dumps(load_json(verified_state), sort_keys=True))
        return
    require((root / "state" / "COMPONENT_TESTS.json").is_file(), "Component tests were not run")
    require((root / "state" / "PREPARED.json").is_file(), "Inputs were not prepared")
    require((root / "state" / "ANALYSIS_COMPLETE.json").is_file(), "Metric analysis is incomplete")
    require((root / "state" / "REPORT_COMPLETE.json").is_file(), "Report generation is incomplete")

    manifest_files = verify_manifest(root, repo_root)
    embedding_path = root / "intermediate" / "candidate_embeddings.npz"
    metadata_path = embedding_path.with_suffix(".metadata.json")
    rejection_path = embedding_path.with_suffix(".rejections.csv")
    metadata = load_json(metadata_path)
    requested = config["reencoding"]
    execution = metadata["execution"]
    require(execution["backend"] == requested["backend"], "Wrong re-encoding backend")
    require(int(execution["batch_size"]) == 512, "Re-encoding batch size is not 512")
    require(int(execution["workers"]) == 48, "Re-encoding worker count is not 48")
    require(int(execution["node_budget"]) == int(requested["node_budget"]), "Wrong re-encoding node budget")
    require(int(execution["threads"]) == int(requested["threads"]), "Wrong re-encoding thread count")
    require(str(execution["device"]).startswith("cuda"), "Re-encoding was not run on CUDA")
    require(execution["invalid_policy"] == "error", "Re-encoding did not fail on invalid inputs")
    require(int(metadata["rows"]["total"]) == 2_108_115, "Wrong re-encoding input row count")
    require(int(metadata["rows"]["accepted"]) == 2_108_115, "Wrong re-encoding accepted row count")
    require(int(metadata["rows"]["rejected"]) == 0, "Re-encoding has rejected rows")
    require(int(metadata["embedding"]["dimensions"]) == 384, "Wrong embedding dimensionality")
    require(metadata["embedding"]["space"] == "released_hybrid_w3", "Wrong embedding space")
    require(metadata["outputs"][embedding_path.name]["sha256"] == sha256_file(embedding_path), "Embedding bundle hash mismatch")
    require(rejection_path.read_text(encoding="utf-8").count("\n") == 1, "Re-encoding rejection table is not header-only")

    with np.load(embedding_path, allow_pickle=False) as payload:
        require(int(np.asarray(payload["embedding_dimensions"]).item()) == 384, "Embedding bundle self-description is wrong")
        require(str(np.asarray(payload["embedding_space"]).item()) == "released_hybrid_w3", "Embedding bundle space is wrong")
        require(str(np.asarray(payload["input_sha256"]).item()) == sha256_file(root / "inputs" / "encoder_input.csv"), "Embedding bundle input hash mismatch")

    latent_path = root / "outputs" / "tables" / "candidate_latent_metrics.parquet"
    per_seed_path = root / "outputs" / "tables" / "per_seed_budget_metrics.parquet"
    summary_path = root / "outputs" / "tables" / "summary_metrics_by_budget.csv"
    ci_path = root / "outputs" / "tables" / "bootstrap_cis.csv"
    connection = duckdb.connect(database=":memory:")
    latent_check = connection.execute(
        """
        SELECT count(*) AS rows,
               count(DISTINCT query_position) AS seeds,
               sum(CASE WHEN NOT isfinite(latent_l2_to_seed_condition)
                              OR NOT isfinite(latent_relative_l2_to_seed_condition)
                              OR NOT isfinite(latent_cosine_to_seed_condition)
                        THEN 1 ELSE 0 END) AS nonfinite,
               min(candidate_structure_index) AS min_index,
               max(candidate_structure_index) AS max_index
        FROM read_parquet(?)
        """,
        [str(latent_path)],
    ).fetchone()
    connection.close()
    require(int(latent_check[0]) == 2_116_072, "Wrong candidate latent row count")
    require(int(latent_check[1]) == 10_000, "Wrong candidate latent seed count")
    require(int(latent_check[2]) == 0, "Candidate latent metrics contain non-finite values")
    require(int(latent_check[3]) >= 0 and int(latent_check[4]) < 2_108_115, "Candidate structure indices are out of bounds")

    required_per_seed = [
        "query_position",
        "budget",
        "candidate_set_nonempty",
        "greedy_exact_seed_identity",
        "generator_order_exact_seed_identity",
        "oracle_exact_seed_recall",
        "reranked_available",
        "reranked_exact_seed_identity",
        "reranked_identity_gain_over_greedy",
        "reranked_seed_scaffold_recovery",
        "reranked_morgan_similarity_to_seed",
        "reranked_latent_l2_to_seed_condition",
        "reranked_latent_relative_l2_to_seed_condition",
        "reranked_latent_cosine_to_seed_condition",
    ]
    per_seed = pd.read_parquet(per_seed_path, columns=required_per_seed)
    require(len(per_seed) == 50_000, "Per-seed budget table does not contain 50,000 rows")
    require(sorted(per_seed["budget"].unique().tolist()) == config["budgets"], "Per-seed table has wrong budgets")
    require(not per_seed.duplicated(["budget", "query_position"]).any(), "Per-seed table has duplicate budget/seed rows")
    require((per_seed.groupby("budget")["query_position"].nunique() == 10_000).all(), "A budget lacks seeds")
    require((per_seed["reranked_exact_seed_identity"] <= per_seed["oracle_exact_seed_recall"]).all(), "Reranked identity exceeds oracle")
    require(np.array_equal(per_seed["candidate_set_nonempty"].to_numpy(), per_seed["reranked_available"].to_numpy()), "Reranked availability differs from candidate availability")
    require(np.allclose(per_seed["reranked_identity_gain_over_greedy"], per_seed["reranked_exact_seed_identity"] - per_seed["greedy_exact_seed_identity"]), "Stored gain over greedy is inconsistent")
    selected = per_seed["reranked_available"] == 1.0
    latent_columns = [
        "reranked_latent_l2_to_seed_condition",
        "reranked_latent_relative_l2_to_seed_condition",
        "reranked_latent_cosine_to_seed_condition",
    ]
    require(np.isfinite(per_seed.loc[selected, latent_columns].to_numpy(dtype=float)).all(), "Selected top-1 latent metrics contain non-finite values")
    require(per_seed.loc[~selected, latent_columns].isna().all().all(), "Unavailable top-1 rows unexpectedly have latent metrics")

    ci = pd.read_csv(ci_path)
    require(len(ci) == len(config["budgets"]) * len(REQUIRED_METRICS), "Bootstrap table has wrong row count")
    require(set(ci["metric"]) == REQUIRED_METRICS, "Bootstrap table has missing or extra metrics")
    require(sorted(ci["budget"].unique().tolist()) == config["budgets"], "Bootstrap table has wrong budgets")
    require((ci["bootstrap_resamples"] == 2_000).all(), "Bootstrap resample count is not 2,000")
    require((ci["confidence_level"] == 0.95).all(), "Bootstrap confidence level is not 0.95")
    require(np.isfinite(ci[["estimate", "ci_lower", "ci_upper"]].to_numpy(dtype=float)).all(), "Bootstrap outputs contain non-finite values")
    summary = pd.read_csv(summary_path)
    require(summary["budget"].tolist() == config["budgets"], "Summary table has wrong budget order")

    for stem in PLOT_STEMS:
        source = root / "outputs" / "plot-data" / f"{stem}.csv"
        png = root / "outputs" / "figures" / f"{stem}.png"
        svg = root / "outputs" / "figures" / f"{stem}.svg"
        require(source.is_file() and source.stat().st_size > 100, f"Missing plot source data: {stem}")
        require(png.is_file() and png.stat().st_size > 5_000, f"Missing or empty PNG: {stem}")
        require(svg.is_file() and svg.stat().st_size > 5_000, f"Missing or empty SVG: {stem}")
    require((root / "RESULTS.md").is_file(), "Missing RESULTS.md")
    require((root / "outputs" / "analysis_summary.json").is_file(), "Missing machine-readable summary")
    ledger_files = verify_ledger(root)

    headline = ci.loc[
        ci["metric"].isin(
            [
                "greedy_exact_seed_identity_at_1",
                "generator_order_exact_seed_identity_at_1",
                "exact_seed_oracle_recall_at_budget",
                "reranked_exact_seed_identity_at_1",
                "rerank_selection_efficiency_given_oracle_presence",
                "reranked_identity_gain_over_greedy",
            ]
        ),
        ["budget", "metric", "estimate", "ci_lower", "ci_upper"],
    ]
    verification = {
        "schema_version": 1,
        "status": "passed",
        "study_id": config["study_id"],
        "verified_at": utc_now(),
        "writes_confined_by_runtime_mount": str(root),
        "immutable_manifest_files_verified": manifest_files,
        "output_ledger_files_verified": ledger_files,
        "seed_rows_per_budget": 10_000,
        "candidate_rows": 2_116_072,
        "reencoded_unique_molecules": 2_108_115,
        "embedding_space": "released_hybrid_w3",
        "reencoding_batch_size": 512,
        "reencoding_workers": 48,
        "budgets": config["budgets"],
        "bootstrap_resamples": 2_000,
        "figures_png": len(PLOT_STEMS),
        "figures_svg": len(PLOT_STEMS),
        "plot_source_csvs": len(PLOT_STEMS),
        "headline_metrics": json.loads(headline.to_json(orient="records")),
        "no_candidate_generation": True,
        "no_target_information_used_for_ranking": True,
        "no_property_or_control_analysis": True,
    }
    verification_path = root / "outputs" / "verification.json"
    atomic_write_json(verification_path, verification, root)
    state = dict(verification)
    state["verification_sha256"] = sha256_file(verification_path)
    atomic_write_json(verified_state, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
