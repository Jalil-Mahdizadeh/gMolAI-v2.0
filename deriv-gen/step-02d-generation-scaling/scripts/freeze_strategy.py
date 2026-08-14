#!/usr/bin/env python3
"""Select and seal one candidate-generation strategy using development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    STEP_ROOT,
    atomic_write_csv,
    atomic_write_json,
    load_json,
    protocol,
    sha256_file,
    stable_digest,
    utc_now,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    root = args.step_root.resolve()
    state_path = root / "state" / "STRATEGY_FROZEN.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    if not (root / "state" / "DEVELOPMENT_ANALYSIS_COMPLETE.json").is_file():
        raise RuntimeError("Development analysis must finish before selection")
    forbidden = list((root / "outputs" / "raw" / "final").glob("*.parquet"))
    forbidden.extend(root.glob("outputs/tables/final_*"))
    forbidden.extend(root.glob("state/FINAL_*"))
    if forbidden:
        raise RuntimeError(f"Final results existed before development selection: {forbidden[:3]}")

    cfg = protocol(root)
    selection_cfg = cfg["development_selection"]
    summary_path = root / "outputs" / "tables" / "development_budget_summary.csv"
    seed_path = root / "outputs" / "tables" / "development_seed_budget_metrics.parquet"
    summary = pd.read_csv(summary_path)
    seeds = pd.read_parquet(seed_path)
    maximum_budget = max(int(value) for value in cfg["budgets"])
    comparison = summary.loc[summary["budget"] == maximum_budget].copy()
    order_lookup = {
        value["name"]: int(value["registered_order"])
        for value in cfg["generation"]["development_strategies"]
    }
    comparison["registered_order"] = comparison["strategy"].map(order_lookup).astype(int)
    eligibility = selection_cfg["eligibility"]
    comparison["passes_policy_acceptance"] = (
        comparison["raw_policy_acceptance_fraction"] >= float(eligibility["minimum_raw_policy_acceptance"])
    )
    comparison["passes_unique_yield"] = (
        comparison["mean_unique_accepted_identities"] >= float(eligibility["minimum_mean_unique_policy_accepted"])
    )
    comparison["passes_locality"] = (
        comparison["median_seed_candidate_morgan_nonseed"] >= float(eligibility["minimum_median_nonseed_morgan"])
    )
    comparison["passes_useful_coverage"] = (
        comparison["seed_fraction_with_1_novel_useful_local"] >= float(eligibility["minimum_seed_coverage_with_one_novel_useful_local"])
    )
    comparison["eligible"] = comparison[
        [
            "passes_policy_acceptance",
            "passes_unique_yield",
            "passes_locality",
            "passes_useful_coverage",
        ]
    ].all(axis=1)
    eligible = comparison.loc[comparison["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No preregistered generation strategy passed development eligibility")
    best_primary = float(eligible["mean_novel_useful_local"].max())
    margin = float(selection_cfg["equivalence_margin_fraction_of_best_primary"])
    threshold = best_primary * (1.0 - margin)
    comparison["within_primary_equivalence"] = comparison["eligible"] & (
        comparison["mean_novel_useful_local"] >= threshold
    )
    equivalent = comparison.loc[comparison["within_primary_equivalence"]].copy()
    equivalent = equivalent.sort_values(
        [
            "mean_novel_mmp",
            "mean_novel_same_scaffold_non_mmp",
            "mean_novel_genuine_nonseed",
            "raw_policy_acceptance_fraction",
            "registered_order",
        ],
        ascending=[False, False, False, False, True],
    )
    selected_name = str(equivalent.iloc[0]["strategy"])
    selected_strategy = next(
        dict(value)
        for value in cfg["generation"]["development_strategies"]
        if value["name"] == selected_name
    )
    comparison["selected"] = comparison["strategy"] == selected_name

    maximum_seed = seeds.loc[seeds["budget"] == maximum_budget, ["strategy", "query_position", "novel_useful_local_count"]]
    pivot = maximum_seed.pivot(
        index="query_position", columns="strategy", values="novel_useful_local_count"
    ).sort_index()
    if pivot.isna().any().any():
        raise RuntimeError("Development strategies are not paired on identical seeds")
    rng = np.random.default_rng(
        int(stable_digest(cfg["seed"], "step2d-strategy-bootstrap")[:16], 16)
    )
    bootstrap_rows = []
    selected_values = pivot[selected_name].to_numpy(dtype=np.float64)
    resamples = int(selection_cfg["paired_bootstrap_resamples"])
    for other in pivot.columns:
        difference = selected_values - pivot[other].to_numpy(dtype=np.float64)
        boot = np.empty(resamples, dtype=np.float64)
        for index in range(resamples):
            positions = rng.integers(0, len(difference), size=len(difference))
            boot[index] = difference[positions].mean()
        low, high = np.quantile(boot, [0.025, 0.975])
        bootstrap_rows.append(
            {
                "selected_strategy": selected_name,
                "comparator_strategy": other,
                "paired_mean_difference_novel_useful_local": float(difference.mean()),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "bootstrap_probability_difference_positive": float(np.mean(boot > 0)),
                "resamples": resamples,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison_path = root / "outputs" / "tables" / "development_strategy_selection.csv"
    bootstrap_path = root / "outputs" / "tables" / "development_strategy_bootstrap.csv"
    atomic_write_csv(comparison_path, comparison.sort_values("registered_order"), root)
    atomic_write_csv(bootstrap_path, bootstrap, root)
    panel_metadata = load_json(root / "prepared" / "panel_metadata.json")
    state = {
        "schema_version": 1,
        "status": "frozen_before_final_generation",
        "frozen_at": utc_now(),
        "selected_strategy": selected_strategy,
        "selection_primary_metric": selection_cfg["primary_metric"],
        "selected_primary_value": float(
            comparison.loc[comparison["strategy"] == selected_name, "mean_novel_useful_local"].iloc[0]
        ),
        "best_primary_value": best_primary,
        "equivalence_threshold": threshold,
        "equivalence_margin_fraction": margin,
        "equivalent_strategies": equivalent["strategy"].astype(str).tolist(),
        "global_seed": int(cfg["seed"]),
        "sampling_seed_definition": cfg["generation"]["sampling_seed_definition"],
        "proposal_budgets": cfg["budgets"],
        "development_panel_sha256": panel_metadata["development"]["panel_sha256"],
        "final_panel_sha256": panel_metadata["final"]["panel_sha256"],
        "development_summary_sha256": sha256_file(summary_path),
        "selection_table_sha256": sha256_file(comparison_path),
        "bootstrap_table_sha256": sha256_file(bootstrap_path),
        "final_data_visible_during_selection": False,
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
