#!/usr/bin/env python3
"""Create bounded Step-2d figures, tables, decision, and human-readable reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    STEP_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    protocol,
    sha256_file,
    utc_now,
)


def finite(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.nan


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    root = args.step_root.resolve()
    state_path = root / "state" / "REPORT_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    if not (root / "state" / "FINAL_ANALYSIS_COMPLETE.json").is_file():
        raise RuntimeError("Final analysis must finish before reporting")
    cfg = protocol(root)
    frozen = load_json(root / "state" / "STRATEGY_FROZEN.json")
    strategy = str(frozen["selected_strategy"]["name"])
    table_root = root / "outputs" / "tables"
    summary = pd.read_csv(table_root / "final_budget_summary.csv")
    summary = summary.loc[summary["strategy"] == strategy].sort_values("budget")
    seed_metrics = pd.read_parquet(table_root / "final_seed_budget_metrics.parquet")
    candidates = pd.read_parquet(table_root / "final_candidate_characterization.parquet")
    similarity = pd.read_csv(table_root / "final_similarity_by_category_budget.csv")
    incremental = pd.read_csv(table_root / "final_incremental_yield.csv")
    selection = pd.read_csv(table_root / "development_strategy_selection.csv")
    budgets = [int(value) for value in cfg["budgets"]]
    if summary["budget"].astype(int).tolist() != budgets:
        raise RuntimeError("Final summary budgets changed")
    maximum = summary.loc[summary["budget"] == max(budgets)].iloc[0]

    maximum_utility = float(maximum["mean_novel_useful_local"])
    target_utility = float(cfg["scaling_decision"]["recommended_budget_fraction_of_1000_utility"]) * maximum_utility
    eligible_budgets = summary.loc[summary["mean_novel_useful_local"] >= target_utility, "budget"]
    recommended_budget = int(eligible_budgets.min()) if len(eligible_budgets) else max(budgets)

    incremental = incremental.loc[incremental["strategy"] == strategy].sort_values("budget")
    rate_column = "mean_rate_novel_useful_local_count"
    first_rate = float(incremental.loc[incremental["budget"] == min(budgets), rate_column].iloc[0])
    diminish_ratio = float(cfg["scaling_decision"]["diminishing_return_fraction_of_first_50_rate"])
    saturation_ratio = float(cfg["scaling_decision"]["saturation_fraction_of_first_50_rate"])
    diminishing_candidates = incremental.loc[incremental[rate_column] <= first_rate * diminish_ratio, "budget"]
    diminishing_budget = int(diminishing_candidates.min()) if len(diminishing_candidates) else None
    saturation_budget = None
    for budget in budgets:
        later = incremental.loc[incremental["budget"] >= budget, rate_column]
        if len(later) and bool((later <= first_rate * saturation_ratio).all()):
            saturation_budget = int(budget)
            break

    gates_cfg = cfg["scaling_decision"]["large_library_gates"]
    gates = {
        "raw_policy_acceptance": bool(maximum["raw_policy_acceptance_fraction"] >= float(gates_cfg["minimum_raw_policy_acceptance"])),
        "median_unique_policy_accepted": bool(maximum["median_unique_accepted_identities"] >= float(gates_cfg["minimum_median_unique_policy_accepted"])),
        "novel_fraction_among_genuine_nonseed": bool(maximum["novel_fraction_among_genuine_nonseed"] >= float(gates_cfg["minimum_novel_fraction_among_genuine_nonseed"])),
        "median_novel_useful_local": bool(maximum["median_novel_useful_local"] >= float(gates_cfg["minimum_median_novel_useful_local"])),
        "seed_fraction_with_10_novel_useful_local": bool(maximum["seed_fraction_with_10_novel_useful_local"] >= float(gates_cfg["minimum_seed_fraction_with_10_novel_useful_local"])),
    }
    large_library_supported = all(gates.values())

    category_rows = []
    for budget in budgets:
        current = candidates.loc[candidates["first_proposal_rank"] <= budget]
        for category, group in current.groupby("chemical_category", sort=False):
            genuine = group.loc[group["is_genuine_nonseed"].astype(bool)]
            category_rows.append(
                {
                    "strategy": strategy,
                    "budget": budget,
                    "chemical_category": category,
                    "unique_candidate_rows": len(group),
                    "genuine_nonseed_rows": len(genuine),
                    "novel_genuine_nonseed_rows": int(genuine["is_novel_to_decoder_training"].sum()),
                    "novel_fraction_among_genuine_nonseed": (
                        float(genuine["is_novel_to_decoder_training"].mean()) if len(genuine) else math.nan
                    ),
                    "seed_coverage": float(group["query_position"].nunique() / seed_metrics["query_position"].nunique()),
                }
            )
    category_table = pd.DataFrame(category_rows)
    category_path = table_root / "final_category_novelty_by_budget.csv"
    atomic_write_csv(category_path, category_table, root)
    mmp = candidates.loc[
        (candidates["first_proposal_rank"] <= max(budgets))
        & candidates["is_one_cut_mmp"].astype(bool)
    ]
    if len(mmp):
        transformations = (
            mmp.groupby(
                ["seed_to_candidate_transform", "mmp_edit_class"],
                dropna=False,
                as_index=False,
            )
            .agg(
                candidate_rows=("candidate_hash", "size"),
                distinct_seeds=("query_position", "nunique"),
                distinct_candidate_identities=("candidate_hash", "nunique"),
                novel_candidate_rows=("is_novel_to_decoder_training", "sum"),
            )
            .sort_values(["candidate_rows", "distinct_seeds"], ascending=False)
        )
    else:
        transformations = pd.DataFrame(
            columns=[
                "seed_to_candidate_transform",
                "mmp_edit_class",
                "candidate_rows",
                "distinct_seeds",
                "distinct_candidate_identities",
                "novel_candidate_rows",
            ]
        )
    transformation_path = table_root / "final_mmp_transformations.csv"
    atomic_write_csv(transformation_path, transformations, root)

    figure_root = root / "outputs" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    for column, label in (
        ("mean_unique_accepted_identities", "unique accepted identities"),
        ("mean_novel_genuine_nonseed", "novel non-seed identities"),
        ("mean_novel_useful_local", "novel useful-local analogues"),
        ("mean_mmp_derivatives", "one-cut MMPs"),
    ):
        plt.plot(summary["budget"], summary[column], marker="o", label=label)
    plt.axvline(recommended_budget, color="black", linestyle="--", linewidth=1, label=f"recommended {recommended_budget}")
    plt.xlabel("Raw proposal budget per seed")
    plt.ylabel("Mean unique molecules per seed")
    plt.title("Frozen decoder chemical yield scaling")
    plt.legend(fontsize=8)
    yield_figure = figure_root / "yield_scaling.png"
    save_figure(yield_figure)

    fig, first = plt.subplots(figsize=(7.2, 4.8))
    first.plot(summary["budget"], summary["raw_valid_fraction"], marker="o", label="valid", color="#2b8cbe")
    first.plot(summary["budget"], summary["raw_policy_acceptance_fraction"], marker="o", label="policy accepted", color="#41ab5d")
    first.plot(summary["budget"], summary["novel_fraction_among_genuine_nonseed"], marker="o", label="novel among non-seed", color="#756bb1")
    first.set_ylim(0, 1.02)
    first.set_xlabel("Raw proposal budget per seed")
    first.set_ylabel("Fraction")
    second = first.twinx()
    second.plot(summary["budget"], summary["median_seed_candidate_morgan_nonseed"], marker="s", linestyle="--", color="#e6550d", label="median seed similarity")
    second.plot(summary["budget"], summary["within_pairwise_morgan_weighted_mean"], marker="s", linestyle=":", color="#636363", label="within-set similarity")
    second.set_ylim(0, 1.02)
    second.set_ylabel("Morgan/Tanimoto")
    lines = first.get_lines() + second.get_lines()
    first.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    plt.title("Validity, novelty, locality, and diversity")
    quality_figure = figure_root / "quality_locality_diversity_scaling.png"
    save_figure(quality_figure)

    selected_dev = selection.loc[selection["selected"].astype(bool)].iloc[0]
    decision = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "selected_generation_strategy": frozen["selected_strategy"],
        "development_selection": {
            "primary_mean_novel_useful_local_at_1000": float(selected_dev["mean_novel_useful_local"]),
            "raw_policy_acceptance_at_1000": float(selected_dev["raw_policy_acceptance_fraction"]),
            "median_seed_candidate_morgan_at_1000": float(selected_dev["median_seed_candidate_morgan_nonseed"]),
            "final_data_visible": False,
        },
        "final_panel_rows": int(seed_metrics["query_position"].nunique()),
        "recommended_raw_proposal_budget": recommended_budget,
        "recommended_budget_rule": "smallest registered budget reaching at least 90% of budget-1000 mean novel useful-local yield",
        "diminishing_returns_begin_budget": diminishing_budget,
        "saturation_budget": saturation_budget,
        "budget_1000": {
            "valid_fraction": float(maximum["raw_valid_fraction"]),
            "policy_accepted_fraction": float(maximum["raw_policy_acceptance_fraction"]),
            "mean_unique_identities_per_seed": float(maximum["mean_unique_accepted_identities"]),
            "median_unique_identities_per_seed": float(maximum["median_unique_accepted_identities"]),
            "mean_genuine_nonseed_per_seed": float(maximum["mean_genuine_nonseed"]),
            "mean_mmp_per_seed": float(maximum["mean_mmp_derivatives"]),
            "mmp_fraction_among_genuine_nonseed": float(maximum["mmp_fraction_among_genuine_nonseed"]),
            "mean_same_scaffold_non_mmp_per_seed": float(maximum["mean_same_scaffold_non_mmp"]),
            "same_scaffold_fraction_among_eligible_nonseed": float(maximum["same_scaffold_fraction_among_scaffold_eligible_nonseed"]),
            "mean_distinct_scaffolds_per_seed": float(maximum["mean_distinct_scaffolds"]),
            "novel_fraction_among_genuine_nonseed": float(maximum["novel_fraction_among_genuine_nonseed"]),
            "mean_novel_useful_local_per_seed": float(maximum["mean_novel_useful_local"]),
            "median_seed_candidate_morgan_nonseed": float(maximum["median_seed_candidate_morgan_nonseed"]),
            "within_pairwise_morgan_weighted_mean": float(maximum["within_pairwise_morgan_weighted_mean"]),
        },
        "prospective_large_library_gates": gates,
        "large_useful_analogue_library_supported": large_library_supported,
        "bounded_answer": (
            "The frozen unperturbed gMolAI-conditioned decoder meets the preregistered large-library definition."
            if large_library_supported
            else "The frozen decoder generates measurable valid, unique, novel analogue yield, but it does not meet every preregistered large-library gate."
        ),
        "scope_limits": [
            "No claim of synthesizability, activity, or property improvement.",
            "No latent perturbation, MMP-direction editing, property optimization, or Step 3 was performed.",
            "Novelty is defined only against the 980,000 decoder-fit identities.",
        ],
    }
    decision_path = root / "outputs" / "decision.json"
    atomic_write_json(decision_path, decision, root)

    recommended = summary.loc[summary["budget"] == recommended_budget].iloc[0]
    results_text = f"""# Step 2d results: frozen decoder candidate scaling

The development-only comparison selected **{strategy}** before any final
generation. Final evaluation used {int(decision['final_panel_rows']):,} fresh validation seeds
and literal nested raw proposal budgets of 50, 100, 250, 500, and 1,000.

## Main result

At 1,000 raw proposals per seed, {percent(float(maximum['raw_valid_fraction']))} were
RDKit-valid and {percent(float(maximum['raw_policy_acceptance_fraction']))} passed the
unchanged gMolAI policy. The decoder yielded a mean of
{float(maximum['mean_unique_accepted_identities']):.2f} unique accepted identities,
{float(maximum['mean_genuine_nonseed']):.2f} genuine non-seed molecules,
{float(maximum['mean_mmp_derivatives']):.2f} exact Step-1b one-cut MMP derivatives, and
{float(maximum['mean_novel_useful_local']):.2f} novel useful-local analogues per seed.
Novelty among genuine non-seed molecules was
{percent(float(maximum['novel_fraction_among_genuine_nonseed']))} relative only to the
980,000 decoder-training identities.

Locality broadened with budget: the budget-1,000 median seed-candidate Morgan
similarity was {float(maximum['median_seed_candidate_morgan_nonseed']):.3f}; the
weighted sampled/exact within-set mean was
{float(maximum['within_pairwise_morgan_weighted_mean']):.3f}. Non-empty scaffold
retention among eligible non-seed candidates was
{percent(float(maximum['same_scaffold_fraction_among_scaffold_eligible_nonseed']))};
acyclic seeds are tabulated separately in the machine-readable outputs.

## Scaling decision

The preregistered 90%-utility rule recommends **{recommended_budget} raw proposals
per seed**, which yields {float(recommended['mean_novel_useful_local']):.2f} mean novel
useful-local analogues per seed versus {maximum_utility:.2f} at 1,000.
Diminishing returns begin at {diminishing_budget if diminishing_budget is not None else 'no registered budget'};
strict saturation begins at {saturation_budget if saturation_budget is not None else 'no registered budget'}.

Prospective large-library classification: **{'SUPPORTED' if large_library_supported else 'NOT SUPPORTED'}**
({sum(gates.values())}/{len(gates)} gates passed). This is a bounded chemical-yield
statement, not evidence of synthesis feasibility, bioactivity, or property gain.

## Reproducibility notes

- Every budget is a prefix of the same frozen 1,000-slot stream.
- Invalid outputs, policy failures, duplicate strings, alternative strings for one
  molecular identity, and seed identities remain in their true raw denominators.
- MMPs use the exact imported Step-1b one-cut implementation.
- Morgan fingerprints use radius 2, 2,048 bits, and no chirality flag.
- Candidate novelty is assessed only against the decoder-fit 980,000 molecules.
"""
    atomic_write_text(root / "RESULTS.md", results_text, root)
    decision_text = f"""# Step 2d decision

Selected strategy: **{strategy}**. Recommended nominal budget:
**{recommended_budget} raw proposals per seed**.

{decision['bounded_answer']}

At the maximum budget the key yields were {float(maximum['mean_unique_accepted_identities']):.2f}
unique accepted identities, {float(maximum['mean_mmp_derivatives']):.2f} one-cut MMPs,
and {float(maximum['mean_novel_useful_local']):.2f} novel useful-local analogues per
seed. The candidate population remains broader than a strictly local analogue space,
as shown by median seed similarity {float(maximum['median_seed_candidate_morgan_nonseed']):.3f}
and scaffold-retention fraction
{percent(float(maximum['same_scaffold_fraction_among_scaffold_eligible_nonseed']))}.

No latent perturbation, property optimization, decoder/encoder training, or Step 3
was performed.
"""
    atomic_write_text(root / "DECISION.md", decision_text, root)
    readme = """# Step 2d: frozen decoder generation scaling

This directory is a self-contained, preregistered no-training study of chemical
yield as raw candidate budgets scale from 50 to 1,000 per seed. See
`PROTOCOL.md` for frozen rules, `RESULTS.md` for findings, `DECISION.md` for the
bounded decision, and `outputs/tables/` for machine-readable results.

Run `scripts/run_study.sh` inside a four-GPU SLURM allocation, or submit
`scripts/submit_step2d.slurm`.
"""
    atomic_write_text(root / "README.md", readme, root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "selected_strategy": strategy,
        "recommended_budget": recommended_budget,
        "decision_sha256": sha256_file(decision_path),
        "results_sha256": sha256_file(root / "RESULTS.md"),
        "figure_sha256": {
            "yield_scaling": sha256_file(yield_figure),
            "quality_locality_diversity": sha256_file(quality_figure),
        },
        "table_sha256": {
            "category_novelty": sha256_file(category_path),
            "mmp_transformations": sha256_file(transformation_path),
        },
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
