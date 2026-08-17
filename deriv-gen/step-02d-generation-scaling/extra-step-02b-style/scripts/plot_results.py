#!/usr/bin/env python3
"""Create source-backed figures and a concise report for the frozen analysis."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_inside,
    load_config,
    load_json,
    require_analysis_root,
    sha256_file,
)


LABELS = {
    "raw_token_decode_fraction": "Token decoded",
    "raw_rdkit_valid_fraction": "RDKit valid",
    "raw_policy_acceptance_fraction": "Policy accepted",
    "unique_identity_yield_per_raw_slot": "Unique identity yield",
    "mean_unique_accepted_identities": "Mean unique identities",
    "candidate_availability_rate": "Candidate available",
    "greedy_exact_seed_identity_at_1": "Greedy",
    "generator_order_exact_seed_identity_at_1": "Generator order",
    "exact_seed_oracle_recall_at_budget": "Exact-seed oracle",
    "reranked_exact_seed_identity_at_1": "Latent reranked",
    "greedy_seed_scaffold_recovery": "Greedy",
    "generator_order_seed_scaffold_recovery": "Generator order",
    "reranked_seed_scaffold_recovery": "Latent reranked",
    "greedy_mean_morgan_to_seed": "Greedy",
    "generator_order_mean_morgan_to_seed": "Generator order",
    "reranked_mean_morgan_to_seed": "Latent reranked",
    "reranked_mean_latent_l2": "L2",
    "reranked_mean_latent_relative_l2": "Relative L2",
    "reranked_mean_latent_cosine": "Cosine",
    "rerank_selection_efficiency_given_oracle_presence": "Selection efficiency",
    "reranked_identity_gain_over_greedy": "Gain over greedy",
}

COLORS = {
    "Token decoded": "#4C78A8",
    "RDKit valid": "#59A14F",
    "Policy accepted": "#E15759",
    "Unique identity yield": "#B279A2",
    "Candidate available": "#F28E2B",
    "Greedy": "#9C755F",
    "Generator order": "#4C78A8",
    "Exact-seed oracle": "#F28E2B",
    "Latent reranked": "#59A14F",
    "L2": "#4C78A8",
    "Relative L2": "#E15759",
    "Cosine": "#59A14F",
    "Selection efficiency": "#59A14F",
    "Gain over greedy": "#B279A2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metric_rows(ci: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    result = ci.loc[ci["metric"].isin(metrics)].copy()
    order = {metric: index for index, metric in enumerate(metrics)}
    result["metric_order"] = result["metric"].map(order).astype(int)
    result["display_label"] = result["metric"].map(LABELS)
    return result.sort_values(["metric_order", "budget"], ignore_index=True)


def plot_series(
    axis: plt.Axes,
    source: pd.DataFrame,
    metrics: list[str],
    *,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    for metric in metrics:
        rows = source.loc[source["metric"] == metric].sort_values("budget")
        label = LABELS[metric]
        values = rows["estimate"].to_numpy(dtype=float)
        low = rows["ci_lower"].to_numpy(dtype=float)
        high = rows["ci_upper"].to_numpy(dtype=float)
        axis.errorbar(
            rows["budget"],
            values,
            yerr=np.vstack([values - low, high - values]),
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            capsize=2.5,
            label=label,
            color=COLORS.get(label),
        )
    axis.set_xlabel("Raw proposal budget")
    axis.set_ylabel(ylabel)
    axis.set_xticks(sorted(source["budget"].unique()))
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(True, alpha=0.22, linewidth=0.7)


def save_figure(figure: plt.Figure, stem: Path, root: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".png", ".svg"):
        target = ensure_inside(stem.with_suffix(suffix), root)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.stem}.tmp{suffix}"
        try:
            figure.savefig(
                temporary,
                format=suffix.removeprefix("."),
                dpi=220 if suffix == ".png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append(target)
    plt.close(figure)
    return outputs


def create_figures(ci: pd.DataFrame, root: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure_outputs: list[Path] = []
    plot_data = root / "outputs" / "plot-data"
    figures = root / "outputs" / "figures"

    quality_metrics = [
        "raw_token_decode_fraction",
        "raw_rdkit_valid_fraction",
        "raw_policy_acceptance_fraction",
        "unique_identity_yield_per_raw_slot",
        "candidate_availability_rate",
        "mean_unique_accepted_identities",
    ]
    source = metric_rows(ci, quality_metrics)
    atomic_write_csv(plot_data / "candidate_quality_and_availability.csv", source, root)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    plot_series(axes[0], source, quality_metrics[:-1], ylabel="Fraction", ylim=(0.0, 1.02))
    axes[0].set_title("Proposal quality and candidate availability")
    axes[0].legend(loc="best")
    plot_series(axes[1], source, [quality_metrics[-1]], ylabel="Unique accepted identities per seed")
    axes[1].set_title("Candidate-set size")
    figure.suptitle("Step 2d candidate quality across nested raw budgets")
    figure.tight_layout()
    figure_outputs.extend(save_figure(figure, figures / "candidate_quality_and_availability", root))

    identity_metrics = [
        "greedy_exact_seed_identity_at_1",
        "generator_order_exact_seed_identity_at_1",
        "exact_seed_oracle_recall_at_budget",
        "reranked_exact_seed_identity_at_1",
    ]
    source = metric_rows(ci, identity_metrics)
    atomic_write_csv(plot_data / "exact_identity_recovery.csv", source, root)
    figure, axis = plt.subplots(figsize=(7.1, 4.4))
    plot_series(axis, source, identity_metrics, ylabel="Exact seed identity recovery", ylim=(0.0, 1.02))
    axis.set_title("Greedy, generator-order, oracle, and latent-reranked recovery")
    axis.legend(loc="best")
    figure.tight_layout()
    figure_outputs.extend(save_figure(figure, figures / "exact_identity_recovery", root))

    scaffold_metrics = [
        "greedy_seed_scaffold_recovery",
        "generator_order_seed_scaffold_recovery",
        "reranked_seed_scaffold_recovery",
    ]
    morgan_metrics = [
        "greedy_mean_morgan_to_seed",
        "generator_order_mean_morgan_to_seed",
        "reranked_mean_morgan_to_seed",
    ]
    source = metric_rows(ci, scaffold_metrics + morgan_metrics)
    atomic_write_csv(plot_data / "top1_structural_fidelity.csv", source, root)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    plot_series(axes[0], source, scaffold_metrics, ylabel="Scaffold recovery", ylim=(0.0, 1.02))
    axes[0].set_title("Top-1 scaffold recovery")
    axes[0].legend(loc="best")
    plot_series(axes[1], source, morgan_metrics, ylabel="Mean Morgan similarity", ylim=(0.0, 1.02))
    axes[1].set_title("Top-1 Morgan similarity to seed")
    axes[1].legend(loc="best")
    figure.suptitle("Unconditional top-1 structural fidelity")
    figure.tight_layout()
    figure_outputs.extend(save_figure(figure, figures / "top1_structural_fidelity", root))

    latent_metrics = [
        "reranked_mean_latent_l2",
        "reranked_mean_latent_relative_l2",
        "reranked_mean_latent_cosine",
    ]
    source = metric_rows(ci, latent_metrics)
    atomic_write_csv(plot_data / "reranked_latent_metrics.csv", source, root)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for axis, metric, ylabel in zip(
        axes,
        latent_metrics,
        ["Mean L2", "Mean relative L2", "Mean cosine"],
    ):
        plot_series(axis, source, [metric], ylabel=ylabel)
        axis.set_title(LABELS[metric])
    figure.suptitle("Latent consistency of selected reranked top-1 candidates")
    figure.tight_layout()
    figure_outputs.extend(save_figure(figure, figures / "reranked_latent_metrics", root))

    effect_metrics = [
        "rerank_selection_efficiency_given_oracle_presence",
        "reranked_identity_gain_over_greedy",
    ]
    source = metric_rows(ci, effect_metrics)
    atomic_write_csv(plot_data / "reranking_effect.csv", source, root)
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    plot_series(axes[0], source, [effect_metrics[0]], ylabel="Selection efficiency", ylim=(0.0, 1.02))
    axes[0].set_title("Exact selection given oracle presence")
    plot_series(axes[1], source, [effect_metrics[1]], ylabel="Identity fraction-point gain")
    axes[1].axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")
    axes[1].set_title("Paired gain over retained greedy")
    figure.suptitle("Effect of target-blind latent reranking")
    figure.tight_layout()
    figure_outputs.extend(save_figure(figure, figures / "reranking_effect", root))
    return figure_outputs


def estimate(ci_index: pd.DataFrame, budget: int, metric: str) -> float:
    return float(ci_index.loc[(int(budget), metric), "estimate"])


def make_report(ci: pd.DataFrame, config: dict, root: Path) -> str:
    indexed = ci.set_index(["budget", "metric"])
    lines = [
        "# Step 2b-style search and latent-reranking analysis of Step 2d",
        "",
        "This report evaluates the frozen 10,000-seed Step 2d final candidate library at literal raw proposal prefixes. It adds no generation, training, latent perturbation, property analysis, controls, or decision gate.",
        "",
        "All globally unique accepted molecules were re-encoded once in released_hybrid_w3 space with batch size 512 and 48 workers. Reranking is target-blind: lower relative L2, then higher cosine, lower raw rank, and lexical canonical SMILES. Intervals are paired 2,000-resample seed-bootstrap 95% percentile intervals.",
        "",
        "## Main estimates",
        "",
        "| Raw budget | RDKit valid | Policy accepted | Mean unique | Candidate available | Greedy exact@1 | Generator exact@1 | Oracle recall | Reranked exact@1 | Selection efficiency | Gain over greedy | Reranked scaffold | Reranked Morgan | Reranked rel-L2 | Reranked cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in config["budgets"]:
        values = {
            "valid": estimate(indexed, budget, "raw_rdkit_valid_fraction"),
            "accepted": estimate(indexed, budget, "raw_policy_acceptance_fraction"),
            "unique": estimate(indexed, budget, "mean_unique_accepted_identities"),
            "available": estimate(indexed, budget, "candidate_availability_rate"),
            "greedy": estimate(indexed, budget, "greedy_exact_seed_identity_at_1"),
            "generator": estimate(indexed, budget, "generator_order_exact_seed_identity_at_1"),
            "oracle": estimate(indexed, budget, "exact_seed_oracle_recall_at_budget"),
            "reranked": estimate(indexed, budget, "reranked_exact_seed_identity_at_1"),
            "efficiency": estimate(indexed, budget, "rerank_selection_efficiency_given_oracle_presence"),
            "gain": estimate(indexed, budget, "reranked_identity_gain_over_greedy"),
            "scaffold": estimate(indexed, budget, "reranked_seed_scaffold_recovery"),
            "morgan": estimate(indexed, budget, "reranked_mean_morgan_to_seed"),
            "relative": estimate(indexed, budget, "reranked_mean_latent_relative_l2"),
            "cosine": estimate(indexed, budget, "reranked_mean_latent_cosine"),
        }
        lines.append(
            f"| {budget:,} | {values['valid']:.2%} | {values['accepted']:.2%} | {values['unique']:.1f} | {values['available']:.2%} | {values['greedy']:.2%} | {values['generator']:.2%} | {values['oracle']:.2%} | {values['reranked']:.2%} | {values['efficiency']:.2%} | {values['gain']:+.2%} | {values['scaffold']:.2%} | {values['morgan']:.4f} | {values['relative']:.4f} | {values['cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The complete estimates and confidence limits are in outputs/tables/bootstrap_cis.csv; the per-seed numerator data are retained in outputs/tables/per_seed_budget_metrics.parquet.",
            "",
            "## Figures",
            "",
            "![Candidate quality and availability](outputs/figures/candidate_quality_and_availability.png)",
            "",
            "![Exact identity recovery](outputs/figures/exact_identity_recovery.png)",
            "",
            "![Top-1 structural fidelity](outputs/figures/top1_structural_fidelity.png)",
            "",
            "![Reranked latent metrics](outputs/figures/reranked_latent_metrics.png)",
            "",
            "![Reranking effect](outputs/figures/reranking_effect.png)",
            "",
            "Every figure is also exported as SVG, and each has an exact CSV source table under outputs/plot-data.",
            "",
        ]
    )
    return "\n".join(lines)


def write_hash_ledger(root: Path) -> Path:
    ledger_path = root / "outputs" / "SHA256SUMS"
    files = [
        path
        for path in (root / "outputs").rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "verification.json"}
    ]
    files.append(root / "RESULTS.md")
    records = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in sorted(files)]
    atomic_write_text(ledger_path, "\n".join(records) + "\n", root)
    return ledger_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = require_analysis_root(args.analysis_root)
    config = load_config(root)
    analysis_state = root / "state" / "ANALYSIS_COMPLETE.json"
    if not analysis_state.is_file():
        raise RuntimeError("Metric analysis is not complete")
    report_state = root / "state" / "REPORT_COMPLETE.json"
    if report_state.exists():
        print(json.dumps(load_json(report_state), sort_keys=True))
        return
    started = time.monotonic()
    ci_path = root / "outputs" / "tables" / "bootstrap_cis.csv"
    ci = pd.read_csv(ci_path)
    expected_metrics = set(LABELS)
    if not expected_metrics.issubset(set(ci["metric"])):
        raise RuntimeError(f"Bootstrap table lacks plotted metrics: {sorted(expected_metrics.difference(set(ci['metric'])))}")
    figure_outputs = create_figures(ci, root)
    report_path = root / "RESULTS.md"
    atomic_write_text(report_path, make_report(ci, config, root), root)
    summary_path = root / "outputs" / "analysis_summary.json"
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at": utc_now(),
        "scope": "Step 2b-style direct search, structural recovery, and frozen-latent reranking only",
        "budgets": config["budgets"],
        "seed_rows_per_budget": 10_000,
        "bootstrap": config["bootstrap"],
        "reencoding": config["reencoding"],
        "confidence_intervals": json.loads(ci.to_json(orient="records")),
    }
    atomic_write_json(summary_path, summary, root)
    ledger_path = write_hash_ledger(root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "completed_at": utc_now(),
        "figures": [str(path.relative_to(root)) for path in figure_outputs],
        "plot_source_tables": sorted(str(path.relative_to(root)) for path in (root / "outputs" / "plot-data").glob("*.csv")),
        "results_sha256": sha256_file(report_path),
        "analysis_summary_sha256": sha256_file(summary_path),
        "ledger_sha256": sha256_file(ledger_path),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(report_state, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
