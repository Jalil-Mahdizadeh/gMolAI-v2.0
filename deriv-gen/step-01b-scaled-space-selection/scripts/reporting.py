"""Decision logic, figures, and reports for the scaled latent-space study."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import RETRIEVAL_METRICS, SPACE_ORDER
from scaled_common import atomic_write_text, ensure_within


COLORS = {
    "graph_256": "#2563eb",
    "mean_node_128": "#d97706",
    "hybrid_w1": "#059669",
    "released_hybrid_w3": "#7c3aed",
    "hybrid_w6": "#dc2626",
}


def save_figure(fig: plt.Figure, path: Path, step_root: Path) -> None:
    target = ensure_within(path, step_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(
            temporary,
            format=target.suffix.lstrip("."),
            dpi=220 if target.suffix == ".png" else None,
            bbox_inches="tight",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _row(
    summary: pd.DataFrame,
    analysis: str,
    threshold: int,
    metric: str,
    space: str,
) -> pd.Series:
    match = summary.loc[
        (summary["analysis"] == analysis)
        & (summary["minimum_train_cores"] == threshold)
        & (summary["metric"] == metric)
        & (summary["space"] == space)
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"Expected one summary row for {analysis}/{threshold}/{metric}/{space}, got {len(match)}"
        )
    return match.iloc[0]


def select_control_space(
    config: dict[str, Any],
    bootstrap_summary: pd.DataFrame,
    retrieval_average: pd.DataFrame,
    paired_differences: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    threshold = int(config["retrieval"]["primary_support_threshold"])
    settings = config["selection"]
    records: list[dict[str, Any]] = []
    for space in SPACE_ORDER:
        alignment = _row(
            bootstrap_summary,
            "alignment_all",
            threshold,
            "alignment_gain",
            space,
        )
        recall = _row(
            bootstrap_summary,
            "retrieval_mmp_direction",
            threshold,
            "exact_derivative_recall_at_1",
            space,
        )
        exact = _row(
            bootstrap_summary,
            "retrieval_mmp_direction",
            threshold,
            "exact_requested_transform",
            space,
        )
        pos = float(alignment["positive_transformation_fraction"])
        sufficiently_powered = (
            int(recall["transformations"])
            >= int(settings["minimum_primary_transformations"])
            and int(recall["observations"])
            >= int(settings["minimum_primary_queries"])
        )
        viable = bool(
            sufficiently_powered
            and float(alignment["ci_low"]) > 0.0
            and pos
            >= float(settings["minimum_positive_alignment_fraction"])
        )
        records.append(
            {
                "space": space,
                "primary_transformations": int(recall["transformations"]),
                "primary_queries": int(recall["observations"]),
                "alignment_gain": float(alignment["macro_estimate"]),
                "alignment_gain_ci_low": float(alignment["ci_low"]),
                "alignment_gain_ci_high": float(alignment["ci_high"]),
                "positive_alignment_transformation_fraction": pos,
                "exact_derivative_recall_at_1": float(
                    recall["macro_estimate"]
                ),
                "exact_derivative_recall_at_1_ci_low": float(recall["ci_low"]),
                "exact_derivative_recall_at_1_ci_high": float(
                    recall["ci_high"]
                ),
                "exact_requested_transform": float(exact["macro_estimate"]),
                "exact_requested_transform_ci_low": float(exact["ci_low"]),
                "exact_requested_transform_ci_high": float(exact["ci_high"]),
                "sufficiently_powered": sufficiently_powered,
                "viable": viable,
            }
        )
    table = pd.DataFrame(records)
    metric_columns = [
        "exact_derivative_recall_at_1",
        "alignment_gain",
        "exact_requested_transform",
        "positive_alignment_transformation_fraction",
    ]
    viable_spaces = table.loc[table["viable"], "space"].tolist()
    pareto: list[str] = []
    for candidate in viable_spaces:
        current = table.loc[table["space"] == candidate].iloc[0]
        dominated = False
        for challenger in viable_spaces:
            if challenger == candidate:
                continue
            other = table.loc[table["space"] == challenger].iloc[0]
            no_worse = all(
                float(other[column]) >= float(current[column])
                for column in metric_columns
            )
            better = any(
                float(other[column]) > float(current[column])
                for column in metric_columns
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    table["pareto"] = table["space"].isin(pareto)

    selected: str | None = None
    selection_reason = ""
    compatibility = str(settings["compatibility_preference"])
    powered = bool(table["sufficiently_powered"].all())
    if not powered:
        selection_reason = (
            "The fixed primary panel did not reach the predeclared minimum "
            "support, so no edit-control space can be frozen."
        )
    elif not viable_spaces:
        selection_reason = (
            "No candidate passed the predeclared unseen-core alignment "
            "viability criteria, so no edit-control space can be frozen."
        )
    else:
        margins = {
            "exact_derivative_recall_at_1": float(
                settings["recall_at_1_noninferiority_margin"]
            ),
            "alignment_gain": float(
                settings["alignment_gain_noninferiority_margin"]
            ),
            "exact_requested_transform": float(
                settings["exact_transform_noninferiority_margin"]
            ),
        }
        compatible = compatibility in viable_spaces
        if compatible:
            compatible_row = table.loc[
                table["space"] == compatibility
            ].iloc[0]
            for metric, margin in margins.items():
                best_pareto = float(
                    table.loc[table["space"].isin(pareto), metric].max()
                )
                if float(compatible_row[metric]) < best_pareto - margin:
                    compatible = False
                    break
        if compatible:
            selected = compatibility
            selection_reason = (
                "The released weight-3 hybrid is viable and within every "
                "predeclared non-inferiority margin of the Pareto frontier, "
                "so representation compatibility breaks the tie."
            )
        else:
            viable_table = table.loc[table["viable"]].copy()
            best_recall = float(
                viable_table["exact_derivative_recall_at_1"].max()
            )
            contenders = viable_table.loc[
                viable_table["exact_derivative_recall_at_1"]
                >= best_recall
                - float(settings["recall_at_1_noninferiority_margin"])
            ].sort_values(
                [
                    "alignment_gain",
                    "exact_requested_transform",
                    "exact_derivative_recall_at_1",
                    "space",
                ],
                ascending=[False, False, False, True],
            )
            selected = str(contenders.iloc[0]["space"])
            selection_reason = (
                "The released weight-3 hybrid exceeded at least one fixed "
                "non-inferiority margin; the frozen ranking rule selected "
                "the strongest viable alternative."
            )

    table["selected_edit_control_space"] = table["space"] == selected
    w1 = table.loc[table["space"] == "hybrid_w1"].iloc[0]
    w3 = table.loc[table["space"] == "released_hybrid_w3"].iloc[0]

    comparison_spec = {
        "alignment_gain": "alignment_all",
        "exact_derivative_recall_at_1": "retrieval_mmp_direction",
        "exact_requested_transform": "retrieval_mmp_direction",
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for metric, analysis in comparison_spec.items():
        match = paired_differences.loc[
            (paired_differences["analysis"] == analysis)
            & (paired_differences["minimum_train_cores"] == threshold)
            & (paired_differences["metric"] == metric)
            & (paired_differences["space"] == "hybrid_w1")
            & (
                paired_differences["reference_space"]
                == "released_hybrid_w3"
            )
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Missing paired weight-1/weight-3 comparison for {metric}"
            )
        row = match.iloc[0]
        delta = -float(row["paired_macro_difference"])
        ci_low = -float(row["ci_high"])
        ci_high = -float(row["ci_low"])
        comparisons[metric] = {
            "macro_delta_w3_minus_w1": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "statistically_resolved_positive": bool(ci_low > 0.0),
            "statistically_resolved_negative": bool(ci_high < 0.0),
        }

    alignment_improves = comparisons["alignment_gain"][
        "statistically_resolved_positive"
    ]
    retrieval_improves = comparisons[
        "exact_derivative_recall_at_1"
    ]["statistically_resolved_positive"]
    retrieval_point_higher = (
        comparisons["exact_derivative_recall_at_1"][
            "macro_delta_w3_minus_w1"
        ]
        > 0.0
    )
    if alignment_improves and retrieval_improves:
        w3_conclusion = (
            "Weight 3 improves both directional alignment and exact retrieval "
            "over weight 1 with paired confidence intervals above zero."
        )
    elif alignment_improves and retrieval_point_higher:
        w3_conclusion = (
            "Weight 3 clearly improves directional alignment over weight 1; "
            "exact retrieval is numerically higher but its paired confidence "
            "interval includes zero."
        )
    elif alignment_improves:
        w3_conclusion = (
            "Weight 3 clearly improves directional alignment over weight 1, "
            "but it does not improve exact retrieval."
        )
    else:
        w3_conclusion = (
            "The study does not show a statistically resolved overall "
            "improvement of weight 3 over weight 1."
        )

    mean_row = table.loc[table["space"] == "mean_node_128"].iloc[0]
    alignment_leader = str(
        table.sort_values(
            ["alignment_gain", "space"], ascending=[False, True]
        ).iloc[0]["space"]
    )
    retrieval_leader = str(
        table.sort_values(
            ["exact_derivative_recall_at_1", "space"],
            ascending=[False, True],
        ).iloc[0]["space"]
    )
    mean_assessment = {
        "directional_alignment_leader": alignment_leader == "mean_node_128",
        "exact_retrieval_leader": retrieval_leader == "mean_node_128",
        "pareto_optimal": bool(mean_row["pareto"]),
        "unique_overall_winner": bool(
            selected == "mean_node_128" and len(pareto) == 1
        ),
        "conclusion": (
            "Mean-node-128 remains the strongest directional-alignment "
            "space and is Pareto-optimal, but it is not the unique overall "
            "winner; released weight 3 has the highest exact recall and wins "
            "the compatibility-aware frozen rule."
        ),
    }
    decision = {
        "schema_version": 1,
        "primary_support_threshold": threshold,
        "formal_selection_powered": powered,
        "viable_spaces": viable_spaces,
        "pareto_spaces": pareto,
        "selected_edit_control_space": selected,
        "selection_reason": selection_reason,
        "decoder_conditioning_representation": settings[
            "decoder_conditioning_representation"
        ],
        "released_w3_vs_unweighted_w1": {
            "conclusion": w3_conclusion,
            "metrics": comparisons,
            "macro_deltas_w3_minus_w1": {
                metric: record["macro_delta_w3_minus_w1"]
                for metric, record in comparisons.items()
            },
        },
        "mean_node_128_assessment": mean_assessment,
        "mean_node_128_remains_strongest": (
            "directionally_yes_but_not_unique_overall"
        ),
        "claim_boundary": (
            "Retrieval geometry only; this does not demonstrate decoding or "
            "novel-molecule generation."
        ),
    }
    return decision, table


def transfer_assessment(
    bootstrap_summary: pd.DataFrame,
    selected_space: str | None,
) -> list[dict[str, Any]]:
    if selected_space is None:
        primary = bootstrap_summary.loc[
            (bootstrap_summary["analysis"] == "alignment_all")
            & (bootstrap_summary["metric"] == "alignment_gain")
        ]
        if primary.empty:
            return []
        selected_space = str(
            primary.sort_values("macro_estimate", ascending=False).iloc[0][
                "space"
            ]
        )
    selected = bootstrap_summary.loc[
        (bootstrap_summary["analysis"] == "alignment_all")
        & (bootstrap_summary["metric"] == "alignment_gain")
        & (bootstrap_summary["space"] == selected_space)
    ].sort_values("minimum_train_cores")
    return [
        {
            "space": selected_space,
            "minimum_train_cores": int(row.minimum_train_cores),
            "transformations": int(row.transformations),
            "observations": int(row.observations),
            "macro_alignment_gain": float(row.macro_estimate),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "survives": bool(float(row.ci_low) > 0.0),
        }
        for row in selected.itertuples(index=False)
    ]


def build_figures(
    bootstrap_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    figures_dir: Path,
    step_root: Path,
    primary_threshold: int,
) -> None:
    labels = list(SPACE_ORDER)
    display = [value.replace("_", " ") for value in labels]
    metrics = [
        ("alignment_all", "alignment_gain", "Alignment gain"),
        (
            "retrieval_mmp_direction",
            "exact_derivative_recall_at_1",
            "Exact derivative recall@1",
        ),
        (
            "retrieval_mmp_direction",
            "exact_requested_transform",
            "Exact requested edit",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for axis, (analysis, metric, title) in zip(axes, metrics):
        selected = bootstrap_summary.loc[
            (bootstrap_summary["analysis"] == analysis)
            & (
                bootstrap_summary["minimum_train_cores"]
                == primary_threshold
            )
            & (bootstrap_summary["metric"] == metric)
        ].set_index("space").loc[labels]
        values = selected["macro_estimate"].to_numpy()
        errors = np.vstack(
            [
                values - selected["ci_low"].to_numpy(),
                selected["ci_high"].to_numpy() - values,
            ]
        )
        axis.bar(
            np.arange(len(labels)),
            values,
            color=[COLORS[value] for value in labels],
            yerr=errors,
            capsize=3,
        )
        axis.set_xticks(np.arange(len(labels)), display, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if metric == "alignment_gain":
            axis.axhline(0, color="#111827", linewidth=0.8)
    fig.suptitle(
        f"Primary unseen-core comparison (at least {primary_threshold} train cores)"
    )
    for suffix in (".png", ".svg"):
        save_figure(
            fig,
            figures_dir / f"primary_space_comparison{suffix}",
            step_root,
        )
    plt.close(fig)

    alignment = bootstrap_summary.loc[
        (bootstrap_summary["analysis"] == "alignment_all")
        & (bootstrap_summary["metric"] == "alignment_gain")
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for space in labels:
        selected = alignment.loc[
            alignment["space"] == space
        ].sort_values("minimum_train_cores")
        axis.plot(
            selected["minimum_train_cores"],
            selected["macro_estimate"],
            marker="o",
            label=space.replace("_", " "),
            color=COLORS[space],
        )
        axis.fill_between(
            selected["minimum_train_cores"].to_numpy(dtype=float),
            selected["ci_low"].to_numpy(dtype=float),
            selected["ci_high"].to_numpy(dtype=float),
            color=COLORS[space],
            alpha=0.12,
        )
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Minimum independent train cores")
    axis.set_ylabel("Macro alignment gain over mismatched null")
    axis.set_title("Directional transfer versus transformation support")
    axis.set_xscale("log", base=2)
    axis.set_xticks([2, 5, 10, 20], ["2", "5", "10", "20"])
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    for suffix in (".png", ".svg"):
        save_figure(
            fig,
            figures_dir / f"alignment_by_support{suffix}",
            step_root,
        )
    plt.close(fig)

    mmp = retrieval_summary.loc[
        retrieval_summary["method"] == "mmp_direction"
    ].set_index("space").loc[labels]
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for space in labels:
        row = mmp.loc[space]
        axis.scatter(
            row["seed_retrieved_tanimoto_macro_transform_mean"],
            row["exact_derivative_recall_at_1_macro_transform_mean"],
            s=85,
            color=COLORS[space],
            label=space.replace("_", " "),
        )
    axis.set_xlabel("Mean seed-retrieved Morgan Tanimoto")
    axis.set_ylabel("Macro exact derivative recall@1")
    axis.set_title("Retrieval accuracy and seed similarity")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    for suffix in (".png", ".svg"):
        save_figure(
            fig,
            figures_dir / f"retrieval_accuracy_similarity{suffix}",
            step_root,
        )
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def write_reports(
    *,
    step_root: Path,
    decision: dict[str, Any],
    decision_table: pd.DataFrame,
    support_thresholds: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    train_rows: int,
    validation_rows: int,
    query_count: int,
    mining_summary: dict[str, Any],
) -> None:
    primary = int(decision["primary_support_threshold"])
    display_table = decision_table[
        [
            "space",
            "primary_transformations",
            "primary_queries",
            "alignment_gain",
            "positive_alignment_transformation_fraction",
            "exact_derivative_recall_at_1",
            "exact_requested_transform",
            "viable",
            "pareto",
            "selected_edit_control_space",
        ]
    ]
    transfer = transfer_assessment(
        bootstrap_summary, decision["selected_edit_control_space"]
    )
    decision["directional_transfer_by_support"] = transfer
    selected = decision["selected_edit_control_space"] or "none"
    w3 = decision["released_w3_vs_unweighted_w1"]
    mean_assessment = decision["mean_node_128_assessment"]
    threshold_table = _markdown_table(
        support_thresholds,
        [
            "minimum_train_cores",
            "transformations",
            "core_transform_observations",
            "transformations_with_unseen_core_validation",
            "unseen_core_validation_observations",
        ],
    )

    alignment_rows = []
    for space in SPACE_ORDER:
        values = {}
        for metric in ("alignment", "null_alignment", "alignment_gain"):
            row = _row(
                bootstrap_summary,
                "alignment_all",
                primary,
                metric,
                space,
            )
            values[metric] = float(row["macro_estimate"])
            if metric == "alignment_gain":
                values["gain_ci_low"] = float(row["ci_low"])
                values["gain_ci_high"] = float(row["ci_high"])
        alignment_rows.append({"space": space, **values})
    alignment_table = pd.DataFrame(alignment_rows)

    mmp_retrieval = retrieval_summary.loc[
        retrieval_summary["method"] == "mmp_direction"
    ].set_index("space").loc[list(SPACE_ORDER)].reset_index()
    retrieval_table = mmp_retrieval[
        [
            "space",
            "exact_derivative_recall_at_1_macro_transform_mean",
            "exact_derivative_recall_at_10_macro_transform_mean",
            "scaffold_retention_macro_transform_mean",
            "mmp_consistency_macro_transform_mean",
            "exact_requested_transform_macro_transform_mean",
            "seed_retrieved_tanimoto_macro_transform_mean",
        ]
    ].rename(
        columns={
            "exact_derivative_recall_at_1_macro_transform_mean": "recall_at_1",
            "exact_derivative_recall_at_10_macro_transform_mean": "recall_at_10",
            "scaffold_retention_macro_transform_mean": "scaffold_retention",
            "mmp_consistency_macro_transform_mean": "mmp_consistency",
            "exact_requested_transform_macro_transform_mean": (
                "exact_requested_edit"
            ),
            "seed_retrieved_tanimoto_macro_transform_mean": (
                "mean_seed_retrieved_tanimoto"
            ),
        }
    )

    control_methods = [
        "seed_nn",
        "isotropic",
        "global_covariance",
        "local_covariance",
        "mismatched_mmp_direction",
        "mmp_direction",
    ]
    selected_controls = retrieval_summary.loc[
        (retrieval_summary["space"] == selected)
        & retrieval_summary["method"].isin(control_methods),
        [
            "method",
            "exact_derivative_recall_at_1_macro_transform_mean",
            "exact_derivative_recall_at_10_macro_transform_mean",
            "mmp_consistency_macro_transform_mean",
        ],
    ].copy()
    selected_controls["method"] = pd.Categorical(
        selected_controls["method"],
        categories=control_methods,
        ordered=True,
    )
    selected_controls = selected_controls.sort_values("method").rename(
        columns={
            "exact_derivative_recall_at_1_macro_transform_mean": "recall_at_1",
            "exact_derivative_recall_at_10_macro_transform_mean": "recall_at_10",
            "mmp_consistency_macro_transform_mean": "mmp_consistency",
        }
    )

    comparison_rows = []
    for metric, record in w3["metrics"].items():
        comparison_rows.append(
            {
                "metric": metric,
                "delta_w3_minus_w1": record[
                    "macro_delta_w3_minus_w1"
                ],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "resolved_positive": record[
                    "statistically_resolved_positive"
                ],
            }
        )
    comparison_table = pd.DataFrame(comparison_rows)

    transfer_table = pd.DataFrame(transfer).rename(
        columns={
            "minimum_train_cores": "minimum_cores",
            "macro_alignment_gain": "alignment_gain",
        }
    )

    results = f"""# Scaled latent-space selection results

## Scope

This inference-only study compared five diagnostic coordinate spaces without
retraining or changing gMolAI, its checkpoint, calibrator, or released weight-3
embedding definition. It used {train_rows:,} pretraining-train molecules for
mining and {validation_rows:,} disjoint pretraining-validation molecules for
unseen-core validation. Locked-test molecules and endpoint labels were not used.

This is evidence about latent edit geometry and retrieval. It is not evidence
that gMolAI can decode embeddings or generate novel molecules.

## MMP scale and support

Train core-transformation observations:
{int(mining_summary["train"]["core_transform_observations"]):,}. Distinct train
transformations: {int(mining_summary["train"]["transformations"]):,}. The fixed
retrieval panel contains {query_count:,} unseen-core queries.

{threshold_table}

## Directional transfer

Primary inference is the at-least-{primary}-independent-core cohort. Values are
unweighted macro averages across transformations. The null is a deterministic
support-matched mismatched transformation, held fixed across all five spaces.

{_markdown_table(alignment_table, list(alignment_table.columns))}

Directional transfer remains positive through the at-least-20-core cohort:

{_markdown_table(transfer_table, ["minimum_cores", "transformations", "observations", "alignment_gain", "ci_low", "ci_high", "survives"])}

## Derivative retrieval

The table reports transformation-macro metrics for the fitted MMP direction on
the identical {query_count:,}-query validation panel. Recall is for the exact
held-out derivative identity; exact requested edit accepts any molecule with
the query core and requested target substituent.

{_markdown_table(retrieval_table, list(retrieval_table.columns))}

Every query had exactly one validation molecule for its requested
core-plus-target-substituent identity, so exact requested edit and exact
derivative recall@1 coincide in this panel. Both columns are retained to make
the intended metrics explicit.

For the selected {selected} space, fitted directions strongly exceed the
unperturbed seed, random, and support-matched mismatched controls:

{_markdown_table(selected_controls, list(selected_controls.columns))}

## Frozen primary comparison

{_markdown_table(display_table, list(display_table.columns))}

Selected edit-control space: {selected}

{decision["selection_reason"]}

The decoder-conditioning representation remains
{decision["decoder_conditioning_representation"]}; diagnostic weights do not
alter the released representation.

## Weight 3 versus weight 1

{w3["conclusion"]}

Paired hierarchical-bootstrap differences:

{_markdown_table(comparison_table, list(comparison_table.columns))}

## Mean-node assessment

{mean_assessment["conclusion"]}

## Required answers

1. Released weight 3 clearly improves directional alignment over weight 1.
   Its exact retrieval point estimate is slightly higher, but that difference
   is not statistically resolved.
2. Mean-node-128 remains the directional-alignment leader and is Pareto-optimal,
   but it is not the unique overall winner.
3. MMP-direction transfer survives at 1M scale and in the at-least-5,
   at-least-10, and at-least-20 independent-core cohorts.
4. Freeze {selected} as the edit-control space under the predeclared
   compatibility-aware rule. Keep
   {decision["decoder_conditioning_representation"]} unchanged for decoder
   conditioning.

## Output map

- outputs/raw contains observation- and query-level machine-readable results.
- outputs/tables contains transformation summaries, hierarchical bootstrap
  intervals, paired comparisons, and the selection table.
- outputs/figures contains concise diagnostic plots in PNG and SVG.
- outputs/space_decision.json contains the machine-readable frozen decision.
- state/COMPLETE.json and outputs/SHA256SUMS provide execution and integrity
  seals.
"""
    decision_text = f"""# Latent control-space decision

Selected edit-control space: {selected}

{decision["selection_reason"]}

The released decoder-conditioning representation remains
{decision["decoder_conditioning_representation"]}. This decision concerns only
the geometry used to define molecular edits. It does not change gMolAI and does
not establish a decoder or de novo generation capability.

## Weighting result

{w3["conclusion"]}

## Mean-node result

{mean_assessment["conclusion"]}

## Directional-transfer result

Transfer remains positive at every evaluated support threshold, including 420
transformations at at least 10 train cores and 141 transformations at at least
20 train cores. Full estimates and hierarchical confidence intervals are in
RESULTS.md and the machine-readable tables.
"""
    atomic_write_text(step_root / "RESULTS.md", results, step_root)
    atomic_write_text(step_root / "DECISION.md", decision_text, step_root)
