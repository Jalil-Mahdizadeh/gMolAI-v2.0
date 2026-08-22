#!/usr/bin/env python3
"""Aggregate draws within targets and perform target-level paired inference."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_csv,
    sha256_file,
    write_csv,
)
from metrics import deterministic_seed


METRICS = ("ef1", "bedroc20", "roc_auc", "average_precision")


def bootstrap_estimates(
    values: np.ndarray,
    classes: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if values.ndim != 1 or classes.shape != values.shape or values.size == 0:
        raise ValueError("Invalid target-level bootstrap input")
    generator = np.random.default_rng(seed)
    groups = [np.flatnonzero(classes == label) for label in sorted(set(classes.tolist()))]
    means = np.empty(repetitions, dtype=np.float64)
    medians = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = np.concatenate(
            [generator.choice(group, size=group.size, replace=True) for group in groups]
        )
        means[repetition] = float(values[sampled].mean())
        medians[repetition] = float(np.median(values[sampled]))
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def main() -> None:
    protocol = load_protocol()
    retrieval_state_path = BENCHMARK_DIR / "state/RETRIEVAL_COMPLETE.json"
    retrieval_state = load_json(retrieval_state_path)
    draw_path = BENCHMARK_DIR / "results/tables/retrieval_per_draw.csv"
    if retrieval_state.get("status") != "ok" or retrieval_state.get(
        "retrieval_per_draw_sha256"
    ) != sha256_file(draw_path):
        raise RuntimeError("Retrieval output is absent or changed")
    rows = read_csv(draw_path)
    models = tuple(protocol["models"]["primary_order"])
    all_models = (*models, "random")
    draws_expected = int(protocol["retrieval"]["draws_per_target"])
    scaffold_min_draws = int(
        protocol["coverage_and_eligibility"]["scaffold_target_minimum_eligible_draws"]
    )

    invariant_groups: dict[tuple[str, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        invariant_groups[
            (
                row["condition"],
                int(row["shots"]),
                row["target_id"],
                int(row["draw_id"]),
            )
        ].append(row)
    for key, group in invariant_groups.items():
        if {row["model"] for row in group} != set(all_models):
            raise RuntimeError(f"Model rows differ for retrieval draw {key}")
        for field in (
            "anchor_identity_sha256",
            "anchor_molecule_hashes",
            "candidate_count",
            "active_count",
            "inactive_or_lower_affinity_count",
            "cutoff_k",
            "realized_screened_fraction",
        ):
            if len({row[field] for row in group}) != 1:
                raise RuntimeError(f"Cross-model {field} differs for draw {key}")

    grouped: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["model"],
                row["target_id"],
                row["target_class"],
                int(row["shots"]),
                row["condition"],
            )
        ].append(row)
    target_rows = []
    for (model, target_id, target_class, shots, condition), group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["draw_id"]))
        draw_count = len(group)
        included = (
            draw_count == draws_expected
            if condition == "standard"
            else draw_count >= scaffold_min_draws
        )
        output: dict[str, object] = {
            "model": model,
            "display_name": group[0]["display_name"],
            "is_random_control": group[0]["is_random_control"],
            "target_id": target_id,
            "target_class": target_class,
            "shots": shots,
            "condition": condition,
            "eligible_draws": draw_count,
            "included_in_across_target_summary": included,
            "candidate_count_min": min(int(row["candidate_count"]) for row in group),
            "candidate_count_max": max(int(row["candidate_count"]) for row in group),
            "active_count_min": min(int(row["active_count"]) for row in group),
            "active_count_max": max(int(row["active_count"]) for row in group),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_median"] = float(np.median(values))
            output[f"{metric}_sd"] = float(values.std(ddof=0))
            output[f"{metric}_min"] = float(values.min())
            output[f"{metric}_max"] = float(values.max())
        target_rows.append(output)
    target_path = BENCHMARK_DIR / "results/tables/retrieval_per_target.csv"
    write_csv(target_path, target_rows, tuple(target_rows[0]))

    repetitions = int(protocol["statistics"]["bootstrap_repetitions"])
    bootstrap_master = int(protocol["statistics"]["bootstrap_seed"])
    summary_rows = []
    settings = sorted(
        {
            (int(row["shots"]), str(row["condition"]))
            for row in target_rows
            if bool(row["included_in_across_target_summary"])
        }
    )
    for shots, condition in settings:
        for model in all_models:
            subset = [
                row
                for row in target_rows
                if row["model"] == model
                and int(row["shots"]) == shots
                and row["condition"] == condition
                and bool(row["included_in_across_target_summary"])
            ]
            if not subset:
                continue
            for metric in METRICS:
                values = np.asarray(
                    [float(row[f"{metric}_mean"]) for row in subset], dtype=np.float64
                )
                classes = np.asarray([row["target_class"] for row in subset], dtype=object)
                ci_mean_low, ci_mean_high, ci_median_low, ci_median_high = bootstrap_estimates(
                    values,
                    classes,
                    repetitions=repetitions,
                    seed=deterministic_seed(
                        "model-summary", bootstrap_master, shots, condition, model, metric
                    ),
                )
                summary_rows.append(
                    {
                        "shots": shots,
                        "condition": condition,
                        "model": model,
                        "display_name": subset[0]["display_name"],
                        "is_random_control": model == "random",
                        "metric": metric,
                        "targets": len(subset),
                        "target_level_mean": float(values.mean()),
                        "target_level_mean_ci95_lower": ci_mean_low,
                        "target_level_mean_ci95_upper": ci_mean_high,
                        "target_level_median": float(np.median(values)),
                        "target_level_median_ci95_lower": ci_median_low,
                        "target_level_median_ci95_upper": ci_median_high,
                        "target_level_sd": float(values.std(ddof=0)),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_unit": "protein_target_stratified_by_target_class",
                    }
                )
    summary_path = BENCHMARK_DIR / "results/tables/model_summary.csv"
    write_csv(summary_path, summary_rows, tuple(summary_rows[0]))

    paired_rows = []
    for shots, condition in settings:
        for comparator in models:
            if comparator == "gmolai":
                continue
            for metric in METRICS:
                reference = {
                    row["target_id"]: row
                    for row in target_rows
                    if row["model"] == "gmolai"
                    and int(row["shots"]) == shots
                    and row["condition"] == condition
                    and bool(row["included_in_across_target_summary"])
                }
                compared = {
                    row["target_id"]: row
                    for row in target_rows
                    if row["model"] == comparator
                    and int(row["shots"]) == shots
                    and row["condition"] == condition
                    and bool(row["included_in_across_target_summary"])
                }
                common_targets = sorted(set(reference) & set(compared))
                if set(reference) != set(compared) or not common_targets:
                    raise RuntimeError(
                        f"Paired target population differs for {shots}/{condition}/{comparator}"
                    )
                gmolai_values = np.asarray(
                    [float(reference[target][f"{metric}_mean"]) for target in common_targets]
                )
                comparator_values = np.asarray(
                    [float(compared[target][f"{metric}_mean"]) for target in common_targets]
                )
                differences = gmolai_values - comparator_values
                classes = np.asarray(
                    [reference[target]["target_class"] for target in common_targets],
                    dtype=object,
                )
                ci_mean_low, ci_mean_high, ci_median_low, ci_median_high = bootstrap_estimates(
                    differences,
                    classes,
                    repetitions=repetitions,
                    seed=deterministic_seed(
                        "paired", bootstrap_master, shots, condition, comparator, metric
                    ),
                )
                wins = int(np.count_nonzero(differences > 0.0))
                losses = int(np.count_nonzero(differences < 0.0))
                ties = int(np.count_nonzero(differences == 0.0))
                paired_rows.append(
                    {
                        "shots": shots,
                        "condition": condition,
                        "metric": metric,
                        "reference_model": "gmolai",
                        "comparator_model": comparator,
                        "targets": len(common_targets),
                        "reference_mean": float(gmolai_values.mean()),
                        "comparator_mean": float(comparator_values.mean()),
                        "paired_mean_difference": float(differences.mean()),
                        "paired_mean_difference_ci95_lower": ci_mean_low,
                        "paired_mean_difference_ci95_upper": ci_mean_high,
                        "paired_median_difference": float(np.median(differences)),
                        "paired_median_difference_ci95_lower": ci_median_low,
                        "paired_median_difference_ci95_upper": ci_median_high,
                        "strict_wins": wins,
                        "strict_losses": losses,
                        "exact_ties": ties,
                        "strict_win_fraction": wins / len(common_targets),
                        "strict_loss_fraction": losses / len(common_targets),
                        "tie_fraction": ties / len(common_targets),
                        "formal_p_value": "not_performed",
                    }
                )
    paired_path = BENCHMARK_DIR / "results/tables/paired_comparisons.csv"
    write_csv(paired_path, paired_rows, tuple(paired_rows[0]))

    scaffold_rows = [
        row
        for row in target_rows
        if row["condition"] == "scaffold_excluded"
        and int(row["shots"]) == int(protocol["retrieval"]["primary_shots"])
        and row["model"] in models
        and bool(row["included_in_across_target_summary"])
    ]
    scaffold_path = BENCHMARK_DIR / "results/tables/scaffold_excluded_summary.csv"
    write_csv(scaffold_path, scaffold_rows, tuple(scaffold_rows[0]))

    random_primary = [
        float(row["ef1"])
        for row in rows
        if row["model"] == "random"
        and row["condition"] == "standard"
        and int(row["shots"]) == int(protocol["retrieval"]["primary_shots"])
    ]
    random_mean = float(np.mean(random_primary))
    if not 0.7 <= random_mean <= 1.3:
        raise RuntimeError(
            f"Random-ranking EF1 control is unexpectedly far from 1: {random_mean}"
        )
    result = {
        "schema_version": 1,
        "status": "ok",
        "retrieval_state_sha256": sha256_file(retrieval_state_path),
        "retrieval_per_target": str(target_path),
        "retrieval_per_target_sha256": sha256_file(target_path),
        "model_summary": str(summary_path),
        "model_summary_sha256": sha256_file(summary_path),
        "paired_comparisons": str(paired_path),
        "paired_comparisons_sha256": sha256_file(paired_path),
        "scaffold_excluded_summary": str(scaffold_path),
        "scaffold_excluded_summary_sha256": sha256_file(scaffold_path),
        "random_primary_ef1_mean": random_mean,
        "random_primary_ef1_draws": len(random_primary),
        "anchor_draws_treated_as_inferential_replicates": False,
        "bootstrap_unit": "protein_target_stratified_by_target_class",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

