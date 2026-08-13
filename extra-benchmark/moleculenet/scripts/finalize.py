#!/usr/bin/env python3
"""Aggregate, audit, and checksum-seal the completed endpoint benchmark."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    protocol_digest,
    sha256_file,
)


def primary_metric(task: str) -> str:
    return "rmse" if task == "regression" else "roc_auc"


def main() -> None:
    protocol = load_protocol()
    model_order = tuple(protocol["comparators"]["model_order"])
    model_specs = protocol["comparators"]["models"]
    dataset_order = tuple(protocol["datasets"]["order"])
    common_manifest_path = BENCHMARK_DIR / "inputs" / "common_manifest.json"
    common_manifest = load_json(common_manifest_path)

    model_results: dict[str, dict[str, Any]] = {}
    for model in model_order:
        path = BENCHMARK_DIR / "outputs" / "results" / f"{model}.json"
        result = load_json(path)
        if result.get("status") != "complete" or result.get("model") != model:
            raise RuntimeError(f"Incomplete evaluator result for {model}")
        if result["common_panel"]["ordered_identity_sha256"] != common_manifest[
            "ordered_identity_sha256"
        ]:
            raise RuntimeError(f"{model} result used a different common panel")
        if set(result["datasets"]) != set(dataset_order):
            raise RuntimeError(f"{model} dataset order changed")
        model_results[model] = result

    primary_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    headline: dict[str, Any] = {}
    for dataset in dataset_order:
        task = model_results["gmolai"]["datasets"][dataset]["task"]
        metric = primary_metric(task)
        values_by_model = {
            model: model_results[model]["datasets"][dataset]["feature_results"][
                "summary"
            ][metric]
            for model in model_order
        }
        ranked = sorted(
            model_order,
            key=lambda model: values_by_model[model]["mean"],
            reverse=(task == "classification"),
        )
        gmolai_values = np.asarray(values_by_model["gmolai"]["values"])
        for rank, model in enumerate(ranked, start=1):
            summary = values_by_model[model]
            model_values = np.asarray(summary["values"])
            advantage = (
                gmolai_values - model_values
                if task == "classification"
                else model_values - gmolai_values
            )
            primary_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "common_molecules": common_manifest["datasets"][dataset][
                        "common_rows"
                    ],
                    "model": model,
                    "display_name": model_specs[model]["display_name"],
                    "dimension": model_specs[model]["dimension"],
                    "primary_metric": metric,
                    "mean": summary["mean"],
                    "population_std": summary["std"],
                    "rank": rank,
                    "gmolai_favorable_paired_mean_difference": float(
                        advantage.mean()
                    ),
                    "gmolai_split_wins": int(np.count_nonzero(advantage > 0)),
                    "ties": int(np.count_nonzero(advantage == 0)),
                    "gmolai_split_losses": int(np.count_nonzero(advantage < 0)),
                }
            )
            for metric_name, metric_summary in model_results[model]["datasets"][
                dataset
            ]["feature_results"]["summary"].items():
                all_metric_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "model": model,
                        "display_name": model_specs[model]["display_name"],
                        "metric": metric_name,
                        "mean": metric_summary["mean"],
                        "population_std": metric_summary["std"],
                        "values_json": json.dumps(metric_summary["values"]),
                    }
                )
        headline[dataset] = {
            "task": task,
            "primary_metric": metric,
            "common_molecules": common_manifest["datasets"][dataset]["common_rows"],
            "best_model": ranked[0],
            "best_display_name": model_specs[ranked[0]]["display_name"],
            "best_mean": values_by_model[ranked[0]]["mean"],
            "gmolai_rank": ranked.index("gmolai") + 1,
            "gmolai_mean": values_by_model["gmolai"]["mean"],
            "morgan_rank": ranked.index("morgan") + 1,
            "morgan_mean": values_by_model["morgan"]["mean"],
        }

    primary_path = BENCHMARK_DIR / "outputs" / "common_panel_primary_metrics.csv"
    atomic_write_csv(
        primary_path,
        primary_rows,
        (
            "dataset",
            "task",
            "common_molecules",
            "model",
            "display_name",
            "dimension",
            "primary_metric",
            "mean",
            "population_std",
            "rank",
            "gmolai_favorable_paired_mean_difference",
            "gmolai_split_wins",
            "ties",
            "gmolai_split_losses",
        ),
    )
    all_metrics_path = BENCHMARK_DIR / "outputs" / "common_panel_all_metrics.csv"
    atomic_write_csv(
        all_metrics_path,
        all_metric_rows,
        (
            "dataset",
            "task",
            "model",
            "display_name",
            "metric",
            "mean",
            "population_std",
            "values_json",
        ),
    )

    development = load_json(
        REPOSITORY_ROOT / protocol["references"]["development"]["path"]
    )
    hiv = load_json(REPOSITORY_ROOT / protocol["references"]["hiv"]["path"])
    descriptor = load_json(
        REPOSITORY_ROOT / protocol["references"]["descriptor_control"]["path"]
    )
    full_rows: list[dict[str, Any]] = []
    for dataset in dataset_order:
        reference = hiv if dataset == "hiv" else development
        source = reference["datasets"][dataset]
        task = source["task"]
        metric = primary_metric(task)
        features = source["feature_results"]
        for model, feature in (
            ("gmolai", "molecule_embedding"),
            ("morgan", "morgan_radius2_2048"),
        ):
            if feature not in features:
                continue
            summary = features[feature]["summary"][metric]
            full_rows.append(
                {
                    "dataset": dataset,
                    "full_molecules": source["preparation"]["molecules"],
                    "control": model,
                    "display_name": model_specs[model]["display_name"],
                    "primary_metric": metric,
                    "mean": summary["mean"],
                    "population_std": summary["std"],
                    "source_artifact": protocol["references"][
                        "hiv" if dataset == "hiv" else "development"
                    ]["path"],
                }
            )
        if dataset in descriptor["datasets"]:
            summary = descriptor["datasets"][dataset]["feature_results"][
                "auxiliary_descriptors_13"
            ]["summary"][metric]
            full_rows.append(
                {
                    "dataset": dataset,
                    "full_molecules": source["preparation"]["molecules"],
                    "control": "descriptor_13",
                    "display_name": "13-descriptor control",
                    "primary_metric": metric,
                    "mean": summary["mean"],
                    "population_std": summary["std"],
                    "source_artifact": protocol["references"]["descriptor_control"][
                        "path"
                    ],
                }
            )
    full_reference_path = BENCHMARK_DIR / "outputs" / "full_panel_reference_controls.csv"
    atomic_write_csv(
        full_reference_path,
        full_rows,
        (
            "dataset",
            "full_molecules",
            "control",
            "display_name",
            "primary_metric",
            "mean",
            "population_std",
            "source_artifact",
        ),
    )

    timing_rows = []
    for model in model_order:
        metadata = load_json(
            BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}.json"
        )
        timing_rows.append(
            {
                "model": model,
                "display_name": model_specs[model]["display_name"],
                "rows": metadata["rows"],
                "dimension": metadata["dimension"],
                "wall_seconds_including_load_warmup_export": metadata[
                    "wall_seconds_model_load_warmup_and_export"
                ],
                "rows_per_second_including_load_warmup_export": metadata[
                    "rows_per_second_including_load_warmup_and_export"
                ],
                "peak_gpu_memory_bytes": metadata.get("peak_gpu_memory_bytes"),
                "gpu_name": metadata.get("gpu_name"),
                "interpretation": (
                    "observed common-panel export timing; not the separately controlled "
                    "scalability benchmark"
                ),
            }
        )
    timing_path = BENCHMARK_DIR / "outputs" / "encoding_runtime_observed.csv"
    atomic_write_csv(
        timing_path,
        timing_rows,
        (
            "model",
            "display_name",
            "rows",
            "dimension",
            "wall_seconds_including_load_warmup_export",
            "rows_per_second_including_load_warmup_export",
            "peak_gpu_memory_bytes",
            "gpu_name",
            "interpretation",
        ),
    )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "Frozen-feature MoleculeNet development panel plus HIV confirmation",
        "protocol_sha256": protocol_digest(protocol),
        "common_panel": common_manifest,
        "headline": headline,
        "interpretation_constraints": [
            "BACE, BBBP, ESOL, FreeSolv and Lipophilicity remain selection-conditioned development evidence.",
            "HIV remains an external post-selection confirmatory endpoint.",
            "All neural encoders were frozen; only fold-local linear predictors were fitted.",
            "Population standard deviations summarize ten overlapping outer scaffold splits and are not standard errors.",
            "Common-panel results and full-panel historical controls are deliberately reported separately.",
            "Observed export runtimes are provenance metadata, not a controlled speed leaderboard.",
        ],
        "artifacts": {
            "primary_csv": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
            "all_metrics_csv": {"path": str(all_metrics_path), "sha256": sha256_file(all_metrics_path)},
            "coverage_csv": {
                "path": str(BENCHMARK_DIR / "outputs" / "coverage.csv"),
                "sha256": sha256_file(BENCHMARK_DIR / "outputs" / "coverage.csv"),
            },
            "full_reference_csv": {
                "path": str(full_reference_path),
                "sha256": sha256_file(full_reference_path),
            },
            "observed_runtime_csv": {"path": str(timing_path), "sha256": sha256_file(timing_path)},
        },
    }
    summary_path = BENCHMARK_DIR / "outputs" / "moleculenet_hiv_summary.json"
    atomic_write_json(summary_path, summary)

    output_files = sorted(
        path
        for path in (BENCHMARK_DIR / "outputs").rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(BENCHMARK_DIR / 'outputs')}"
        for path in output_files
    ]
    checksum_path = BENCHMARK_DIR / "outputs" / "SHA256SUMS"
    atomic_write_text(checksum_path, "\n".join(lines) + "\n")
    complete = {
        "schema_version": 1,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "protocol_sha256": protocol_digest(protocol),
        "summary_sha256": sha256_file(summary_path),
        "sha256sums_sha256": sha256_file(checksum_path),
        "files_hashed": len(lines),
        "models": list(model_order),
        "datasets": list(dataset_order),
        "common_rows": common_manifest["rows"],
        "neural_training_or_finetuning": False,
        "checkpoint_or_calibrator_modified": False,
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "COMPLETE.json", complete)
    print(json.dumps({"status": "complete", "headline": headline}, sort_keys=True))


if __name__ == "__main__":
    main()
