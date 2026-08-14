#!/usr/bin/env python3
"""Aggregate, rank, audit, and checksum-seal the TDC ADMET results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    protocol_digest,
    sha256_file,
)


def display_names(protocol: dict[str, Any]) -> dict[str, str]:
    values = {
        model: spec["display_name"]
        for model, spec in protocol["comparators"]["models"].items()
    }
    values["descriptor_13"] = protocol["diagnostic_control"]["display_name"]
    return values


def main() -> None:
    protocol = load_protocol()
    primary_models = tuple(protocol["comparators"]["model_order"])
    all_models = (*primary_models, "descriptor_13")
    endpoints = tuple(protocol["data"]["endpoint_order"])
    names = display_names(protocol)
    common_manifest = load_json(BENCHMARK_DIR / "inputs" / "common_manifest.json")
    overlap_audit = load_json(
        BENCHMARK_DIR / "inputs" / "prior_development_overlap.json"
    )
    results: dict[str, dict[str, Any]] = {}
    for model in all_models:
        path = BENCHMARK_DIR / "outputs" / "results" / f"{model}.json"
        result = load_json(path)
        if result.get("status") != "complete" or result.get("model") != model:
            raise RuntimeError(f"Incomplete result for {model}")
        if result["common_panel"]["ordered_identity_sha256"] != common_manifest[
            "ordered_identity_sha256"
        ]:
            raise RuntimeError(f"{model} evaluated a different common panel")
        if set(result["endpoints"]) != set(endpoints):
            raise RuntimeError(f"Endpoint membership changed for {model}")
        results[model] = result

    primary_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    endpoint_ranks: dict[str, dict[str, float]] = {}
    headline: dict[str, Any] = {}
    for endpoint in endpoints:
        spec = protocol["data"]["endpoints"][endpoint]
        metric = spec["metric"]
        maximize = metric != "mae"
        means = np.asarray(
            [
                results[model]["endpoints"][endpoint]["result"]["primary"]["mean"]
                for model in primary_models
            ],
            dtype=np.float64,
        )
        ranks = rankdata(-means if maximize else means, method="average")
        endpoint_ranks[endpoint] = {
            model: float(rank) for model, rank in zip(primary_models, ranks)
        }
        gmolai_values = np.asarray(
            results["gmolai"]["endpoints"][endpoint]["result"]["primary"]["values"]
        )
        for model in all_models:
            endpoint_result = results[model]["endpoints"][endpoint]["result"]
            summary = endpoint_result["primary"]
            model_values = np.asarray(summary["values"])
            favorable = (
                gmolai_values - model_values if maximize else model_values - gmolai_values
            )
            primary_rows.append(
                {
                    "endpoint": endpoint,
                    "category": spec["category"],
                    "task": spec["task"],
                    "common_occurrences": results[model]["endpoints"][endpoint][
                        "common_occurrences"
                    ],
                    "common_unique_identities": results[model]["endpoints"][endpoint][
                        "common_unique_identities"
                    ],
                    "model": model,
                    "display_name": names[model],
                    "official_metric": metric,
                    "direction": "maximize" if maximize else "minimize",
                    "mean": summary["mean"],
                    "population_std": summary["population_std"],
                    "primary_seven_model_rank": (
                        endpoint_ranks[endpoint][model] if model in primary_models else ""
                    ),
                    "gmolai_favorable_paired_mean_difference": float(favorable.mean()),
                    "gmolai_seed_wins": int(np.count_nonzero(favorable > 0)),
                    "ties": int(np.count_nonzero(favorable == 0)),
                    "gmolai_seed_losses": int(np.count_nonzero(favorable < 0)),
                    "strict_identity_disjoint_mean": (
                        endpoint_result["strict_identity_disjoint_primary"]["mean"]
                        if endpoint_result["strict_identity_disjoint_primary"] is not None
                        else ""
                    ),
                }
            )
            for metric_name, metric_summary in endpoint_result["all_metrics"].items():
                all_metric_rows.append(
                    {
                        "endpoint": endpoint,
                        "category": spec["category"],
                        "task": spec["task"],
                        "model": model,
                        "display_name": names[model],
                        "metric": metric_name,
                        "mean": metric_summary["mean"],
                        "population_std": metric_summary["population_std"],
                        "values_json": json.dumps(metric_summary["values"]),
                    }
                )
        ordered = sorted(
            primary_models,
            key=lambda model: endpoint_ranks[endpoint][model],
        )
        headline[endpoint] = {
            "category": spec["category"],
            "official_metric": metric,
            "best_primary_model": ordered[0],
            "best_primary_display_name": names[ordered[0]],
            "gmolai_rank": endpoint_ranks[endpoint]["gmolai"],
            "gmolai_mean": results["gmolai"]["endpoints"][endpoint]["result"][
                "primary"
            ]["mean"],
            "morgan_rank": endpoint_ranks[endpoint]["morgan"],
        }

    primary_path = BENCHMARK_DIR / "outputs" / "endpoint_primary_metrics.csv"
    atomic_write_csv(
        primary_path,
        primary_rows,
        (
            "endpoint",
            "category",
            "task",
            "common_occurrences",
            "common_unique_identities",
            "model",
            "display_name",
            "official_metric",
            "direction",
            "mean",
            "population_std",
            "primary_seven_model_rank",
            "gmolai_favorable_paired_mean_difference",
            "gmolai_seed_wins",
            "ties",
            "gmolai_seed_losses",
            "strict_identity_disjoint_mean",
        ),
    )
    all_metrics_path = BENCHMARK_DIR / "outputs" / "all_metrics.csv"
    atomic_write_csv(
        all_metrics_path,
        all_metric_rows,
        (
            "endpoint",
            "category",
            "task",
            "model",
            "display_name",
            "metric",
            "mean",
            "population_std",
            "values_json",
        ),
    )

    category_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    categories = tuple(protocol["evaluation"]["categories"])
    excluded = set(
        protocol["selection_conditioning"]["direct_or_near_reuse_endpoints"]
    )
    for model in primary_models:
        category_means = []
        robust_category_means = []
        all_ranks = []
        robust_ranks = []
        for category in categories:
            category_endpoints = [
                endpoint
                for endpoint in endpoints
                if protocol["data"]["endpoints"][endpoint]["category"] == category
            ]
            category_robust_endpoints = [
                endpoint for endpoint in category_endpoints if endpoint not in excluded
            ]
            ranks = [endpoint_ranks[endpoint][model] for endpoint in category_endpoints]
            sensitivity_ranks = [
                endpoint_ranks[endpoint][model] for endpoint in category_robust_endpoints
            ]
            mean_rank = float(np.mean(ranks))
            robust_mean_rank = float(np.mean(sensitivity_ranks))
            category_means.append(mean_rank)
            robust_category_means.append(robust_mean_rank)
            all_ranks.extend(ranks)
            robust_ranks.extend(sensitivity_ranks)
            for analysis, analysis_endpoints, analysis_ranks, analysis_mean in (
                ("panel_complete_22", category_endpoints, ranks, mean_rank),
                (
                    "selection_robust_19",
                    category_robust_endpoints,
                    sensitivity_ranks,
                    robust_mean_rank,
                ),
            ):
                category_rows.append(
                    {
                        "analysis": analysis,
                        "model": model,
                        "display_name": names[model],
                        "category": category,
                        "endpoints": len(analysis_endpoints),
                        "mean_endpoint_rank": analysis_mean,
                        "endpoint_ranks_json": json.dumps(analysis_ranks),
                    }
                )
        model_rows.append(
            {
                "model": model,
                "display_name": names[model],
                "category_balanced_mean_rank": float(np.mean(category_means)),
                "endpoint_mean_rank": float(np.mean(all_ranks)),
                "endpoint_median_rank": float(np.median(all_ranks)),
                "rank_one_endpoints": int(np.count_nonzero(np.asarray(all_ranks) == 1.0)),
                "top_three_endpoints": int(np.count_nonzero(np.asarray(all_ranks) <= 3.0)),
                "selection_robust_category_balanced_mean_rank": float(
                    np.mean(robust_category_means)
                ),
                "selection_robust_endpoint_mean_rank": float(np.mean(robust_ranks)),
                "selection_robust_endpoint_median_rank": float(np.median(robust_ranks)),
            }
        )
    complete_ranks = rankdata(
        [row["category_balanced_mean_rank"] for row in model_rows], method="average"
    )
    robust_summary_ranks = rankdata(
        [row["selection_robust_category_balanced_mean_rank"] for row in model_rows],
        method="average",
    )
    for row, complete_rank, robust_rank in zip(
        model_rows, complete_ranks, robust_summary_ranks
    ):
        row["category_balanced_rank"] = float(complete_rank)
        row["selection_robust_category_balanced_rank"] = float(robust_rank)
    model_rows.sort(key=lambda row: row["category_balanced_rank"])

    category_path = BENCHMARK_DIR / "outputs" / "category_rank_summary.csv"
    atomic_write_csv(
        category_path,
        category_rows,
        (
            "analysis",
            "model",
            "display_name",
            "category",
            "endpoints",
            "mean_endpoint_rank",
            "endpoint_ranks_json",
        ),
    )
    model_path = BENCHMARK_DIR / "outputs" / "model_summary.csv"
    atomic_write_csv(
        model_path,
        model_rows,
        (
            "category_balanced_rank",
            "model",
            "display_name",
            "category_balanced_mean_rank",
            "endpoint_mean_rank",
            "endpoint_median_rank",
            "rank_one_endpoints",
            "top_three_endpoints",
            "selection_robust_category_balanced_rank",
            "selection_robust_category_balanced_mean_rank",
            "selection_robust_endpoint_mean_rank",
            "selection_robust_endpoint_median_rank",
        ),
    )

    timing_rows = []
    for model in all_models:
        metadata = load_json(
            BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}.json"
        )
        timing_rows.append(
            {
                "model": model,
                "display_name": names[model],
                "unique_identities": metadata["rows"],
                "dimension": metadata["dimension"],
                "wall_seconds_including_load_warmup_export": metadata[
                    "wall_seconds_model_load_warmup_and_export"
                ],
                "rows_per_second_including_load_warmup_export": metadata[
                    "rows_per_second_including_load_warmup_and_export"
                ],
                "peak_gpu_memory_bytes": metadata.get("peak_gpu_memory_bytes"),
                "gpu_name": metadata.get("gpu_name"),
                "interpretation": "observed export provenance; not a controlled speed benchmark",
            }
        )
    timing_path = BENCHMARK_DIR / "outputs" / "encoding_runtime_observed.csv"
    atomic_write_csv(
        timing_path,
        timing_rows,
        (
            "model",
            "display_name",
            "unique_identities",
            "dimension",
            "wall_seconds_including_load_warmup_export",
            "rows_per_second_including_load_warmup_export",
            "peak_gpu_memory_bytes",
            "gpu_name",
            "interpretation",
        ),
    )

    gmolai_summary = next(row for row in model_rows if row["model"] == "gmolai")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": protocol["study"]["name"],
        "protocol_sha256": protocol_digest(protocol),
        "source": {
            "title": protocol["data"]["title"],
            "doi": protocol["data"]["doi"],
            "archive_sha256": protocol["data"]["archive_sha256"],
        },
        "common_panel": common_manifest,
        "selection_conditioning": {
            "audit": overlap_audit,
            "panel_complete_endpoints": len(endpoints),
            "selection_robust_endpoints": len(endpoints) - len(excluded),
            "excluded_from_sensitivity_only": sorted(excluded),
        },
        "primary_model_summary": model_rows,
        "gmolai_summary": gmolai_summary,
        "endpoint_headline": headline,
        "descriptor_control": {
            "status": "diagnostic_only",
            "included_in_primary_rank": False,
            "reason": "hand-designed physicochemical features are closely related to several ADMET endpoints",
        },
        "interpretation_constraints": [
            "All 22 endpoints were included without result-based endpoint selection.",
            "BBB and Lipophilicity are exact prior development reuse and AqSolDB strongly overlaps ESOL; the predeclared 19-endpoint sensitivity addresses this selection conditioning.",
            "All neural representations were frozen; only shallow endpoint probes were fitted.",
            "Every fixed test set remained isolated from scaling and regularization selection.",
            "The common panel is a support intersection and is not a literal unfiltered TDC leaderboard submission.",
            "Five-value dispersions measure train/validation selection sensitivity on the same test set, not independent test replicates or standard errors.",
            "Public benchmark results are retrospective and do not establish prospective ADMET utility.",
            "This study does not evaluate decoder-generated candidates or reopen derivative generation.",
        ],
        "artifacts": {
            "endpoint_primary_metrics": {"path": str(primary_path), "sha256": sha256_file(primary_path)},
            "all_metrics": {"path": str(all_metrics_path), "sha256": sha256_file(all_metrics_path)},
            "model_summary": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "category_rank_summary": {"path": str(category_path), "sha256": sha256_file(category_path)},
            "coverage": {"path": str(BENCHMARK_DIR / "outputs" / "coverage.csv"), "sha256": sha256_file(BENCHMARK_DIR / "outputs" / "coverage.csv")},
            "encoding_runtime": {"path": str(timing_path), "sha256": sha256_file(timing_path)},
        },
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    summary_path = BENCHMARK_DIR / "outputs" / "tdc_admet_summary.json"
    atomic_write_json(summary_path, summary)

    checksum_targets = [
        BENCHMARK_DIR / "protocol.json",
        BENCHMARK_DIR / "inputs" / "source_manifest.json",
        BENCHMARK_DIR / "inputs" / "prepared_manifest.json",
        BENCHMARK_DIR / "inputs" / "common_manifest.json",
        BENCHMARK_DIR / "inputs" / "prior_development_overlap.json",
        BENCHMARK_DIR / "outputs" / "coverage.csv",
        primary_path,
        all_metrics_path,
        model_path,
        category_path,
        timing_path,
        summary_path,
    ]
    checksum_targets.extend(
        BENCHMARK_DIR / "outputs" / "results" / f"{model}.json"
        for model in all_models
    )
    checksum_targets.extend(
        BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}.json"
        for model in all_models
    )
    checksum_targets.extend(
        BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}.npy"
        for model in all_models
    )
    checksum_lines = []
    for path in checksum_targets:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Cannot checksum missing artifact: {path}")
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(BENCHMARK_DIR)}")
    checksum_path = BENCHMARK_DIR / "outputs" / "SHA256SUMS"
    atomic_write_text(checksum_path, "\n".join(checksum_lines) + "\n")

    complete = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": protocol["study"]["name"],
        "protocol_sha256": protocol_digest(protocol),
        "summary_sha256": sha256_file(summary_path),
        "checksums_sha256": sha256_file(checksum_path),
        "primary_models": list(primary_models),
        "diagnostic_control": "descriptor_13",
        "endpoints": len(endpoints),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "COMPLETE.json", complete)
    print(
        json.dumps(
            {
                "status": "complete",
                "gmolai": gmolai_summary,
                "best_category_balanced": model_rows[0],
                "best_selection_robust": min(
                    model_rows,
                    key=lambda row: row["selection_robust_category_balanced_rank"],
                ),
                "summary_sha256": complete["summary_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
