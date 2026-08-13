#!/usr/bin/env python3
"""Validate raw timings, create compact tables/plots, and seal completion."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    sha256_file,
)


SUMMARY_COLUMNS = (
    "model",
    "display_name",
    "device_class",
    "batch_size",
    "rows",
    "dimension",
    "wall_seconds",
    "rows_per_second",
    "milliseconds_per_molecule",
    "batch_latency_p50_seconds",
    "batch_latency_p95_seconds",
    "batch_latency_p99_seconds",
    "model_load_seconds_excluded",
    "worker_startup_seconds_excluded",
    "warmup_seconds_excluded",
    "peak_gpu_memory_bytes",
    "exactly_equal_to_reference",
    "within_tolerance_of_reference",
    "cross_batch_integrity_policy",
    "maximum_absolute_delta_from_reference",
    "root_mean_square_delta_from_reference",
    "relative_l2_delta_p99_from_reference",
    "maximum_relative_l2_delta_from_reference",
    "minimum_cosine_similarity_to_reference",
    "speedup_vs_gmolai_same_batch",
    "speedup_vs_morgan_same_batch",
    "host",
    "slurm_job_id",
)


def csv_text(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def validate_and_collect(
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_sizes = [int(value) for value in protocol["execution"]["batch_sizes"]]
    expected_rows = int(protocol["panel"]["rows"])
    expected_input_hash = protocol["panel"]["tsv_sha256"]
    expected_identity_hash = protocol["panel"]["ordered_identity_sha256"]
    record_only_nonconformance_models = set(
        protocol["execution"]["record_only_cross_batch_nonconformance_models"]
    )
    if not record_only_nonconformance_models.issubset(protocol["model_order"]):
        raise RuntimeError("Unknown record-only nonconformance model")
    reports: dict[str, dict[str, Any]] = {}

    for model in protocol["model_order"]:
        path = BENCHMARK_DIR / "outputs" / "raw" / f"{model}.json"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing regular raw result: {path}")
        report = load_json(path)
        specification = protocol["models"][model]
        if report.get("status") != "ok":
            raise RuntimeError(f"Incomplete raw result for {model}")
        if report.get("execution") != "inference_only_single_pass_speed_benchmark":
            raise RuntimeError(f"Unexpected execution mode for {model}")
        if report.get("training_performed") is not False:
            raise RuntimeError(f"Training flag is not false for {model}")
        if report.get("model_weights_modified") is not False:
            raise RuntimeError(f"Model-weight flag is not false for {model}")
        if report.get("scientific_embedding_artifact_written") is not False:
            raise RuntimeError(f"Scientific-embedding flag is not false for {model}")
        if report.get("model") != model:
            raise RuntimeError(f"Raw result model mismatch for {model}")
        if int(report.get("rows", -1)) != expected_rows:
            raise RuntimeError(f"Raw result row mismatch for {model}")
        if int(report.get("dimension", -1)) != int(specification["dimension"]):
            raise RuntimeError(f"Raw result dimension mismatch for {model}")
        if report.get("input_sha256") != expected_input_hash:
            raise RuntimeError(f"Raw input hash mismatch for {model}")
        if report.get("ordered_identity_sha256") != expected_identity_hash:
            raise RuntimeError(f"Raw identity hash mismatch for {model}")
        if report.get("condition_order") != batch_sizes:
            raise RuntimeError(f"Raw batch order mismatch for {model}")
        repeatability = report.get("fixed_batch_repeatability_qualification")
        if not isinstance(repeatability, dict) or repeatability.get("passed") is not True:
            raise RuntimeError(f"Fixed-batch repeatability failed for {model}")
        conditions = report.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != len(batch_sizes):
            raise RuntimeError(f"Raw condition count mismatch for {model}")
        for expected_batch, condition in zip(batch_sizes, conditions, strict=True):
            if int(condition.get("batch_size", -1)) != expected_batch:
                raise RuntimeError(f"Condition batch mismatch for {model}")
            if int(condition.get("measured_passes", -1)) != 1:
                raise RuntimeError(f"Measured-pass count mismatch for {model}")
            if int(condition.get("warmup_batches", -1)) != 1:
                raise RuntimeError(f"Warm-up count mismatch for {model}")
            if (
                condition.get("within_tolerance_of_reference") is not True
                and model not in record_only_nonconformance_models
            ):
                raise RuntimeError(f"Cross-batch integrity failed for {model}")
            for field in (
                "wall_seconds",
                "rows_per_second",
                "milliseconds_per_molecule",
            ):
                value = float(condition[field])
                if not math.isfinite(value) or value <= 0.0:
                    raise RuntimeError(f"Invalid {field} for {model}/{expected_batch}")
            latencies = condition.get("batch_latency_seconds")
            for field in (
                "batch_latency_p50_seconds",
                "batch_latency_p95_seconds",
                "batch_latency_p99_seconds",
            ):
                value = condition[field]
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                    raise RuntimeError(f"Invalid {field} for {model}/{expected_batch}")
            if not isinstance(latencies, list) or len(latencies) != int(
                condition["batch_count"]
            ):
                raise RuntimeError(f"Batch-latency record mismatch for {model}")
        observed_integrity = bool(
            all(
                condition["within_tolerance_of_reference"]
                for condition in conditions
            )
        )
        if report.get("cross_batch_integrity_passed") is not observed_integrity:
            raise RuntimeError(f"Cross-batch integrity summary mismatch for {model}")
        expected_policy = (
            "record_only_known_native_nonconformance"
            if model in record_only_nonconformance_models
            else "fail_closed"
        )
        if report.get("cross_batch_integrity_policy") != expected_policy:
            raise RuntimeError(f"Cross-batch integrity policy mismatch for {model}")
        if model in record_only_nonconformance_models and observed_integrity:
            raise RuntimeError(
                f"Expected native-path nonconformance was not reproduced for {model}"
            )
        reports[model] = report

    hosts = {report["host"] for report in reports.values()}
    jobs = {report["slurm_job_id"] for report in reports.values()}
    if len(hosts) != 1 or len(jobs) != 1 or None in jobs:
        raise RuntimeError(
            f"All models must originate from one host/job: hosts={hosts}, jobs={jobs}"
        )
    gpu_names = {
        report["gpu_name"]
        for model, report in reports.items()
        if protocol["models"][model]["device"] == "gpu"
    }
    if len(gpu_names) != 1 or None in gpu_names:
        raise RuntimeError(f"Neural GPU identity is inconsistent: {gpu_names}")

    throughput = {
        (model, int(condition["batch_size"])): float(condition["rows_per_second"])
        for model, report in reports.items()
        for condition in report["conditions"]
    }
    summary_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for model in protocol["model_order"]:
        report = reports[model]
        for condition in report["conditions"]:
            batch = int(condition["batch_size"])
            rows_per_second = float(condition["rows_per_second"])
            summary_rows.append(
                {
                    "model": model,
                    "display_name": report["display_name"],
                    "device_class": report["device_class"],
                    "batch_size": batch,
                    "rows": report["rows"],
                    "dimension": report["dimension"],
                    "wall_seconds": condition["wall_seconds"],
                    "rows_per_second": rows_per_second,
                    "milliseconds_per_molecule": condition["milliseconds_per_molecule"],
                    "batch_latency_p50_seconds": condition["batch_latency_p50_seconds"],
                    "batch_latency_p95_seconds": condition["batch_latency_p95_seconds"],
                    "batch_latency_p99_seconds": condition["batch_latency_p99_seconds"],
                    "model_load_seconds_excluded": report[
                        "model_load_seconds_excluded_from_primary"
                    ],
                    "worker_startup_seconds_excluded": report[
                        "worker_startup_seconds_excluded_from_primary"
                    ],
                    "warmup_seconds_excluded": condition["warmup_seconds"],
                    "peak_gpu_memory_bytes": condition["peak_gpu_memory_bytes"],
                    "exactly_equal_to_reference": condition["exactly_equal_to_reference"],
                    "within_tolerance_of_reference": condition[
                        "within_tolerance_of_reference"
                    ],
                    "cross_batch_integrity_policy": report[
                        "cross_batch_integrity_policy"
                    ],
                    "maximum_absolute_delta_from_reference": condition[
                        "maximum_absolute_delta_from_reference"
                    ],
                    "root_mean_square_delta_from_reference": condition[
                        "root_mean_square_delta_from_reference"
                    ],
                    "relative_l2_delta_p99_from_reference": condition[
                        "relative_l2_delta_p99_from_reference"
                    ],
                    "maximum_relative_l2_delta_from_reference": condition[
                        "maximum_relative_l2_delta_from_reference"
                    ],
                    "minimum_cosine_similarity_to_reference": condition[
                        "minimum_cosine_similarity_to_reference"
                    ],
                    "speedup_vs_gmolai_same_batch": rows_per_second
                    / throughput[("gmolai", batch)],
                    "speedup_vs_morgan_same_batch": rows_per_second
                    / throughput[("morgan", batch)],
                    "host": report["host"],
                    "slurm_job_id": report["slurm_job_id"],
                }
            )
            for index, latency in enumerate(condition["batch_latency_seconds"]):
                start = index * batch
                latency_rows.append(
                    {
                        "model": model,
                        "display_name": report["display_name"],
                        "device_class": report["device_class"],
                        "batch_size": batch,
                        "batch_index": index,
                        "batch_rows": min(batch, expected_rows - start),
                        "latency_seconds": latency,
                    }
                )

    metadata = {
        "host": next(iter(hosts)),
        "slurm_job_id": next(iter(jobs)),
        "gpu_name": next(iter(gpu_names)),
        "raw_reports": reports,
    }
    return summary_rows, latency_rows, metadata


def make_plot(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    colors = {
        "gmolai": "#0072B2",
        "morgan": "#222222",
        "molai": "#D55E00",
        "molformer": "#009E73",
        "smi_ted": "#CC79A7",
        "molclr_gin": "#E69F00",
        "kermt_v2": "#56B4E9",
    }
    by_model = {
        model: [row for row in rows if row["model"] == model]
        for model in protocol["model_order"]
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.3, 5.2))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.24)
    for model in protocol["model_order"]:
        model_rows = sorted(by_model[model], key=lambda value: value["batch_size"])
        axis.plot(
            [row["batch_size"] for row in model_rows],
            [row["rows_per_second"] for row in model_rows],
            marker="X" if model == "kermt_v2" else "o",
            markersize=6 if model == "kermt_v2" else 5,
            linewidth=1.8,
            linestyle=(
                ":" if model == "kermt_v2"
                else "--" if model == "morgan"
                else "-"
            ),
            color=colors[model],
            label=(
                model_rows[0]["display_name"] + "†"
                if model == "kermt_v2"
                else model_rows[0]["display_name"]
            ),
        )
    axis.set_yscale("log")
    axis.set_xticks(protocol["execution"]["batch_sizes"])
    axis.set_xlabel("Batch size (molecules)")
    axis.set_ylabel("Throughput (molecules s$^{-1}$; log scale)")
    axis.set_title("Single-pass encoding throughput on the common locked-test panel")
    axis.grid(True, which="major", axis="both", color="#D9D9D9", linewidth=0.7)
    axis.grid(True, which="minor", axis="y", color="#EFEFEF", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=2, frameon=False, loc="best")
    figure.text(
        0.5,
        0.005,
        "49,844 common molecules; one pass; Morgan is CPU-only; no error bars.\n"
        "† KERMT v2 failed output equivalence across batch sizes; compute throughput only.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    for suffix, options in (
        ("png", {"dpi": 400}),
        ("pdf", {"metadata": {"Creator": "gMolAI speed benchmark"}}),
        ("svg", {"metadata": {"Creator": "gMolAI speed benchmark"}}),
    ):
        destination = BENCHMARK_DIR / "outputs" / f"throughput_by_batch_size.{suffix}"
        temporary = destination.with_name(destination.name + ".partial")
        figure.savefig(temporary, format=suffix, bbox_inches="tight", **options)
        temporary.replace(destination)
    plt.close(figure)


def make_results_markdown(
    rows: list[dict[str, Any]], protocol: dict[str, Any], metadata: dict[str, Any]
) -> str:
    lookup = {
        (row["model"], int(row["batch_size"])): row for row in rows
    }
    lines = [
        "# Encoding-speed benchmark results",
        "",
        f"- **Panel:** {protocol['panel']['rows']:,} all-model common locked-test molecules",
        f"- **Host:** {metadata['host']}",
        f"- **GPU:** {metadata['gpu_name']} (neural encoders only)",
        f"- **Slurm job:** {metadata['slurm_job_id']}",
        f"- **CPU allowance:** {protocol['execution']['cpus_per_task']} CPUs; gMolAI uses {protocol['execution']['cpu_workers']} RDKit workers",
        "- **Measurement:** one complete pass after one warm-up batch; no confidence intervals",
        "",
        "| Encoder | Device | Batch 64 (mol/s) | Batch 128 (mol/s) | Batch 256 (mol/s) | Batch 512 (mol/s) | Output-equivalent? |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    batches = tuple(int(value) for value in protocol["execution"]["batch_sizes"])
    for model in protocol["model_order"]:
        specification = protocol["models"][model]
        values = [lookup[(model, batch)]["rows_per_second"] for batch in batches]
        equivalent = all(
            lookup[(model, batch)]["within_tolerance_of_reference"]
            for batch in batches
        )
        display_name = (
            specification["display_name"] + "†"
            if model == "kermt_v2"
            else specification["display_name"]
        )
        lines.append(
            f"| {display_name} | {specification['device'].upper()} | "
            f"{values[0]:,.2f} | {values[1]:,.2f} | {values[2]:,.2f} | {values[3]:,.2f} | "
            f"{'Yes' if equivalent else 'No'} |"
        )
    kermt_conditions = metadata["raw_reports"]["kermt_v2"]["conditions"]
    kermt_minimum_cosine = min(
        float(condition["minimum_cosine_similarity_to_reference"])
        for condition in kermt_conditions
    )
    kermt_maximum_relative_l2 = max(
        float(condition["maximum_relative_l2_delta_from_reference"])
        for condition in kermt_conditions
    )
    lines.extend(
        [
            "",
            "The timer covers canonical SMILES already in RAM through complete ordered FP32 vectors in host RAM. It includes each model's required preprocessing and device transfers but excludes SIF/model loading, worker startup, warm-up, validation, hashing and disk serialization.",
            "",
            (
                "† Native KERMT v2 failed the frozen cross-batch output-equivalence "
                f"gate (minimum cosine {kermt_minimum_cosine:.6f}; maximum "
                f"relative-L2 delta {kermt_maximum_relative_l2:.6f}). Its "
                "batch-size points describe computational throughput for batch-dependent "
                "outputs, not scaling of one invariant molecular representation."
            ),
            "",
            "These are descriptive single-pass point measurements. Small differences must not be presented as established speed superiority, and Morgan's CPU result must not be described as GPU-forward performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    protocol = load_protocol()
    record_only_nonconformance_models = set(
        protocol["execution"]["record_only_cross_batch_nonconformance_models"]
    )
    summary_rows, latency_rows, metadata = validate_and_collect(protocol)
    outputs = BENCHMARK_DIR / "outputs"
    atomic_write_text(
        outputs / "speed_results.csv", csv_text(SUMMARY_COLUMNS, summary_rows)
    )
    latency_columns = (
        "model",
        "display_name",
        "device_class",
        "batch_size",
        "batch_index",
        "batch_rows",
        "latency_seconds",
    )
    atomic_write_text(
        outputs / "batch_latencies.csv", csv_text(latency_columns, latency_rows)
    )
    compact = {
        "schema_version": 1,
        "status": "complete_with_declared_kermt_cross_batch_nonconformance",
        "execution": "inference_only_single_pass_speed_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "panel": protocol["panel"],
        "timing": protocol["timing"],
        "execution_conditions": protocol["execution"],
        "interpretation": protocol["interpretation"],
        "host": metadata["host"],
        "gpu_name": metadata["gpu_name"],
        "slurm_job_id": metadata["slurm_job_id"],
        "results": summary_rows,
    }
    atomic_write_json(outputs / "speed_results.json", compact)
    make_plot(summary_rows, protocol)
    atomic_write_text(
        BENCHMARK_DIR / "RESULTS.md",
        make_results_markdown(summary_rows, protocol, metadata),
    )

    checksum_paths: list[Path] = []
    for directory_name in ("inputs", "outputs", "state"):
        directory = BENCHMARK_DIR / directory_name
        for path in directory.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name in {".gitkeep", "SHA256SUMS", "COMPLETE.json", "status.json"}:
                continue
            if (BENCHMARK_DIR / "outputs" / "failed") in path.parents:
                continue
            if path.name.endswith(".partial"):
                raise RuntimeError(f"Incomplete temporary artifact remains: {path}")
            checksum_paths.append(path)
    checksum_paths.append(BENCHMARK_DIR / "RESULTS.md")
    checksum_paths = sorted(
        set(checksum_paths), key=lambda value: str(value.relative_to(BENCHMARK_DIR))
    )
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.relative_to(BENCHMARK_DIR)}\n"
        for path in checksum_paths
    )
    checksum_path = outputs / "SHA256SUMS"
    atomic_write_text(checksum_path, checksum_text)

    complete = {
        "schema_version": 1,
        "status": "complete_with_declared_kermt_cross_batch_nonconformance",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "inference_only_single_pass_speed_benchmark",
        "training_performed": False,
        "model_weights_modified": False,
        "gmolai_checkpoint_or_calibrator_modified": False,
        "scientific_embedding_artifacts_written": False,
        "panel_rows": protocol["panel"]["rows"],
        "batch_sizes": protocol["execution"]["batch_sizes"],
        "measured_passes_per_condition": 1,
        "models": protocol["model_order"],
        "known_cross_batch_nonconformance_models": sorted(
            record_only_nonconformance_models
        ),
        "all_other_models_cross_batch_integrity_passed": True,
        "host": metadata["host"],
        "gpu_name": metadata["gpu_name"],
        "slurm_job_id": metadata["slurm_job_id"],
        "protocol_sha256": sha256_file(BENCHMARK_DIR / "protocol.json"),
        "summary_sha256": sha256_file(outputs / "speed_results.json"),
        "sha256sums_sha256": sha256_file(checksum_path),
        "checksummed_files": len(checksum_paths),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "COMPLETE.json", complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()

