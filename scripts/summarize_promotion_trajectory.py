#!/usr/bin/env python3
"""Audit a full Table-5 checkpoint sweep and write a publication-ready CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


STEPS = (5_000, 7_500, 10_000, 12_500, 15_000)
STEP_DIRECTORY = "step-{step:09d}"
CHECKPOINT_NAME = "checkpoints/step-{step:09d}.pt"
CALIBRATION_SOURCE = "calibration-source-raw-hybrid-100k-seed271828.pt"
CALIBRATOR = "embedding-calibrator-100k-seed271828.pt"
PROBE_RESULT = "representation-probe-standardized-raw-hybrid-w3-50k-sim5k.json"
DOWNSTREAM_RESULT = (
    "moleculenet-full-diagnostic-standardized-raw-hybrid-w3-10splits.json"
)

CANONICAL_EMBEDDING = (
    "clean_graph_z_plus_mean_node_z_train_standardized_raw_blocks"
)
STRATIFIED_SAMPLING = "deterministic_hash_bucket_stratified_without_replacement"
EXPORT_WIDE_SAMPLING = "seeded_without_replacement_across_export"
DIAGNOSTIC_FEATURES = {
    "molecule_embedding",
    "morgan_radius2_2048",
    "unit_graph_z",
    "unit_mean_node_z",
    "graph_z",
    "mean_node_z",
    "raw_graph_z_plus_mean_node_z",
}
DATASET_MINIMUMS = {
    "bace": 1_400,
    "bbbp": 1_800,
    "esol": 1_000,
    "freesolv": 600,
    "lipophilicity": 4_000,
}


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def table_rows() -> list[dict[str, Any]]:
    def probe(*keys: str) -> Callable[[dict[str, Any], dict[str, Any]], float]:
        return lambda probe_data, downstream_data: float(nested(probe_data, *keys))

    def downstream(
        dataset: str, metric: str
    ) -> Callable[[dict[str, Any], dict[str, Any]], float]:
        return lambda probe_data, downstream_data: float(
            nested(
                downstream_data,
                "datasets",
                dataset,
                "feature_results",
                "molecule_embedding",
                "summary",
                metric,
                "mean",
            )
        )

    return [
        {
            "domain": "Latent utilization",
            "metric": "Effective rank",
            "criterion": ">= 25",
            "direction": "minimum",
            "reference": lambda p, d: 25.0,
            "value": probe("embedding_diagnostics", "effective_rank"),
        },
        {
            "domain": "Held-out topology",
            "metric": "Mean R2",
            "criterion": ">= 0.90",
            "direction": "minimum",
            "reference": lambda p, d: 0.90,
            "value": probe("held_out_linear_probe", "mean_r2"),
        },
        {
            "domain": "Held-out topology",
            "metric": "Median R2",
            "criterion": ">= 0.95",
            "direction": "minimum",
            "reference": lambda p, d: 0.95,
            "value": probe("held_out_linear_probe", "median_r2"),
        },
        {
            "domain": "Held-out topology",
            "metric": "Mean standardized MAE",
            "criterion": "<= 0.15",
            "direction": "maximum",
            "reference": lambda p, d: 0.15,
            "value": probe("held_out_linear_probe", "mean_standardized_mae"),
        },
        {
            "domain": "Held-out topology",
            "metric": "Scaffold-disjoint mean R2",
            "criterion": ">= 0.95",
            "direction": "minimum",
            "reference": lambda p, d: 0.95,
            "value": probe("scaffold_disjoint_linear_probe", "mean_r2"),
        },
        {
            "domain": "Retrieval",
            "metric": "Morgan recall@10",
            "criterion": ">= 0.18",
            "direction": "minimum",
            "reference": lambda p, d: 0.18,
            "value": probe("similarity", "latent_to_morgan_recall_at_10"),
        },
        {
            "domain": "Retrieval",
            "metric": "Cosine-Tanimoto Spearman",
            "criterion": ">= 0.35",
            "direction": "minimum",
            "reference": lambda p, d: 0.35,
            "value": probe("similarity", "latent_cosine_vs_morgan_spearman"),
        },
        {
            "domain": "Retrieval",
            "metric": "Neighbor mean Tanimoto",
            "criterion": ">= 0.20",
            "direction": "minimum",
            "reference": lambda p, d: 0.20,
            "value": probe("similarity", "latent_neighbor_mean_tanimoto"),
        },
        {
            "domain": "Retrieval",
            "metric": "Neighbor-Tanimoto enrichment",
            "criterion": ">= 1.70-fold",
            "direction": "minimum",
            "reference": lambda p, d: 1.70,
            "value": probe("similarity", "neighbor_tanimoto_enrichment"),
        },
        {
            "domain": "Scaffold neighborhood",
            "metric": "Purity enrichment",
            "criterion": ">= 25-fold",
            "direction": "minimum",
            "reference": lambda p, d: 25.0,
            "value": probe("similarity", "scaffold_purity_enrichment"),
        },
        {
            "domain": "Clustering",
            "metric": "Learned ARI",
            "criterion": ">= matched Morgan ARI",
            "direction": "minimum",
            "reference": probe(
                "clustering", "morgan_spherical_kmeans", "adjusted_rand_index"
            ),
            "value": probe(
                "clustering", "latent_spherical_kmeans", "adjusted_rand_index"
            ),
        },
        {
            "domain": "Clustering",
            "metric": "Learned NMI",
            "criterion": ">= Morgan NMI - 0.03",
            "direction": "minimum",
            "reference": lambda p, d: float(
                nested(
                    p,
                    "clustering",
                    "morgan_spherical_kmeans",
                    "normalized_mutual_information",
                )
            )
            - 0.03,
            "value": probe(
                "clustering",
                "latent_spherical_kmeans",
                "normalized_mutual_information",
            ),
        },
        {
            "domain": "BACE",
            "metric": "Mean ROC-AUC",
            "criterion": ">= 0.82",
            "direction": "minimum",
            "reference": lambda p, d: 0.82,
            "value": downstream("bace", "roc_auc"),
        },
        {
            "domain": "BBBP",
            "metric": "Mean ROC-AUC",
            "criterion": ">= 0.87",
            "direction": "minimum",
            "reference": lambda p, d: 0.87,
            "value": downstream("bbbp", "roc_auc"),
        },
        {
            "domain": "ESOL",
            "metric": "Mean RMSE",
            "criterion": "<= 0.80",
            "direction": "maximum",
            "reference": lambda p, d: 0.80,
            "value": downstream("esol", "rmse"),
        },
        {
            "domain": "FreeSolv",
            "metric": "Mean RMSE",
            "criterion": "<= 1.30",
            "direction": "maximum",
            "reference": lambda p, d: 1.30,
            "value": downstream("freesolv", "rmse"),
        },
        {
            "domain": "Lipophilicity",
            "metric": "Mean RMSE",
            "criterion": "<= 0.85",
            "direction": "maximum",
            "reference": lambda p, d: 0.85,
            "value": downstream("lipophilicity", "rmse"),
        },
    ]


def add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def audit_step(
    *, step: int, run_dir: Path, sweep_root: Path, datasets_dir: Path
) -> dict[str, Any]:
    import torch

    step_dir = sweep_root / STEP_DIRECTORY.format(step=step)
    checkpoint_name = CHECKPOINT_NAME.format(step=step)
    checkpoint_path = run_dir / checkpoint_name
    calibration_source_path = step_dir / CALIBRATION_SOURCE
    calibrator_path = step_dir / CALIBRATOR
    probe_path = step_dir / PROBE_RESULT
    downstream_path = step_dir / DOWNSTREAM_RESULT
    required = (
        checkpoint_path,
        calibration_source_path,
        calibrator_path,
        probe_path,
        downstream_path,
        step_dir / "COMPLETE",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing sweep artifacts:\n" + "\n".join(missing))

    checkpoint_sha = sha256_file(checkpoint_path)
    calibration_source_sha = sha256_file(calibration_source_path)
    calibrator_sha = sha256_file(calibrator_path)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    calibrator_artifact = torch.load(
        calibrator_path, map_location="cpu", weights_only=False
    )
    calibration_metadata = calibrator_artifact.get("metadata", {})
    coordinate_mean = calibrator_artifact.get("coordinate_mean")
    coordinate_scale = calibrator_artifact.get("coordinate_scale")
    train_metadata = probe.get("train_embedding_metadata", {})
    validation_metadata = probe.get("checkpoint_metadata", {})
    downstream_metadata = downstream.get("checkpoint", {})
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "calibrator.checkpoint",
        calibration_metadata.get("checkpoint") == checkpoint_name,
        calibration_metadata.get("checkpoint"),
        checkpoint_name,
    )
    add_check(
        checks,
        "calibrator.checkpoint_sha256",
        calibration_metadata.get("checkpoint_sha256") == checkpoint_sha,
        calibration_metadata.get("checkpoint_sha256"),
        checkpoint_sha,
    )
    add_check(
        checks,
        "calibrator.global_step",
        calibration_metadata.get("global_step") == step,
        calibration_metadata.get("global_step"),
        step,
    )
    add_check(
        checks,
        "calibrator.source_sha256",
        calibration_metadata.get("source_embedding_sha256")
        == calibration_source_sha,
        calibration_metadata.get("source_embedding_sha256"),
        calibration_source_sha,
    )
    add_check(
        checks,
        "calibrator.train_only_100k_protocol",
        calibration_metadata.get("split") == "train"
        and int(calibration_metadata.get("graphs", 0)) >= 100_000
        and calibration_metadata.get("sampling") == STRATIFIED_SAMPLING
        and calibration_metadata.get("sampling_seed") == 271_828
        and int(calibration_metadata.get("sampled_source_buckets", 0)) == 256,
        {
            key: calibration_metadata.get(key)
            for key in (
                "split",
                "graphs",
                "sampling",
                "sampling_seed",
                "sampled_source_buckets",
            )
        },
        {
            "split": "train",
            "minimum_graphs": 100_000,
            "sampling": STRATIFIED_SAMPLING,
            "sampling_seed": 271_828,
            "sampled_source_buckets": 256,
        },
    )
    tensor_payload_valid = (
        isinstance(coordinate_mean, torch.Tensor)
        and isinstance(coordinate_scale, torch.Tensor)
        and coordinate_mean.shape == (384,)
        and coordinate_scale.shape == (384,)
        and bool(torch.isfinite(coordinate_mean).all())
        and bool(torch.isfinite(coordinate_scale).all())
        and float(coordinate_scale.min()) > 1.0e-8
    )
    add_check(
        checks,
        "calibrator.coordinate_payload",
        tensor_payload_valid,
        {
            "mean_shape": list(getattr(coordinate_mean, "shape", ())),
            "scale_shape": list(getattr(coordinate_scale, "shape", ())),
        },
        {"mean_shape": [384], "scale_shape": [384], "finite": True},
    )

    identity_keys = (
        "checkpoint_sha256",
        "global_step",
        "config_hash",
        "training_plan_hash",
        "graph_manifest_hash",
        "descriptor_schema_hash",
    )
    for label, metadata in (
        ("probe_train", train_metadata),
        ("probe_validation", validation_metadata),
        ("downstream", downstream_metadata),
    ):
        for key in identity_keys:
            expected = (
                checkpoint_sha
                if key == "checkpoint_sha256"
                else step
                if key == "global_step"
                else calibration_metadata.get(key)
            )
            add_check(
                checks,
                f"{label}.{key}",
                metadata.get(key) == expected,
                metadata.get(key),
                expected,
            )

    for label, metadata, expected_name in (
        ("probe_train", train_metadata, checkpoint_name),
        ("probe_validation", validation_metadata, checkpoint_name),
        ("downstream", downstream_metadata, checkpoint_name),
    ):
        name_key = "name" if label == "downstream" else "checkpoint"
        add_check(
            checks,
            f"{label}.checkpoint_name",
            metadata.get(name_key) == expected_name,
            metadata.get(name_key),
            expected_name,
        )
        add_check(
            checks,
            f"{label}.embedding_definition",
            metadata.get("embedding_definition") == CANONICAL_EMBEDDING,
            metadata.get("embedding_definition"),
            CANONICAL_EMBEDDING,
        )

    expected_parameters = {
        "graph_dimensions": 256,
        "mean_node_dimensions": 128,
        "mean_node_weight": 3.0,
        "coordinate_transform": "train_mean_and_population_std",
        "calibrator_sha256": calibrator_sha,
        "calibration_graphs": int(calibration_metadata.get("graphs", 0)),
        "calibration_sampling_seed": calibration_metadata.get("sampling_seed"),
    }
    for label, metadata in (
        ("probe_train", train_metadata),
        ("probe_validation", validation_metadata),
        ("downstream", downstream_metadata),
    ):
        add_check(
            checks,
            f"{label}.embedding_parameters",
            metadata.get("embedding_parameters") == expected_parameters,
            metadata.get("embedding_parameters"),
            expected_parameters,
        )

    for label, metadata, split, graphs in (
        ("probe_train", train_metadata, "train", 10_000),
        ("probe_validation", validation_metadata, "validation", 50_000),
    ):
        observed = {
            key: metadata.get(key)
            for key in (
                "split",
                "graphs",
                "sampling",
                "sampling_seed",
                "sampled_source_buckets",
            )
        }
        passed = (
            metadata.get("split") == split
            and int(metadata.get("graphs", 0)) >= graphs
            and metadata.get("sampling") == STRATIFIED_SAMPLING
            and metadata.get("sampling_seed") == 20_260_810
            and int(metadata.get("sampled_source_buckets", 0)) == 256
        )
        add_check(
            checks,
            f"{label}.sampling_protocol",
            passed,
            observed,
            {
                "split": split,
                "minimum_graphs": graphs,
                "sampling": STRATIFIED_SAMPLING,
                "sampling_seed": 20_260_810,
                "sampled_source_buckets": 256,
            },
        )

    protocol_expectations = {
        "probe_train_graphs": (
            nested(probe, "held_out_linear_probe", "train_graphs"),
            10_000,
        ),
        "probe_validation_graphs": (
            nested(probe, "held_out_linear_probe", "validation_graphs"),
            50_000,
        ),
        "similarity_graphs": (nested(probe, "similarity", "graphs"), 5_000),
        "similarity_available_graphs": (
            nested(probe, "similarity", "available_graphs"),
            50_000,
        ),
        "clustering_graphs": (nested(probe, "clustering", "graphs"), 10_000),
        "clustering_sampled_graphs": (
            nested(probe, "clustering", "sampled_graphs"),
            50_000,
        ),
        "clustering_kmeans_repetitions": (
            nested(probe, "clustering", "kmeans_repetitions"),
            5,
        ),
        "clustering_kmeans_n_init": (
            nested(probe, "clustering", "kmeans_n_init_per_repetition"),
            20,
        ),
    }
    for name, (observed, minimum) in protocol_expectations.items():
        add_check(
            checks,
            f"protocol.{name}",
            finite_number(observed) and float(observed) >= minimum,
            observed,
            {"minimum": minimum},
        )
    add_check(
        checks,
        "protocol.similarity_sampling",
        nested(probe, "similarity", "sampling") == EXPORT_WIDE_SAMPLING,
        nested(probe, "similarity", "sampling"),
        EXPORT_WIDE_SAMPLING,
    )
    add_check(
        checks,
        "protocol.clustering_sampling",
        nested(probe, "clustering", "sampling") == EXPORT_WIDE_SAMPLING,
        nested(probe, "clustering", "sampling"),
        EXPORT_WIDE_SAMPLING,
    )
    add_check(
        checks,
        "protocol.clustering_available",
        nested(probe, "clustering", "available") is True,
        nested(probe, "clustering", "available"),
        True,
    )
    add_check(
        checks,
        "downstream.full_diagnostic_panel",
        downstream.get("selected_only") is False,
        downstream.get("selected_only"),
        False,
    )
    add_check(
        checks,
        "downstream.embedding_dimensions",
        downstream_metadata.get("embedding_dimensions") == 384,
        downstream_metadata.get("embedding_dimensions"),
        384,
    )

    datasets = downstream.get("datasets", {})
    add_check(
        checks,
        "downstream.dataset_panel",
        set(datasets) == set(DATASET_MINIMUMS),
        sorted(datasets),
        sorted(DATASET_MINIMUMS),
    )
    for dataset_name, minimum_molecules in DATASET_MINIMUMS.items():
        dataset = datasets.get(dataset_name, {})
        primary_metric = (
            "roc_auc" if dataset_name in {"bace", "bbbp"} else "rmse"
        )
        feature_results = dataset.get("feature_results", {})
        add_check(
            checks,
            f"downstream.{dataset_name}.molecules",
            int(nested(dataset, "preparation", "molecules") or 0)
            >= minimum_molecules,
            nested(dataset, "preparation", "molecules"),
            {"minimum": minimum_molecules},
        )
        add_check(
            checks,
            f"downstream.{dataset_name}.scaffold_splits",
            int(dataset.get("scaffold_splits", 0)) >= 10,
            dataset.get("scaffold_splits"),
            {"minimum": 10},
        )
        add_check(
            checks,
            f"downstream.{dataset_name}.diagnostic_features",
            DIAGNOSTIC_FEATURES.issubset(feature_results),
            sorted(feature_results),
            sorted(DIAGNOSTIC_FEATURES),
        )
        for feature_name in DIAGNOSTIC_FEATURES:
            value = nested(
                feature_results,
                feature_name,
                "summary",
                primary_metric,
                "mean",
            )
            add_check(
                checks,
                f"downstream.{dataset_name}.{feature_name}.{primary_metric}",
                finite_number(value),
                value,
                "finite",
            )
        source = dataset.get("source", {})
        source_path = datasets_dir / str(source.get("filename", ""))
        actual_source_sha = sha256_file(source_path) if source_path.is_file() else None
        add_check(
            checks,
            f"downstream.{dataset_name}.source_sha256",
            actual_source_sha is not None
            and actual_source_sha == source.get("sha256"),
            actual_source_sha,
            source.get("sha256"),
        )

    evaluated_rows = []
    for row in table_rows():
        value = float(row["value"](probe, downstream))
        reference = float(row["reference"](probe, downstream))
        if not math.isfinite(value) or not math.isfinite(reference):
            raise ValueError(f"Non-finite Table 5 value at {row['domain']}: {row['metric']}")
        if row["direction"] == "minimum":
            margin = value - reference
            passed = value >= reference
        else:
            margin = reference - value
            passed = value <= reference
        evaluated_rows.append(
            {
                "domain": row["domain"],
                "metric": row["metric"],
                "criterion": row["criterion"],
                "direction": row["direction"],
                "value": value,
                "reference": reference,
                "margin": margin,
                "passed": passed,
            }
        )

    protocol_and_identity_passed = all(check["passed"] for check in checks)
    table5_passed = all(row["passed"] for row in evaluated_rows)
    from gmolai_retrain.representations import _validate_promotion_quality

    repository_validator = {"passed": None, "error": None}
    try:
        _validate_promotion_quality(probe, downstream)
        repository_validator["passed"] = True
    except Exception as exc:  # preserve the exact fail-closed reason
        repository_validator["passed"] = False
        repository_validator["error"] = f"{type(exc).__name__}: {exc}"
    validator_agrees = repository_validator["passed"] == table5_passed
    add_check(
        checks,
        "repository_fail_closed_validator.agrees_with_complete_table5_audit",
        validator_agrees,
        repository_validator,
        {"passed": table5_passed},
    )
    if not validator_agrees:
        protocol_and_identity_passed = False

    return {
        "step": step,
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "calibration_source": str(calibration_source_path.resolve()),
            "calibration_source_sha256": calibration_source_sha,
            "calibrator": str(calibrator_path.resolve()),
            "calibrator_sha256": calibrator_sha,
            "representation_probe": str(probe_path.resolve()),
            "representation_probe_sha256": sha256_file(probe_path),
            "downstream_benchmark": str(downstream_path.resolve()),
            "downstream_benchmark_sha256": sha256_file(downstream_path),
        },
        "checks": checks,
        "protocol_and_identity_passed": protocol_and_identity_passed,
        "table5_rows": evaluated_rows,
        "table5_passed": table5_passed,
        "repository_fail_closed_validator": repository_validator,
        "full_promotion_gate_passed": protocol_and_identity_passed
        and table5_passed
        and repository_validator.get("passed") is True,
    }


def format_number(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def write_csv(path: Path, step_results: list[dict[str, Any]]) -> None:
    rows = table_rows()
    headings = ["domain", "metric", "criterion", "direction"]
    for step in STEPS:
        prefix = f"step_{step}"
        headings.extend(
            [
                f"{prefix}_value",
                f"{prefix}_gate_reference",
                f"{prefix}_margin",
                f"{prefix}_outcome",
            ]
        )
    by_step = {result["step"]: result for result in step_results}
    output_rows: list[dict[str, Any]] = []
    for index, definition in enumerate(rows):
        output: dict[str, Any] = {
            key: definition[key]
            for key in ("domain", "metric", "criterion", "direction")
        }
        for step in STEPS:
            result = by_step[step]["table5_rows"][index]
            prefix = f"step_{step}"
            output[f"{prefix}_value"] = format_number(result["value"])
            output[f"{prefix}_gate_reference"] = format_number(result["reference"])
            output[f"{prefix}_margin"] = format_number(result["margin"])
            output[f"{prefix}_outcome"] = "Pass" if result["passed"] else "Fail"
        output_rows.append(output)

    overall: dict[str, Any] = {
        "domain": "Overall",
        "metric": "Full fail-closed promotion gate",
        "criterion": "All 17 Table 5 criteria plus protocol and artifact-integrity checks",
        "direction": "all",
    }
    for step in STEPS:
        result = by_step[step]
        passed_count = sum(row["passed"] for row in result["table5_rows"])
        prefix = f"step_{step}"
        overall[f"{prefix}_value"] = passed_count
        overall[f"{prefix}_gate_reference"] = len(rows)
        overall[f"{prefix}_margin"] = passed_count - len(rows)
        overall[f"{prefix}_outcome"] = (
            "Pass" if result["full_promotion_gate_passed"] else "Fail"
        )
    output_rows.append(overall)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headings)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    sweep_root = args.sweep_root.resolve()
    datasets_dir = args.datasets_dir.resolve()
    if not sweep_root.is_relative_to(run_dir):
        raise ValueError("Sweep root must be inside the production run directory")

    results = [
        audit_step(
            step=step,
            run_dir=run_dir,
            sweep_root=sweep_root,
            datasets_dir=datasets_dir,
        )
        for step in STEPS
    ]
    write_csv(args.output_csv.resolve(), results)
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_run": "primary seed 42",
        "protocol": {
            "calibration_graphs": 100_000,
            "calibration_split": "pretraining train",
            "calibration_sampling_seed": 271_828,
            "probe_train_graphs": 10_000,
            "probe_validation_graphs": 50_000,
            "probe_sampling_seed": 20_260_810,
            "similarity_graphs": 5_000,
            "clustering_kmeans_repetitions": 5,
            "clustering_kmeans_n_init_per_repetition": 20,
            "downstream_scaffold_splits": 10,
            "downstream_diagnostic_feature_panels": sorted(DIAGNOSTIC_FEATURES),
        },
        "steps": results,
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    for result in results:
        passed_count = sum(row["passed"] for row in result["table5_rows"])
        outcome = "PASS" if result["full_promotion_gate_passed"] else "FAIL"
        print(f"step {result['step']:>5}: {passed_count}/17 Table 5 gates; full gate {outcome}")
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"Audit: {output_json}")

    invalid = [
        result["step"]
        for result in results
        if not result["protocol_and_identity_passed"]
    ]
    if invalid:
        raise SystemExit(
            "Protocol or artifact-integrity checks failed at steps: "
            + ", ".join(str(step) for step in invalid)
        )


if __name__ == "__main__":
    main()
