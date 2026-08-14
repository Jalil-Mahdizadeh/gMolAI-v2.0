#!/usr/bin/env python3
"""Evaluate one frozen representation on the common TDC ADMET panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_labels_tsv,
    sha256_file,
)


def split_key(endpoint: str, seed: int, role: str) -> str:
    return f"{endpoint}__seed{seed:02d}__{role}"


def spearman(y_true: np.ndarray, prediction: np.ndarray) -> float:
    value = spearmanr(y_true, prediction).statistic
    if not np.isfinite(value):
        raise RuntimeError("Undefined Spearman correlation")
    return float(value)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if not np.isfinite(prediction).all():
        raise RuntimeError("Regression produced non-finite predictions")
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": spearman(y_true, prediction),
    }


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    if set(y_true.astype(int)) != {0, 1} or not np.isfinite(probability).all():
        raise RuntimeError("Classification metric is undefined")
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def metric_value(
    task: str, official_metric: str, y_true: np.ndarray, prediction: np.ndarray
) -> float:
    if task == "regression":
        if official_metric == "mae":
            return float(mean_absolute_error(y_true, prediction))
        if official_metric == "spearman":
            return spearman(y_true, prediction)
    else:
        if official_metric == "roc_auc":
            return float(roc_auc_score(y_true, prediction))
        if official_metric == "pr_auc":
            return float(average_precision_score(y_true, prediction))
    raise KeyError(f"Unsupported official metric: {task}/{official_metric}")


def regression_endpoint(
    features: np.ndarray,
    targets: np.ndarray,
    identities: list[str],
    arrays: dict[str, np.ndarray],
    endpoint: str,
    metric: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    alphas = [float(value) for value in protocol["evaluation"]["regression"]["alphas"]]
    train_val = arrays[split_key(endpoint, seeds[0], "train_val")]
    test = arrays[split_key(endpoint, seeds[0], "test")]
    train_identities = {identities[int(index)] for index in train_val}
    strict_test = np.asarray(
        [index for index in test if identities[int(index)] not in train_identities],
        dtype=np.int64,
    )
    final_cache: dict[float, tuple[np.ndarray, dict[str, float], dict[str, float] | None]] = {}
    outcomes = []
    minimize = metric == "mae"
    for seed in seeds:
        train = arrays[split_key(endpoint, seed, "train")]
        validation = arrays[split_key(endpoint, seed, "valid")]
        scaler = StandardScaler().fit(features[train])
        x_train = scaler.transform(features[train])
        x_validation = scaler.transform(features[validation])
        y_mean = float(targets[train].mean())
        y_std = max(1.0e-12, float(targets[train].std(ddof=0)))
        best_alpha = None
        best_value = float("inf") if minimize else -float("inf")
        validation_grid = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(x_train, (targets[train] - y_mean) / y_std)
            prediction = model.predict(x_validation) * y_std + y_mean
            value = metric_value(
                "regression", metric, targets[validation], prediction
            )
            validation_grid.append({"alpha": alpha, "official_metric": value})
            if (minimize and value < best_value) or (not minimize and value > best_value):
                best_alpha, best_value = alpha, value
        if best_alpha is None:
            raise RuntimeError(f"No Ridge alpha selected for {endpoint} seed {seed}")
        if best_alpha not in final_cache:
            full_scaler = StandardScaler().fit(features[train_val])
            full_y_mean = float(targets[train_val].mean())
            full_y_std = max(1.0e-12, float(targets[train_val].std(ddof=0)))
            model = Ridge(alpha=best_alpha, solver="lsqr")
            model.fit(
                full_scaler.transform(features[train_val]),
                (targets[train_val] - full_y_mean) / full_y_std,
            )
            prediction = (
                model.predict(full_scaler.transform(features[test])) * full_y_std
                + full_y_mean
            )
            metrics = regression_metrics(targets[test], prediction)
            strict_metrics = None
            strict_positions = [
                position
                for position, index in enumerate(test)
                if int(index) in set(strict_test.tolist())
            ]
            if len(strict_positions) >= 2:
                strict_targets = targets[strict_test]
                if len(np.unique(strict_targets)) >= 2:
                    strict_metrics = regression_metrics(
                        strict_targets, prediction[np.asarray(strict_positions)]
                    )
            final_cache[best_alpha] = (prediction, metrics, strict_metrics)
        _, test_metrics, strict_metrics = final_cache[best_alpha]
        outcomes.append(
            {
                "seed": seed,
                "train": len(train),
                "valid": len(validation),
                "train_val_refit": len(train_val),
                "test": len(test),
                "strict_identity_disjoint_test": len(strict_test),
                "selected_alpha": best_alpha,
                "validation_official_metric": best_value,
                "validation_grid": validation_grid,
                "test_metrics": test_metrics,
                "strict_identity_disjoint_test_metrics": strict_metrics,
            }
        )
    return summarize_endpoint(outcomes, metric)


def classification_endpoint(
    features: np.ndarray,
    targets: np.ndarray,
    identities: list[str],
    arrays: dict[str, np.ndarray],
    endpoint: str,
    metric: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    labels = targets.astype(np.int64)
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    c_values = [float(value) for value in protocol["evaluation"]["classification"]["c_values"]]
    train_val = arrays[split_key(endpoint, seeds[0], "train_val")]
    test = arrays[split_key(endpoint, seeds[0], "test")]
    train_identities = {identities[int(index)] for index in train_val}
    strict_test = np.asarray(
        [index for index in test if identities[int(index)] not in train_identities],
        dtype=np.int64,
    )
    strict_test_set = set(strict_test.tolist())
    final_cache: dict[float, tuple[np.ndarray, dict[str, float], dict[str, float] | None]] = {}
    outcomes = []
    for seed in seeds:
        train = arrays[split_key(endpoint, seed, "train")]
        validation = arrays[split_key(endpoint, seed, "valid")]
        scaler = StandardScaler().fit(features[train])
        x_train = scaler.transform(features[train])
        x_validation = scaler.transform(features[validation])
        best_c = None
        best_value = -float("inf")
        validation_grid = []
        for c_value in c_values:
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=0,
            )
            model.fit(x_train, labels[train])
            probability = model.predict_proba(x_validation)[:, 1]
            value = metric_value(
                "classification", metric, labels[validation], probability
            )
            validation_grid.append({"c": c_value, "official_metric": value})
            if value > best_value:
                best_c, best_value = c_value, value
        if best_c is None:
            raise RuntimeError(f"No logistic C selected for {endpoint} seed {seed}")
        if best_c not in final_cache:
            full_scaler = StandardScaler().fit(features[train_val])
            model = LogisticRegression(
                C=best_c,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=0,
            )
            model.fit(full_scaler.transform(features[train_val]), labels[train_val])
            probability = model.predict_proba(full_scaler.transform(features[test]))[:, 1]
            metrics = classification_metrics(labels[test], probability)
            strict_metrics = None
            strict_positions = [
                position for position, index in enumerate(test) if int(index) in strict_test_set
            ]
            if strict_positions and set(labels[strict_test]) == {0, 1}:
                strict_metrics = classification_metrics(
                    labels[strict_test], probability[np.asarray(strict_positions)]
                )
            final_cache[best_c] = (probability, metrics, strict_metrics)
        _, test_metrics, strict_metrics = final_cache[best_c]
        outcomes.append(
            {
                "seed": seed,
                "train": len(train),
                "valid": len(validation),
                "train_val_refit": len(train_val),
                "test": len(test),
                "strict_identity_disjoint_test": len(strict_test),
                "selected_c": best_c,
                "validation_official_metric": best_value,
                "validation_grid": validation_grid,
                "test_metrics": test_metrics,
                "strict_identity_disjoint_test_metrics": strict_metrics,
            }
        )
    return summarize_endpoint(outcomes, metric)


def summarize_endpoint(outcomes: list[dict[str, Any]], official_metric: str) -> dict[str, Any]:
    metric_names = tuple(outcomes[0]["test_metrics"])
    summaries: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray(
            [row["test_metrics"][name] for row in outcomes], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite endpoint result for {name}")
        summaries[name] = {
            "mean": float(values.mean()),
            "population_std": float(values.std(ddof=0)),
            "values": values.tolist(),
        }
    primary = summaries[official_metric]
    rounded = np.round(np.asarray(primary["values"]), 3)
    strict_values = [
        row["strict_identity_disjoint_test_metrics"][official_metric]
        for row in outcomes
        if row["strict_identity_disjoint_test_metrics"] is not None
    ]
    strict_summary = None
    if len(strict_values) == len(outcomes):
        strict = np.asarray(strict_values, dtype=np.float64)
        strict_summary = {
            "mean": float(strict.mean()),
            "population_std": float(strict.std(ddof=0)),
            "values": strict.tolist(),
        }
    return {
        "official_metric": official_metric,
        "primary": primary,
        "tdc_three_decimal_compatibility": {
            "values": rounded.tolist(),
            "mean": float(rounded.mean()),
            "population_std": float(rounded.std(ddof=0)),
        },
        "all_metrics": summaries,
        "strict_identity_disjoint_primary": strict_summary,
        "per_seed": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    allowed = set(protocol["comparators"]["model_order"]) | {"descriptor_13"}
    if args.model not in allowed:
        raise KeyError(f"Unknown model: {args.model}")
    started = time.perf_counter()
    embedding_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{args.model}.npy"
    metadata_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{args.model}.json"
    matrix = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    common_manifest = load_json(BENCHMARK_DIR / "inputs" / "common_manifest.json")
    labels = read_labels_tsv(BENCHMARK_DIR / "inputs" / "common_labels.tsv")
    expected_dimension = (
        int(protocol["diagnostic_control"]["dimension"])
        if args.model == "descriptor_13"
        else int(protocol["comparators"]["models"][args.model]["dimension"])
    )
    if matrix.shape != (
        int(common_manifest["common_unique_identities"]),
        expected_dimension,
    ):
        raise RuntimeError(f"Unexpected embedding shape for {args.model}: {matrix.shape}")
    with np.load(
        BENCHMARK_DIR / "inputs" / "common_split_indices.npz", allow_pickle=False
    ) as source:
        arrays = {key: source[key] for key in source.files}

    endpoints: dict[str, Any] = {}
    for endpoint in protocol["data"]["endpoint_order"]:
        rows = [row for row in labels if row["endpoint"] == endpoint]
        indices = np.asarray([int(row["panel_index"]) for row in rows], dtype=np.int64)
        features = np.asarray(matrix[indices], dtype=np.float32)
        targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
        identities = [row["molecule_hash"] for row in rows]
        spec = protocol["data"]["endpoints"][endpoint]
        outcome = (
            regression_endpoint(
                features, targets, identities, arrays, endpoint, spec["metric"], protocol
            )
            if spec["task"] == "regression"
            else classification_endpoint(
                features, targets, identities, arrays, endpoint, spec["metric"], protocol
            )
        )
        endpoints[endpoint] = {
            "category": spec["category"],
            "task": spec["task"],
            "common_occurrences": len(rows),
            "common_unique_identities": len(set(identities)),
            "result": outcome,
        }
        print(
            json.dumps(
                {
                    "model": args.model,
                    "endpoint": endpoint,
                    "metric": spec["metric"],
                    "mean": outcome["primary"]["mean"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "TDC ADMET frozen-representation common-panel linear probes",
        "model": args.model,
        "dimension": expected_dimension,
        "neural_encoder_training_or_finetuning": False,
        "downstream_models_fitted": "fold-local Ridge or balanced logistic regression",
        "common_panel": {
            "unique_identities": common_manifest["common_unique_identities"],
            "occurrences": common_manifest["common_occurrences"],
            "ordered_identity_sha256": common_manifest["ordered_identity_sha256"],
            "manifest_sha256": sha256_file(
                BENCHMARK_DIR / "inputs" / "common_manifest.json"
            ),
            "split_indices_sha256": sha256_file(
                BENCHMARK_DIR / "inputs" / "common_split_indices.npz"
            ),
        },
        "embedding": {
            "path": str(embedding_path),
            "sha256": sha256_file(embedding_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
        "protocol": {
            "test": "published fixed TDC test roles intersected with common support",
            "selection": "five exact PyTDC train/validation scaffold seeds",
            "refit": "all common train_val occurrences after regularization selection",
            "dispersion": "population standard deviation across selection seeds; not a standard error",
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "endpoints": endpoints,
    }
    output = BENCHMARK_DIR / "outputs" / "results" / f"{args.model}.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "model": args.model,
                "seconds": result["runtime"]["wall_seconds"],
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
