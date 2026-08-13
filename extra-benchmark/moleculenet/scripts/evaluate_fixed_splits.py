#!/usr/bin/env python3
"""Evaluate one frozen representation on inherited common-panel split roles."""

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
from prepare_datasets import array_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def regression_probe(
    features: np.ndarray,
    targets: np.ndarray,
    splits: dict[str, np.ndarray],
    dataset: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    results = []
    for outer, split_seed in enumerate(seeds):
        train = splits[array_key(dataset, outer, "train")]
        test = splits[array_key(dataset, outer, "test")]
        best_alpha, best_score = None, float("inf")
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            fold_scores = []
            for fold in range(3):
                fit = splits[array_key(dataset, outer, "fit", fold)]
                validation = splits[array_key(dataset, outer, "validation", fold)]
                x_scaler = StandardScaler().fit(features[fit])
                y_mean = float(targets[fit].mean())
                y_std = max(1.0e-12, float(targets[fit].std()))
                model = Ridge(alpha=alpha, solver="lsqr")
                model.fit(
                    x_scaler.transform(features[fit]),
                    (targets[fit] - y_mean) / y_std,
                )
                prediction = (
                    model.predict(x_scaler.transform(features[validation]))
                    * y_std
                    + y_mean
                )
                fold_scores.append(
                    float(mean_squared_error(targets[validation], prediction) ** 0.5)
                )
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_alpha, best_score = alpha, score

        x_scaler = StandardScaler().fit(features[train])
        y_mean = float(targets[train].mean())
        y_std = max(1.0e-12, float(targets[train].std()))
        model = Ridge(alpha=float(best_alpha), solver="lsqr")
        model.fit(
            x_scaler.transform(features[train]),
            (targets[train] - y_mean) / y_std,
        )
        prediction = model.predict(x_scaler.transform(features[test])) * y_std + y_mean
        correlation = spearmanr(targets[test], prediction).statistic
        rmse = float(mean_squared_error(targets[test], prediction) ** 0.5)
        results.append(
            {
                "train": int(len(train)),
                "test": int(len(test)),
                "inner_scaffold_folds": 3,
                "outer_seed": int(split_seed),
                "ridge_alpha": float(best_alpha),
                "inner_mean_rmse": best_score,
                "rmse": rmse,
                "normalized_rmse": rmse / y_std,
                "mae": float(mean_absolute_error(targets[test], prediction)),
                "r2": float(r2_score(targets[test], prediction)),
                "spearman": float(correlation) if np.isfinite(correlation) else None,
            }
        )
    return results


def classification_probe(
    features: np.ndarray,
    targets: np.ndarray,
    splits: dict[str, np.ndarray],
    dataset: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    labels = targets.astype(np.int64)
    results = []
    for outer, split_seed in enumerate(seeds):
        train = splits[array_key(dataset, outer, "train")]
        test = splits[array_key(dataset, outer, "test")]
        best_c, best_score = None, -float("inf")
        for c_value in (0.01, 0.1, 1.0, 10.0):
            fold_scores = []
            for fold in range(3):
                fit = splits[array_key(dataset, outer, "fit", fold)]
                validation = splits[array_key(dataset, outer, "validation", fold)]
                x_scaler = StandardScaler().fit(features[fit])
                model = LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=0,
                )
                model.fit(x_scaler.transform(features[fit]), labels[fit])
                probability = model.predict_proba(
                    x_scaler.transform(features[validation])
                )[:, 1]
                fold_scores.append(
                    float(roc_auc_score(labels[validation], probability))
                )
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_c, best_score = c_value, score

        x_scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=float(best_c),
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=0,
        )
        model.fit(x_scaler.transform(features[train]), labels[train])
        probability = model.predict_proba(x_scaler.transform(features[test]))[:, 1]
        prediction = (probability >= 0.5).astype(np.int64)
        results.append(
            {
                "train": int(len(train)),
                "test": int(len(test)),
                "inner_scaffold_folds": 3,
                "outer_seed": int(split_seed),
                "logistic_c": float(best_c),
                "inner_mean_roc_auc": best_score,
                "test_positive_fraction": float(labels[test].mean()),
                "roc_auc": float(roc_auc_score(labels[test], probability)),
                "average_precision": float(
                    average_precision_score(labels[test], probability)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test], prediction)
                ),
            }
        )
    return results


def summarize(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    metrics = (
        ("rmse", "normalized_rmse", "mae", "r2", "spearman")
        if task == "regression"
        else ("roc_auc", "average_precision", "balanced_accuracy")
    )
    summary = {}
    for metric in metrics:
        values = np.asarray(
            [row[metric] for row in rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        if len(values) != len(rows) or not np.isfinite(values).all():
            raise RuntimeError(f"Incomplete or non-finite {metric} results")
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "values": values.tolist(),
        }
    return {"summary": summary, "per_split": rows}


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if args.model not in protocol["comparators"]["models"]:
        raise KeyError(f"Unknown model: {args.model}")
    started = time.perf_counter()
    embedding_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{args.model}.npy"
    metadata_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{args.model}.json"
    matrix = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    common_manifest = load_json(BENCHMARK_DIR / "inputs" / "common_manifest.json")
    labels = read_labels_tsv(BENCHMARK_DIR / "inputs" / "common_labels.tsv")
    if matrix.shape != (
        len(labels),
        int(protocol["comparators"]["models"][args.model]["dimension"]),
    ):
        raise RuntimeError(f"Unexpected matrix shape for {args.model}: {matrix.shape}")
    with np.load(
        BENCHMARK_DIR / "inputs" / "common_split_indices.npz", allow_pickle=False
    ) as source:
        split_arrays = {key: source[key] for key in source.files}
    full_manifest = load_json(BENCHMARK_DIR / "inputs" / "dataset_manifest.json")

    datasets: dict[str, Any] = {}
    for dataset in protocol["datasets"]["order"]:
        global_indices = np.asarray(
            [index for index, row in enumerate(labels) if row["dataset"] == dataset],
            dtype=np.int64,
        )
        dataset_rows = [labels[int(index)] for index in global_indices]
        features = np.asarray(matrix[global_indices], dtype=np.float32)
        targets = np.asarray([float(row["target"]) for row in dataset_rows])
        task = dataset_rows[0]["task"]
        seeds = [
            int(row["outer_seed"])
            for row in full_manifest["datasets"][dataset]["split_identity_manifest"]
        ]
        if len(seeds) != int(protocol["evaluation"]["outer_splits"]):
            raise RuntimeError(f"{dataset} split count changed")
        per_split = (
            regression_probe(features, targets, split_arrays, dataset, seeds)
            if task == "regression"
            else classification_probe(features, targets, split_arrays, dataset, seeds)
        )
        datasets[dataset] = {
            "task": task,
            "molecules": len(dataset_rows),
            "scaffold_groups": len(
                set(row["scaffold_group"] for row in dataset_rows)
            ),
            "common_identity_set_sha256": common_manifest["datasets"][dataset][
                "common_identity_set_sha256"
            ],
            "split_assignment_rule": (
                "intersection with the frozen full-panel outer/inner identities; "
                "no molecule was reassigned"
            ),
            "feature_results": summarize(per_split, task),
        }

    result = {
        "schema_version": 1,
        "benchmark": "Frozen encoder MoleculeNet/HIV common-panel linear probes",
        "status": "complete",
        "model": args.model,
        "dimension": int(matrix.shape[1]),
        "neural_encoder_training_or_finetuning": False,
        "downstream_models_fitted": "fold-local Ridge or logistic regression",
        "common_panel": {
            "rows": len(labels),
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
            "outer": "ten frozen GroupShuffleSplit scaffold-group assignments",
            "inner": "three frozen grouped folds inherited by identity",
            "scaling": "StandardScaler fit independently on each fit/train fold",
            "regression": "Ridge; alpha in [0.1,1,10,100,1000]; inner RMSE",
            "classification": (
                "balanced liblinear logistic regression; C in [0.01,0.1,1,10]; "
                "inner ROC-AUC"
            ),
            "dispersion": "population standard deviation across ten overlapping outer splits",
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
        "datasets": datasets,
    }
    output = BENCHMARK_DIR / "outputs" / "results" / f"{args.model}.json"
    atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "model": args.model,
                "seconds": result["runtime"]["wall_seconds"],
                "primary": {
                    name: value["feature_results"]["summary"][
                        "rmse" if value["task"] == "regression" else "roc_auc"
                    ]["mean"]
                    for name, value in datasets.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
