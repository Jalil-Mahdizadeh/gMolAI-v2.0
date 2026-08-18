#!/usr/bin/env python3
"""Evaluate one frozen representation on the ClassyFire-25 endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, read_panel_tsv, sha256_file
from metrics_common import atomic_parquet, atomic_save_npz, exact_knn, row_l2_normalize


ALL_REPRESENTATIONS = (
    "gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin",
    "kermt_v2", "morgan_count", "descriptor13",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_REPRESENTATIONS)
    args = parser.parse_args()
    protocol = load_protocol()
    panel_path = BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv"
    embedding_path = BENCHMARK_DIR / "artifacts" / "embeddings" / "classyfire" / f"{args.model}.npy"
    rows = read_panel_tsv(panel_path)
    labels_text = [row["subclass"] for row in rows]
    class_names = sorted(set(labels_text))
    label_map = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([label_map[value] for value in labels_text], dtype=np.int16)
    raw = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if raw.shape[0] != len(rows):
        raise RuntimeError("Structural embedding row count differs from common panel")
    started = time.perf_counter()
    normalized = row_l2_normalize(raw)
    kmeans_spec = protocol["structural_evaluation"]["kmeans"]
    seed_results = []
    cluster_labels = []
    for seed in kmeans_spec["seeds"]:
        estimator = KMeans(
            n_clusters=int(kmeans_spec["n_clusters"]),
            init=str(kmeans_spec["init"]),
            n_init=int(kmeans_spec["n_init"]),
            max_iter=int(kmeans_spec["max_iter"]),
            tol=float(kmeans_spec["tol"]),
            algorithm=str(kmeans_spec["algorithm"]),
            random_state=int(seed),
        )
        predicted = estimator.fit_predict(normalized).astype(np.int16)
        cluster_labels.append(predicted)
        seed_results.append({
            "seed": int(seed),
            "ARI": float(adjusted_rand_score(labels, predicted)),
            "AMI": float(adjusted_mutual_info_score(labels, predicted, average_method="arithmetic")),
            "NMI": float(normalized_mutual_info_score(labels, predicted, average_method="arithmetic")),
            "inertia": float(estimator.inertia_),
            "n_iter": int(estimator.n_iter_),
        })
    cluster_matrix = np.stack(cluster_labels)
    label_path = BENCHMARK_DIR / "artifacts" / "common" / "structural_clusters" / f"{args.model}.npz"
    atomic_save_npz(
        label_path, true_labels=labels, cluster_labels=cluster_matrix,
        seeds=np.asarray(kmeans_spec["seeds"], dtype=np.int64)
    )
    k = int(protocol["structural_evaluation"]["geometry_metric"]["k"])
    neighbors = exact_knn(normalized, k, metric="normalized_euclidean")
    same = np.mean(labels[neighbors] == labels[:, None], axis=1)
    per_class = [float(np.mean(same[labels == class_index])) for class_index in range(len(class_names))]
    macro_same = float(np.mean(per_class))
    native_same = None
    native_neighbors = None
    native_metric = None
    if args.model == "morgan":
        native_metric = "binary_tanimoto"
        native_neighbors = exact_knn(np.asarray(raw, dtype=np.float64), k, metric=native_metric)
    elif args.model == "morgan_count":
        native_metric = "generalized_tanimoto_raw_counts"
        count = np.rint(np.expm1(np.asarray(raw, dtype=np.float64)))
        if np.any(count < 0):
            raise RuntimeError("Recovered count Morgan contains negative counts")
        native_neighbors = exact_knn(count, k, metric="generalized_tanimoto")
    if native_neighbors is not None:
        native_same = np.mean(labels[native_neighbors] == labels[:, None], axis=1)
    neighbor_path = BENCHMARK_DIR / "artifacts" / "common" / "structural_neighbors" / f"{args.model}.npz"
    arrays = {"normalized_euclidean": neighbors}
    if native_neighbors is not None:
        arrays["native"] = native_neighbors
    atomic_save_npz(neighbor_path, **arrays)
    frame = pd.DataFrame({
        "panel_index": np.arange(len(rows), dtype=np.int32),
        "molecule_hash": [row["molecule_hash"] for row in rows],
        "subclass": labels_text,
        "model": args.model,
        "same_subclass_at_100": same,
    })
    if native_same is not None:
        frame["native_same_subclass_at_100"] = native_same
    query_path = BENCHMARK_DIR / "outputs" / "source_data" / "structural_queries" / f"{args.model}.parquet"
    atomic_parquet(query_path, frame)
    result = {
        "schema_version": 1, "status": "ok", "model": args.model,
        "rows": len(rows), "dimension": int(raw.shape[1]),
        "input_embedding": str(embedding_path), "input_embedding_sha256": sha256_file(embedding_path),
        "preprocessing": "float64_row_l2", "kmeans": kmeans_spec,
        "seed_metrics": seed_results,
        "mean_seed_metrics": {
            metric: float(np.mean([row[metric] for row in seed_results]))
            for metric in ("ARI", "AMI", "NMI")
        },
        "macro_same_subclass_at_100": macro_same,
        "per_subclass_same_subclass_at_100": dict(zip(class_names, per_class)),
        "native_metric": native_metric,
        "native_macro_same_subclass_at_100": (
            float(np.mean([np.mean(native_same[labels == index]) for index in range(len(class_names))]))
            if native_same is not None else None
        ),
        "cluster_labels": str(label_path), "cluster_labels_sha256": sha256_file(label_path),
        "neighbors": str(neighbor_path), "neighbors_sha256": sha256_file(neighbor_path),
        "query_source_data": str(query_path), "query_source_data_sha256": sha256_file(query_path),
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = BENCHMARK_DIR / "state" / "structural" / f"{args.model}.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

