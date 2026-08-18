#!/usr/bin/env python3
"""Evaluate one frozen representation on the QMugs property-neighborhood endpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, read_panel_tsv, sha256_file
from metrics_common import atomic_parquet, atomic_save_npz, exact_knn, row_l2_normalize


ALL_REPRESENTATIONS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2", "morgan_count", "descriptor13")


def row_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float64)
    for index, (first, second) in enumerate(zip(left, right)):
        result[index] = len(set(map(int, first)).intersection(map(int, second))) / first.size
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_REPRESENTATIONS)
    args = parser.parse_args()
    protocol = load_protocol()
    panel_path = BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv"
    rows = read_panel_tsv(panel_path)
    embedding_path = BENCHMARK_DIR / "artifacts" / "embeddings" / "qmugs" / f"{args.model}.npy"
    raw_embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if raw_embedding.shape[0] != len(rows):
        raise RuntimeError("QMugs embedding row count differs from common panel")
    reference_path = BENCHMARK_DIR / "artifacts" / "common" / "qmugs_property_reference.npz"
    reference = np.load(reference_path, allow_pickle=False)
    properties = np.asarray(reference["robust_properties"], dtype=np.float64)
    property_neighbors = np.asarray(reference["property_neighbors"], dtype=np.int32)
    strata = np.asarray(reference["heavy_atom_decile"], dtype=np.int8)
    null = np.asarray(reference["null_mean_distance"], dtype=np.float64)
    started = time.perf_counter()
    normalized = row_l2_normalize(raw_embedding)
    k = int(protocol["property_evaluation"]["k"])
    neighbors = exact_knn(normalized, k, metric="normalized_euclidean")
    differences = properties[neighbors] - properties[:, None, :]
    property_distances = np.linalg.norm(differences, axis=2)
    query_neighbor_distance = np.mean(property_distances, axis=1)
    npd = query_neighbor_distance / null
    recall = row_overlap(neighbors, property_neighbors)
    deviations = np.mean(np.abs(differences), axis=1)
    if not np.isfinite(npd).all() or not np.isfinite(recall).all() or not np.isfinite(deviations).all():
        raise RuntimeError("Nonfinite property-neighborhood metric")
    neighbor_path = BENCHMARK_DIR / "artifacts" / "common" / "property_neighbors" / f"{args.model}.npz"
    atomic_save_npz(neighbor_path, normalized_euclidean=neighbors)
    property_names = [str(value) for value in reference["property_names"]]
    frame = pd.DataFrame({
        "panel_index": np.arange(len(rows), dtype=np.int32),
        "molecule_hash": [row["molecule_hash"] for row in rows],
        "heavy_atom_count": np.asarray(reference["heavy_atom_count"], dtype=np.int16),
        "heavy_atom_decile": strata + 1,
        "model": args.model,
        "NPD_at_100": npd,
        "property_neighbor_recall_at_100": recall,
        **{f"deviation_{name}": deviations[:, index] for index, name in enumerate(property_names)},
    })
    query_path = BENCHMARK_DIR / "outputs" / "source_data" / "property_queries" / f"{args.model}.parquet"
    atomic_parquet(query_path, frame)
    per_property = {property_names[index]: float(np.median(deviations[:, index])) for index in range(3)}
    deciles = []
    for decile in range(10):
        mask = strata == decile
        deciles.append({
            "heavy_atom_decile": decile + 1, "rows": int(np.count_nonzero(mask)),
            "NPD_at_100": float(np.mean(npd[mask])),
            "property_neighbor_recall_at_100": float(np.mean(recall[mask])),
            **{f"deviation_{name}": float(np.median(deviations[mask, index])) for index, name in enumerate(property_names)},
        })
    result = {
        "schema_version": 1, "status": "ok", "model": args.model,
        "rows": len(rows), "dimension": int(raw_embedding.shape[1]),
        "embedding_sha256": sha256_file(embedding_path),
        "property_reference_sha256": sha256_file(reference_path),
        "preprocessing": "float64_row_l2", "k": k,
        "NPD_at_100": float(np.mean(npd)),
        "property_neighbor_recall_at_100": float(np.mean(recall)),
        "per_property_median_absolute_neighbor_deviation": per_property,
        "heavy_atom_deciles": deciles,
        "neighbors": str(neighbor_path), "neighbors_sha256": sha256_file(neighbor_path),
        "query_source_data": str(query_path), "query_source_data_sha256": sha256_file(query_path),
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "property" / f"{args.model}.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

