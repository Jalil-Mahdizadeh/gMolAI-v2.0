#!/usr/bin/env python3
"""Create deterministic, visualization-only PCA source data for gMolAI and Morgan."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
from sklearn.decomposition import PCA

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, read_panel_tsv, sha256_file, write_csv
from metrics_common import row_l2_normalize


MODELS = ("gmolai", "morgan")


def structural(protocol):
    panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv")
    labels = sorted({row["subclass"] for row in panel})
    fit_per = int(protocol["visualization"]["structural_fit_per_subclass"])
    plot_per = int(protocol["visualization"]["structural_plot_per_subclass"])
    by_label = {
        label: sorted(
            (index for index, row in enumerate(panel) if row["subclass"] == label),
            key=lambda index: panel[index]["molecule_hash"],
        ) for label in labels
    }
    fit = np.asarray([index for label in labels for index in by_label[label][:fit_per]], dtype=np.int32)
    plot = np.asarray([index for label in labels for index in by_label[label][:plot_per]], dtype=np.int32)
    source_rows = []
    metadata = {}
    for model in MODELS:
        raw = np.load(BENCHMARK_DIR / "artifacts" / "embeddings" / "classyfire" / f"{model}.npy", mmap_mode="r", allow_pickle=False)
        fit_values = row_l2_normalize(np.asarray(raw[fit], dtype=np.float64))
        plot_values = row_l2_normalize(np.asarray(raw[plot], dtype=np.float64))
        estimator = PCA(n_components=2, svd_solver="randomized", random_state=int(protocol["visualization"]["pca_random_state"]))
        estimator.fit(fit_values)
        coordinates = estimator.transform(plot_values)
        metadata[model] = {
            "fit_rows": len(fit), "plot_rows": len(plot),
            "explained_variance_ratio": estimator.explained_variance_ratio_.tolist(),
        }
        for output_index, panel_index in enumerate(plot):
            row = panel[int(panel_index)]
            source_rows.append({
                "model": model, "panel_index": int(panel_index),
                "molecule_hash": row["molecule_hash"], "subclass": row["subclass"],
                "PC1": coordinates[output_index, 0], "PC2": coordinates[output_index, 1],
                "PC1_explained_variance_fraction": estimator.explained_variance_ratio_[0],
                "PC2_explained_variance_fraction": estimator.explained_variance_ratio_[1],
            })
    output = BENCHMARK_DIR / "outputs" / "source_data" / "figure_classyfire_pca.csv"
    write_csv(output, source_rows, tuple(source_rows[0]))
    return output, metadata


def property_space(protocol):
    panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv")
    reference = np.load(BENCHMARK_DIR / "artifacts" / "common" / "qmugs_property_reference.npz", allow_pickle=False)
    order = np.asarray(sorted(range(len(panel)), key=lambda index: panel[index]["molecule_hash"]), dtype=np.int32)
    fit = order[: int(protocol["visualization"]["property_fit_rows"])]
    plot = order[: int(protocol["visualization"]["property_plot_rows"])]
    raw_properties = np.asarray(reference["raw_properties"], dtype=np.float64)
    robust = np.asarray(reference["robust_properties"], dtype=np.float64)
    source_rows = []
    metadata = {}
    for model in MODELS:
        raw = np.load(BENCHMARK_DIR / "artifacts" / "embeddings" / "qmugs" / f"{model}.npy", mmap_mode="r", allow_pickle=False)
        fit_values = row_l2_normalize(np.asarray(raw[fit], dtype=np.float64))
        plot_values = row_l2_normalize(np.asarray(raw[plot], dtype=np.float64))
        estimator = PCA(n_components=2, svd_solver="randomized", random_state=int(protocol["visualization"]["pca_random_state"]))
        estimator.fit(fit_values)
        coordinates = estimator.transform(plot_values)
        metadata[model] = {
            "fit_rows": len(fit), "plot_rows": len(plot),
            "explained_variance_ratio": estimator.explained_variance_ratio_.tolist(),
        }
        for output_index, panel_index in enumerate(plot):
            row = panel[int(panel_index)]
            source_rows.append({
                "model": model, "panel_index": int(panel_index), "molecule_hash": row["molecule_hash"],
                "PC1": coordinates[output_index, 0], "PC2": coordinates[output_index, 1],
                "PC1_explained_variance_fraction": estimator.explained_variance_ratio_[0],
                "PC2_explained_variance_fraction": estimator.explained_variance_ratio_[1],
                "DFT_HOMO_ENERGY": raw_properties[panel_index, 0],
                "DFT_HOMO_LUMO_GAP": raw_properties[panel_index, 1],
                "log1p_DFT_DIPOLE_TOT": raw_properties[panel_index, 2],
                "robust_DFT_HOMO_ENERGY": robust[panel_index, 0],
                "robust_DFT_HOMO_LUMO_GAP": robust[panel_index, 1],
                "robust_log1p_DFT_DIPOLE_TOT": robust[panel_index, 2],
            })
    output = BENCHMARK_DIR / "outputs" / "source_data" / "figure_qmugs_pca.csv"
    write_csv(output, source_rows, tuple(source_rows[0]))
    return output, metadata


def main() -> None:
    protocol = load_protocol()
    structural_path, structural_metadata = structural(protocol)
    property_path, property_metadata = property_space(protocol)
    result = {
        "schema_version": 1, "status": "ok", "visualization_only": True,
        "used_by_metrics": False, "models": list(MODELS),
        "structural_source": str(structural_path), "structural_source_sha256": sha256_file(structural_path),
        "structural": structural_metadata, "property_source": str(property_path),
        "property_source_sha256": sha256_file(property_path), "property": property_metadata,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "pca_visualization.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

