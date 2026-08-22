#!/usr/bin/env python3
"""Render target-balanced macro ROC curves from the frozen five-shot scores."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_csv,
    read_panel_tsv,
    read_tsv,
    sha256_file,
    write_csv,
)
from evaluate import load_anchor_schedule, similarity_to_unique_anchors
from make_figures import COLORS, configure, save_figure
from metrics import candidate_mask


FPR_GRID_POINTS = 10_001
MAX_AUC_APPROXIMATION_ERROR = 2.0e-4


def interpolated_roc(
    scores: np.ndarray, labels: np.ndarray, fpr_grid: np.ndarray
) -> np.ndarray:
    """Interpolate a standard ROC path onto a common FPR grid."""
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    values = np.interp(fpr_grid, fpr, tpr)
    values[0] = 0.0
    values[-1] = 1.0
    if np.any(np.diff(values) < -1.0e-12):
        raise RuntimeError("Interpolated ROC curve is not monotone")
    return np.clip(values, 0.0, 1.0)


def main() -> None:
    protocol = load_protocol()
    models = tuple(protocol["models"]["primary_order"])
    primary_shots = int(protocol["retrieval"]["primary_shots"])
    draws = int(protocol["retrieval"]["draws_per_target"])
    population_path = BENCHMARK_DIR / "state/POPULATION_FROZEN.json"
    anchors_state_path = BENCHMARK_DIR / "state/ANCHORS_FROZEN.json"
    retrieval_state_path = BENCHMARK_DIR / "state/RETRIEVAL_COMPLETE.json"
    summary_state_path = BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json"
    population = load_json(population_path)
    anchors_state = load_json(anchors_state_path)
    retrieval_state = load_json(retrieval_state_path)
    summary_state = load_json(summary_state_path)
    if population.get("status") != "frozen" or anchors_state.get("status") != "frozen":
        raise RuntimeError("Population and anchors must be frozen")
    if retrieval_state.get("status") != "ok" or summary_state.get("status") != "ok":
        raise RuntimeError("Frozen retrieval and target summaries are required")

    anchor_path = BENCHMARK_DIR / "results/tables/anchor_draws.csv"
    common_panel_path = BENCHMARK_DIR / "inputs/prepared/common_panel.tsv"
    membership_path = BENCHMARK_DIR / "inputs/prepared/common_memberships.tsv"
    index_maps_path = BENCHMARK_DIR / "state/model_index_maps.npz"
    if anchors_state.get("anchor_draws_sha256") != sha256_file(anchor_path):
        raise RuntimeError("Frozen anchor schedule changed")
    if population.get("common_panel_sha256") != sha256_file(common_panel_path):
        raise RuntimeError("Frozen common panel changed")
    if population.get("common_memberships_sha256") != sha256_file(membership_path):
        raise RuntimeError("Frozen memberships changed")
    if population.get("model_index_maps_sha256") != sha256_file(index_maps_path):
        raise RuntimeError("Frozen representation index maps changed")

    schedule = load_anchor_schedule(anchor_path)
    panel = read_panel_tsv(common_panel_path)
    global_index = {row["molecule_hash"]: int(row["panel_index"]) for row in panel}
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(membership_path):
        by_target[row["target_id"]].append(row)
    fpr_grid = np.linspace(0.0, 1.0, FPR_GRID_POINTS, dtype=np.float64)

    summary_table_path = BENCHMARK_DIR / "results/tables/model_summary.csv"
    if summary_state.get("model_summary_sha256") != sha256_file(summary_table_path):
        raise RuntimeError("Frozen model summary changed")
    exact_auc = {
        row["model"]: float(row["target_level_mean"])
        for row in read_csv(summary_table_path)
        if int(row["shots"]) == primary_shots
        and row["condition"] == "standard"
        and row["metric"] == "roc_auc"
        and row["model"] in models
    }
    if set(exact_auc) != set(models):
        raise RuntimeError("Primary target-level ROC-AUC summary is incomplete")

    macro_curves: dict[str, np.ndarray] = {}
    target_counts: dict[str, int] = {}
    maps_file = np.load(index_maps_path, allow_pickle=False)
    try:
        for model in models:
            embedding_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{model}.npy"
            if population["models"][model]["embedding_sha256"] != sha256_file(embedding_path):
                raise RuntimeError(f"Frozen representation changed for {model}")
            embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            common_to_model = np.asarray(maps_file[model], dtype=np.int64)
            if common_to_model.shape != (len(panel),):
                raise RuntimeError(f"Invalid common index map for {model}")
            per_target_curves = []
            for target_id in sorted(by_target):
                rows = sorted(by_target[target_id], key=lambda row: row["molecule_hash"])
                identities = tuple(row["molecule_hash"] for row in rows)
                labels = np.asarray([row["label"] == "active" for row in rows], dtype=np.int8)
                scaffolds = np.asarray([row["scaffold"] for row in rows], dtype=object)
                target_global = np.asarray(
                    [global_index[identity] for identity in identities], dtype=np.int64
                )
                target_vectors = np.asarray(embedding[common_to_model[target_global]])
                index_by_identity = {
                    identity: index for index, identity in enumerate(identities)
                }
                unique_anchor_ids = sorted(
                    {
                        identity
                        for draw_id in range(draws)
                        for identity in schedule[(target_id, primary_shots, draw_id)][1]
                    }
                )
                similarities = similarity_to_unique_anchors(
                    model,
                    target_vectors,
                    [index_by_identity[identity] for identity in unique_anchor_ids],
                )
                anchor_column = {
                    identity: index for index, identity in enumerate(unique_anchor_ids)
                }
                draw_curves = []
                for draw_id in range(draws):
                    _, anchor_ids = schedule[(target_id, primary_shots, draw_id)]
                    anchor_indices = [index_by_identity[identity] for identity in anchor_ids]
                    mask = candidate_mask(
                        labels, scaffolds, anchor_indices, scaffold_excluded=False
                    )
                    columns = [anchor_column[identity] for identity in anchor_ids]
                    scores = np.max(similarities[:, columns], axis=1)[mask]
                    draw_curves.append(interpolated_roc(scores, labels[mask], fpr_grid))
                per_target_curves.append(
                    np.mean(np.asarray(draw_curves, dtype=np.float64), axis=0)
                )
            target_matrix = np.asarray(per_target_curves, dtype=np.float64)
            if target_matrix.shape != (len(by_target), FPR_GRID_POINTS):
                raise RuntimeError(f"Unexpected ROC curve matrix for {model}")
            macro_curves[model] = target_matrix.mean(axis=0)
            target_counts[model] = target_matrix.shape[0]
            del embedding
    finally:
        maps_file.close()

    source_rows = []
    audit_rows = []
    for model in models:
        display_name = protocol["models"][model]["display_name"]
        curve = macro_curves[model]
        approximate_auc = float(np.trapz(curve, fpr_grid))
        error = abs(approximate_auc - exact_auc[model])
        if error > MAX_AUC_APPROXIMATION_ERROR:
            raise RuntimeError(
                f"Macro ROC grid does not reproduce target-mean AUC for {model}: {error}"
            )
        audit_rows.append(
            {
                "model": model,
                "display_name": display_name,
                "targets": target_counts[model],
                "draws_per_target": draws,
                "exact_mean_target_roc_auc": exact_auc[model],
                "trapezoid_auc_of_plotted_macro_curve": approximate_auc,
                "absolute_difference": error,
                "maximum_permitted_difference": MAX_AUC_APPROXIMATION_ERROR,
            }
        )
        for fpr, tpr in zip(fpr_grid, curve):
            source_rows.append(
                {
                    "model": model,
                    "display_name": display_name,
                    "false_positive_rate": float(fpr),
                    "macro_true_positive_rate": float(tpr),
                    "exact_mean_target_roc_auc": exact_auc[model],
                    "targets": target_counts[model],
                    "draws_per_target": draws,
                    "shots": primary_shots,
                    "condition": "standard",
                }
            )
    source_path = BENCHMARK_DIR / "figures/source-data/five_shot_macro_roc_curves.csv"
    audit_path = BENCHMARK_DIR / "figures/source-data/five_shot_macro_roc_auc_audit.csv"
    write_csv(source_path, source_rows, tuple(source_rows[0]))
    write_csv(audit_path, audit_rows, tuple(audit_rows[0]))

    configure()
    figure, axis = plt.subplots(figsize=(7.25, 4.9), constrained_layout=True)
    for model in models:
        axis.plot(
            fpr_grid,
            macro_curves[model],
            color=COLORS[model],
            linewidth=1.8 if model == "gmolai" else 1.45,
            label=f"{protocol['models'][model]['display_name']} (AUC {exact_auc[model]:.3f})",
        )
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=0.9, label="Random")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Five-shot ligand retrieval: target-balanced macro ROC", loc="left", fontweight="bold")
    axis.grid(color="#E1E1E1", linewidth=0.45)
    axis.text(
        0.02,
        0.02,
        f"{draws} draws averaged within each target; {len(by_target)} targets weighted equally",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#444444",
        ha="left",
        va="bottom",
    )
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    outputs = save_figure(figure, "five_shot_macro_roc_curves")

    manifest = {
        "schema_version": 1,
        "status": "ok",
        "analysis_role": "post-completion visualization of the prespecified primary ROC-AUC endpoint",
        "new_performance_endpoint_or_model_selection": False,
        "condition": "standard",
        "shots": primary_shots,
        "draws_per_target": draws,
        "targets": len(by_target),
        "aggregation": "interpolate each draw ROC; average draws within target; average targets with equal weight",
        "fpr_grid_points": FPR_GRID_POINTS,
        "maximum_auc_approximation_error": max(
            float(row["absolute_difference"]) for row in audit_rows
        ),
        "maximum_permitted_auc_approximation_error": MAX_AUC_APPROXIMATION_ERROR,
        "population_state_sha256": sha256_file(population_path),
        "anchors_state_sha256": sha256_file(anchors_state_path),
        "retrieval_state_sha256": sha256_file(retrieval_state_path),
        "summary_state_sha256": sha256_file(summary_state_path),
        "outputs": outputs,
        "source_data": [
            {"path": str(source_path), "sha256": sha256_file(source_path)},
            {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audits/roc_curve_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
