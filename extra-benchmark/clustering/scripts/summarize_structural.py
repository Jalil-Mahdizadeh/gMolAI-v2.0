#!/usr/bin/env python3
"""Create paired-bootstrap ClassyFire tables from completed model evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_json, load_protocol, read_panel_tsv, sha256_file, write_csv
from metrics_common import atomic_save_npz


ALL_MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2", "morgan_count", "descriptor13")


def metric_triplet(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(adjusted_rand_score(y_true, y_pred)),
        float(adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
        float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
    )


def main() -> None:
    protocol = load_protocol()
    primary = tuple(protocol["models"]["primary_order"])
    panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv")
    subclasses = np.asarray([row["subclass"] for row in panel], dtype=object)
    names = sorted(set(subclasses))
    y_true = np.asarray([names.index(value) for value in subclasses], dtype=np.int16)
    strata = [np.flatnonzero(y_true == index) for index in range(len(names))]
    evaluations = {model: load_json(BENCHMARK_DIR / "state" / "structural" / f"{model}.json") for model in ALL_MODELS}
    clusters = {}
    same = {}
    native = {}
    for model in ALL_MODELS:
        labels = np.load(BENCHMARK_DIR / "artifacts" / "common" / "structural_clusters" / f"{model}.npz", allow_pickle=False)
        if not np.array_equal(labels["true_labels"], y_true):
            raise RuntimeError(f"True-label binding differs for {model}")
        clusters[model] = np.asarray(labels["cluster_labels"], dtype=np.int16)
        frame = pd.read_parquet(BENCHMARK_DIR / "outputs" / "source_data" / "structural_queries" / f"{model}.parquet")
        same[model] = frame["same_subclass_at_100"].to_numpy(dtype=np.float64)
        if "native_same_subclass_at_100" in frame:
            native[model] = frame["native_same_subclass_at_100"].to_numpy(dtype=np.float64)
    repetitions = int(protocol["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    child_seeds = np.random.SeedSequence(bootstrap_seed).spawn(repetitions)

    def one(seed_sequence):
        rng = np.random.default_rng(seed_sequence)
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        sampled_true = y_true[sampled]
        values = np.empty((len(ALL_MODELS), 4), dtype=np.float64)
        for model_index, model in enumerate(ALL_MODELS):
            seed_metrics = np.asarray([metric_triplet(sampled_true, predicted[sampled]) for predicted in clusters[model]])
            values[model_index, :3] = np.mean(seed_metrics, axis=0)
            values[model_index, 3] = float(np.mean(same[model][sampled]))
        return values

    jobs = min(16, max(1, (os.cpu_count() or 2) // 2))
    bootstrap = np.stack(Parallel(n_jobs=jobs, backend="loky", verbose=10)(delayed(one)(seed) for seed in child_seeds))
    metrics = ("ARI", "AMI", "NMI", "macro_same_subclass_at_100")
    point = np.empty((len(ALL_MODELS), len(metrics)), dtype=np.float64)
    for model_index, model in enumerate(ALL_MODELS):
        point[model_index, :3] = [evaluations[model]["mean_seed_metrics"][key] for key in metrics[:3]]
        point[model_index, 3] = evaluations[model]["macro_same_subclass_at_100"]
    lower = np.quantile(bootstrap, 0.025, axis=0)
    upper = np.quantile(bootstrap, 0.975, axis=0)
    display = {
        model: (
            protocol["models"][model]["display_name"]
            if model in protocol["models"] else protocol["descriptor_diagnostic"]["display_name"]
        ) for model in ALL_MODELS
    }
    rows = []
    for model_index, model in enumerate(ALL_MODELS):
        for metric_index, metric in enumerate(metrics):
            rows.append({
                "model": model, "display_name": display[model],
                "primary_ranking": model in primary, "metric": metric,
                "estimate": point[model_index, metric_index],
                "ci95_lower": lower[model_index, metric_index],
                "ci95_upper": upper[model_index, metric_index],
                "bootstrap_repetitions": repetitions,
            })
    table = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_structural_metrics.csv"
    write_csv(table, rows, tuple(rows[0]))
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_structural_main.csv"
    write_csv(source, [row for row in rows if row["primary_ranking"]], tuple(rows[0]))
    seed_rows = []
    for model in ALL_MODELS:
        for record in evaluations[model]["seed_metrics"]:
            seed_rows.append({"model": model, "display_name": display[model], **record})
    seed_table = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_kmeans_seed_sensitivity.csv"
    write_csv(seed_table, seed_rows, tuple(seed_rows[0]))
    difference_rows = []
    g_index = ALL_MODELS.index("gmolai")
    for model in primary:
        if model == "gmolai":
            continue
        model_index = ALL_MODELS.index(model)
        differences = bootstrap[:, g_index, :] - bootstrap[:, model_index, :]
        for metric_index, metric in enumerate(metrics):
            difference_rows.append({
                "contrast": f"gmolai_minus_{model}", "metric": metric,
                "estimate": point[g_index, metric_index] - point[model_index, metric_index],
                "ci95_lower": float(np.quantile(differences[:, metric_index], 0.025)),
                "ci95_upper": float(np.quantile(differences[:, metric_index], 0.975)),
            })
    difference_table = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_paired_differences.csv"
    write_csv(difference_table, difference_rows, tuple(difference_rows[0]))
    native_rows = []
    for model in ("morgan", "morgan_count"):
        native_rows.extend([
            {
                "model": model, "display_name": display[model], "distance": "normalized_euclidean",
                "macro_same_subclass_at_100": float(np.mean(same[model])),
            },
            {
                "model": model, "display_name": display[model], "distance": evaluations[model]["native_metric"],
                "macro_same_subclass_at_100": float(np.mean(native[model])),
            },
        ])
    native_table = BENCHMARK_DIR / "outputs" / "source_data" / "figure_morgan_native_sensitivity.csv"
    write_csv(native_table, native_rows, tuple(native_rows[0]))
    archive = BENCHMARK_DIR / "artifacts" / "common" / "classyfire_bootstrap.npz"
    atomic_save_npz(archive, values=bootstrap, models=np.asarray(ALL_MODELS), metrics=np.asarray(metrics))
    report = {
        "schema_version": 1, "status": "ok", "rows": len(panel),
        "models": list(ALL_MODELS), "primary_models": list(primary),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": bootstrap_seed,
        "parallel_jobs": jobs, "table": str(table), "table_sha256": sha256_file(table),
        "paired_differences": str(difference_table),
        "paired_differences_sha256": sha256_file(difference_table),
        "bootstrap_archive": str(archive), "bootstrap_archive_sha256": sha256_file(archive),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "structural_summary.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

