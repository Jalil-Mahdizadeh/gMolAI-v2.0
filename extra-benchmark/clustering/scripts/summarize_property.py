#!/usr/bin/env python3
"""Create paired-bootstrap QMugs property tables from completed model evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_json, load_protocol, read_panel_tsv, sha256_file, write_csv
from metrics_common import atomic_save_npz


ALL_MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2", "morgan_count", "descriptor13")
METRICS = ("NPD_at_100", "property_neighbor_recall_at_100")


def main() -> None:
    protocol = load_protocol()
    primary = tuple(protocol["models"]["primary_order"])
    panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv")
    evaluations = {model: load_json(BENCHMARK_DIR / "state" / "property" / f"{model}.json") for model in ALL_MODELS}
    query = {}
    strata = None
    for model in ALL_MODELS:
        frame = pd.read_parquet(BENCHMARK_DIR / "outputs" / "source_data" / "property_queries" / f"{model}.parquet")
        query[model] = frame.loc[:, METRICS].to_numpy(dtype=np.float64).T
        current_strata = frame["heavy_atom_decile"].to_numpy(dtype=np.int8)
        if strata is None:
            strata = current_strata
        elif not np.array_equal(strata, current_strata):
            raise RuntimeError("QMugs heavy-atom strata differ between models")
    assert strata is not None
    stratum_indices = [np.flatnonzero(strata == value) for value in range(1, 11)]
    repetitions = int(protocol["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    child_seeds = np.random.SeedSequence(bootstrap_seed + 1).spawn(repetitions)

    def one(seed_sequence):
        rng = np.random.default_rng(seed_sequence)
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in stratum_indices])
        return np.stack([np.mean(query[model][:, sampled], axis=1) for model in ALL_MODELS])

    jobs = min(16, max(1, (os.cpu_count() or 2) // 2))
    bootstrap = np.stack(Parallel(n_jobs=jobs, backend="loky", verbose=10)(delayed(one)(seed) for seed in child_seeds))
    point = np.asarray([[evaluations[model][metric] for metric in METRICS] for model in ALL_MODELS])
    lower = np.quantile(bootstrap, 0.025, axis=0)
    upper = np.quantile(bootstrap, 0.975, axis=0)
    display = {model: (protocol["models"][model]["display_name"] if model in protocol["models"] else protocol["descriptor_diagnostic"]["display_name"]) for model in ALL_MODELS}
    rows = []
    for model_index, model in enumerate(ALL_MODELS):
        for metric_index, metric in enumerate(METRICS):
            rows.append({
                "model": model, "display_name": display[model], "primary_ranking": model in primary,
                "metric": metric, "estimate": point[model_index, metric_index],
                "ci95_lower": lower[model_index, metric_index],
                "ci95_upper": upper[model_index, metric_index],
                "bootstrap_repetitions": repetitions,
            })
    table = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_metrics.csv"
    write_csv(table, rows, tuple(rows[0]))
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_main.csv"
    write_csv(source, [row for row in rows if row["primary_ranking"]], tuple(rows[0]))
    difference_rows = []
    g_index = ALL_MODELS.index("gmolai")
    for model in primary:
        if model == "gmolai":
            continue
        model_index = ALL_MODELS.index(model)
        difference = bootstrap[:, g_index, :] - bootstrap[:, model_index, :]
        for metric_index, metric in enumerate(METRICS):
            difference_rows.append({
                "contrast": f"gmolai_minus_{model}", "metric": metric,
                "estimate": point[g_index, metric_index] - point[model_index, metric_index],
                "ci95_lower": float(np.quantile(difference[:, metric_index], 0.025)),
                "ci95_upper": float(np.quantile(difference[:, metric_index], 0.975)),
            })
    difference_table = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_paired_differences.csv"
    write_csv(difference_table, difference_rows, tuple(difference_rows[0]))
    deviation_rows = []
    for model in ALL_MODELS:
        for property_name, value in evaluations[model]["per_property_median_absolute_neighbor_deviation"].items():
            deviation_rows.append({
                "model": model, "display_name": display[model], "primary_ranking": model in primary,
                "property": property_name, "median_absolute_neighbor_deviation": value,
            })
    deviation_table = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_deviations.csv"
    write_csv(deviation_table, deviation_rows, tuple(deviation_rows[0]))
    decile_rows = []
    for model in ALL_MODELS:
        for record in evaluations[model]["heavy_atom_deciles"]:
            decile_rows.append({"model": model, "display_name": display[model], **record})
    decile_table = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_heavy_atom_deciles.csv"
    write_csv(decile_table, decile_rows, tuple(decile_rows[0]))
    archive = BENCHMARK_DIR / "artifacts" / "common" / "qmugs_bootstrap.npz"
    atomic_save_npz(archive, values=bootstrap, models=np.asarray(ALL_MODELS), metrics=np.asarray(METRICS))
    report = {
        "schema_version": 1, "status": "ok", "rows": len(panel), "models": list(ALL_MODELS),
        "bootstrap_repetitions": repetitions, "bootstrap_seed": bootstrap_seed + 1,
        "parallel_jobs": jobs, "table": str(table), "table_sha256": sha256_file(table),
        "paired_differences": str(difference_table), "paired_differences_sha256": sha256_file(difference_table),
        "bootstrap_archive": str(archive), "bootstrap_archive_sha256": sha256_file(archive),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "property_summary.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

