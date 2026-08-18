#!/usr/bin/env python3
"""Render publication-quality main and SI figures from exact frozen source tables."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, sha256_file, write_csv


PRIMARY = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")
COLORS = {
    "gmolai": "#D55E00", "morgan": "#0072B2", "molai": "#009E73",
    "molformer": "#CC79A7", "smi_ted": "#E69F00", "molclr_gin": "#56B4E9",
    "kermt_v2": "#6A3D9A", "morgan_count": "#4C78A8", "descriptor13": "#7F7F7F",
}
DISPLAY = {
    "gmolai": "gMolAI", "morgan": "Morgan (binary)", "molai": "MolAI",
    "molformer": "MoLFormer", "smi_ted": "SMI-TED-Light",
    "molclr_gin": "MolCLR-GIN", "kermt_v2": "KERMT v2",
    "morgan_count": "Morgan (count sensitivity)",
    "descriptor13": "13-descriptor diagnostic",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
        "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    })


def save(fig, stem: str, sources: list[Path], manifest: list[dict]) -> None:
    directory = BENCHMARK_DIR / "outputs" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("pdf", "png"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append({"path": str(path), "sha256": sha256_file(path)})
    plt.close(fig)
    manifest.append({
        "figure": stem, "files": outputs,
        "source_data": [{"path": str(path), "sha256": sha256_file(path)} for path in sources],
    })


def metric_point_figure(frame, metrics, titles, stem, source, manifest):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    order = list(PRIMARY)
    display = {row.model: row.display_name for row in frame.itertuples()}
    y = np.arange(len(order))[::-1]
    for axis_index, (axis, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
        subset = frame[frame["metric"] == metric].set_index("model").loc[order]
        estimates = subset["estimate"].to_numpy()
        lower = estimates - subset["ci95_lower"].to_numpy()
        upper = subset["ci95_upper"].to_numpy() - estimates
        for position, model, estimate, lo, hi in zip(y, order, estimates, lower, upper):
            axis.errorbar(estimate, position, xerr=np.asarray([[lo], [hi]]), fmt="o", ms=5,
                          color=COLORS[model], ecolor=COLORS[model], elinewidth=1, capsize=2)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9", zorder=0)
        axis.set_ylim(-0.7, len(order) - 0.3)
        if axis_index % 2 == 0:
            axis.set_yticks(y, [display[model] for model in order])
        else:
            axis.set_yticks(y, [])
        axis.set_xlabel("Estimate (95% paired bootstrap CI)")
    save(fig, stem, [source], manifest)


def classyfire_pca(manifest):
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_classyfire_pca.csv"
    frame = pd.read_csv(source)
    labels = sorted(frame["subclass"].unique())
    cmap = mpl.colormaps["turbo"].resampled(len(labels))
    colors = {label: cmap(index) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), constrained_layout=True)
    for axis, model, title in zip(axes, ("gmolai", "morgan"), ("gMolAI", "Morgan (binary)")):
        subset = frame[frame["model"] == model]
        for label in labels:
            values = subset[subset["subclass"] == label]
            axis.scatter(values["PC1"], values["PC2"], s=3.5, alpha=0.46,
                         color=colors[label], edgecolors="none", rasterized=True)
        v1 = 100 * subset["PC1_explained_variance_fraction"].iloc[0]
        v2 = 100 * subset["PC2_explained_variance_fraction"].iloc[0]
        axis.set_xlabel(f"PC1 ({v1:.1f}%)")
        axis.set_ylabel(f"PC2 ({v2:.1f}%)")
        axis.set_title(title, loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4,
                      markerfacecolor=colors[label], markeredgewidth=0, label=label) for label in labels]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               columnspacing=0.8, handletextpad=0.3, fontsize=6.3)
    save(fig, "figure_classyfire_pca", [source], manifest)


def property_deviations(manifest):
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_deviations.csv"
    frame = pd.read_csv(source)
    frame = frame[frame["primary_ranking"].astype(str).str.lower().isin(("true", "1"))]
    properties = list(frame["property"].unique())
    titles = {"DFT_HOMO_ENERGY": "HOMO energy", "DFT_HOMO_LUMO_GAP": "HOMO–LUMO gap", "log1p_DFT_DIPOLE_TOT": "log(1 + dipole)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2), constrained_layout=True)
    y = np.arange(len(PRIMARY))[::-1]
    display = frame.drop_duplicates("model").set_index("model")["display_name"].to_dict()
    for axis_index, (axis, property_name) in enumerate(zip(axes, properties)):
        subset = frame[frame["property"] == property_name].set_index("model").loc[list(PRIMARY)]
        for position, model, value in zip(y, PRIMARY, subset["median_absolute_neighbor_deviation"]):
            axis.plot(value, position, "o", color=COLORS[model], ms=5)
        axis.set_title(titles.get(property_name, property_name), loc="left", fontweight="bold")
        axis.set_xlabel("Median absolute\nrobust-scaled deviation")
        axis.grid(axis="x", color="#D9D9D9")
        if axis_index == 0:
            axis.set_yticks(y, [display[model] for model in PRIMARY])
        else:
            axis.set_yticks(y, [])
    save(fig, "figure_qmugs_property_deviations", [source], manifest)


def qmugs_pca(manifest):
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_qmugs_pca.csv"
    frame = pd.read_csv(source)
    color_column = "robust_DFT_HOMO_LUMO_GAP"
    lo, hi = np.quantile(frame[color_column], (0.01, 0.99))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    artist = None
    for axis, model, title in zip(axes, ("gmolai", "morgan"), ("gMolAI", "Morgan (binary)")):
        subset = frame[frame["model"] == model]
        artist = axis.scatter(subset["PC1"], subset["PC2"], c=subset[color_column],
                              cmap="viridis", vmin=lo, vmax=hi, s=3, alpha=0.55,
                              edgecolors="none", rasterized=True)
        v1 = 100 * subset["PC1_explained_variance_fraction"].iloc[0]
        v2 = 100 * subset["PC2_explained_variance_fraction"].iloc[0]
        axis.set_xlabel(f"PC1 ({v1:.1f}%)")
        axis.set_ylabel(f"PC2 ({v2:.1f}%)")
        axis.set_title(title, loc="left", fontweight="bold")
    colorbar = fig.colorbar(artist, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label("Robust-scaled DFT HOMO–LUMO gap")
    save(fig, "figure_qmugs_pca_homo_lumo_gap", [source], manifest)


def coverage_figure(manifest):
    paths = [BENCHMARK_DIR / "outputs" / "tables" / name for name in ("classyfire_coverage.csv", "qmugs_coverage.csv")]
    frames = []
    for benchmark, path in zip(("ClassyFire-25", "QMugs"), paths):
        frame = pd.read_csv(path)
        frame["benchmark"] = benchmark
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_coverage.csv"
    write_csv(source, combined.to_dict("records"), tuple(combined.columns))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for axis, benchmark in zip(axes, ("ClassyFire-25", "QMugs")):
        subset = combined[combined["benchmark"] == benchmark].set_index("model").loc[list(PRIMARY)]
        rejection_per_thousand = 1000 * (1 - subset["coverage_fraction"].to_numpy())
        axis.bar(np.arange(len(PRIMARY)), rejection_per_thousand,
                 color=[COLORS[model] for model in PRIMARY], width=0.72)
        axis.set_xticks(np.arange(len(PRIMARY)), [DISPLAY[model] for model in PRIMARY], rotation=40, ha="right")
        axis.set_ylabel("Rejected per 1,000 attempted")
        axis.set_title(benchmark, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D9D9D9")
    save(fig, "figure_si_coverage", [source], manifest)


def native_figure(manifest):
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_morgan_native_sensitivity.csv"
    frame = pd.read_csv(source)
    fig, axis = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
    models = list(frame["model"].unique())
    width = 0.34
    x = np.arange(len(models))
    common_values = []
    native_values = []
    for model in models:
        subset = frame[frame["model"] == model]
        common_values.append(subset[subset["distance"] == "normalized_euclidean"]["macro_same_subclass_at_100"].iloc[0])
        native_row = subset[subset["distance"] != "normalized_euclidean"].iloc[0]
        native_values.append(native_row["macro_same_subclass_at_100"])
    axis.bar(x - width / 2, common_values, width=width, color="#7A7A7A",
             label="Normalized Euclidean (common operator)")
    axis.bar(x + width / 2, native_values, width=width, color="#0072B2",
             label="Native fingerprint similarity")
    axis.set_xticks(x, [frame[frame["model"] == model]["display_name"].iloc[0] for model in models])
    axis.set_ylabel("Macro same-subclass@100")
    axis.set_xlabel("Native: binary Tanimoto (binary); generalized Tanimoto on raw counts (count)",
                    fontsize=6.8)
    axis.set_ylim(0, max(common_values + native_values) * 1.18)
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#D9D9D9")
    save(fig, "figure_si_morgan_native_distance", [source], manifest)


def diagnostic_figure(manifest):
    structural_path = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_structural_metrics.csv"
    property_path = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_metrics.csv"
    structural = pd.read_csv(structural_path)
    prop = pd.read_csv(property_path)
    keep = ("gmolai", "morgan", "morgan_count", "descriptor13")
    records = []
    for benchmark, frame in (("ClassyFire-25", structural), ("QMugs", prop)):
        subset = frame[frame["model"].isin(keep)].copy()
        subset["benchmark"] = benchmark
        records.extend(subset.to_dict("records"))
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_diagnostic_sensitivities.csv"
    columns = list(records[0])
    write_csv(source, records, columns)
    frame = pd.DataFrame(records)
    metric_specs = (("ARI", "ARI"), ("AMI", "AMI"), ("NPD_at_100", "NPD@100 (lower better)"), ("property_neighbor_recall_at_100", "Property Recall@100"))
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metric_specs):
        subset = frame[frame["metric"] == metric].set_index("model").loc[list(keep)]
        axis.bar(np.arange(len(keep)), subset["estimate"], color=[COLORS[model] for model in keep])
        axis.set_xticks(np.arange(len(keep)), ["gMolAI", "Morgan", "Count\nMorgan", "13 desc."], rotation=0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D9D9D9")
    save(fig, "figure_si_count_morgan_descriptor_diagnostics", [source], manifest)


def decile_figure(manifest):
    source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_heavy_atom_deciles.csv"
    frame = pd.read_csv(source)
    frame = frame[frame["model"].isin(PRIMARY)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    specs = (("NPD_at_100", "NPD@100 (lower better)"), ("property_neighbor_recall_at_100", "Property Recall@100"))
    for axis, (metric, title) in zip(axes, specs):
        for model in PRIMARY:
            subset = frame[frame["model"] == model].sort_values("heavy_atom_decile")
            axis.plot(subset["heavy_atom_decile"], subset[metric], marker="o", ms=2.8,
                      lw=1.2 if model in ("gmolai", "morgan") else 0.8,
                      alpha=1 if model in ("gmolai", "morgan") else 0.72,
                      color=COLORS[model], label=subset["display_name"].iloc[0])
        axis.set_xlabel("Heavy-atom-count decile")
        axis.set_ylabel(title)
        axis.set_xticks(range(1, 11))
        axis.grid(color="#D9D9D9")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "figure_si_qmugs_heavy_atom_deciles", [source], manifest)


def main() -> None:
    configure()
    manifest = []
    structural_source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_structural_main.csv"
    structural = pd.read_csv(structural_source)
    metric_point_figure(
        structural, ("ARI", "AMI", "NMI", "macro_same_subclass_at_100"),
        ("Adjusted Rand index", "Adjusted mutual information", "Normalized mutual information", "Macro same-subclass@100"),
        "figure_classyfire_main_metrics", structural_source, manifest,
    )
    classyfire_pca(manifest)
    property_source = BENCHMARK_DIR / "outputs" / "source_data" / "figure_property_main.csv"
    prop = pd.read_csv(property_source)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    display = prop.drop_duplicates("model").set_index("model")["display_name"].to_dict()
    y = np.arange(len(PRIMARY))[::-1]
    for axis_index, (axis, metric, title) in enumerate(zip(axes, ("NPD_at_100", "property_neighbor_recall_at_100"), ("NPD@100 (lower is better)", "Property-neighbor Recall@100"))):
        subset = prop[prop["metric"] == metric].set_index("model").loc[list(PRIMARY)]
        estimate = subset["estimate"].to_numpy()
        for position, model, value, lo, hi in zip(y, PRIMARY, estimate, subset["ci95_lower"], subset["ci95_upper"]):
            axis.errorbar(value, position, xerr=np.asarray([[value-lo], [hi-value]]), fmt="o", ms=5,
                          color=COLORS[model], ecolor=COLORS[model], capsize=2)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", color="#D9D9D9")
        axis.set_xlabel("Estimate (95% paired bootstrap CI)")
        if axis_index == 0:
            axis.set_yticks(y, [display[model] for model in PRIMARY])
        else:
            axis.set_yticks(y, [])
    save(fig, "figure_qmugs_main_metrics", [property_source], manifest)
    property_deviations(manifest)
    qmugs_pca(manifest)
    coverage_figure(manifest)
    native_figure(manifest)
    diagnostic_figure(manifest)
    decile_figure(manifest)
    report = {
        "schema_version": 1, "status": "ok", "figures": manifest,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = BENCHMARK_DIR / "audit" / "figure_manifest.json"
    atomic_write_json(output, report)
    print(json.dumps({"status": "ok", "figures": len(manifest), "manifest": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
