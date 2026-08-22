#!/usr/bin/env python3
"""Render one manuscript figure and one compact SI figure from source-data CSVs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    sha256_file,
)


COLORS = {
    "gmolai": "#D55E00",
    "morgan": "#0072B2",
    "molai": "#009E73",
    "molformer": "#CC79A7",
    "smi_ted": "#E69F00",
    "molclr_gin": "#56B4E9",
    "kermt_v2": "#6A3D9A",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.4,
        }
    )


def jitter(target_id: str, width: float = 0.16) -> float:
    value = int.from_bytes(hashlib.sha256(target_id.encode("utf-8")).digest()[:8], "big")
    return width * (2.0 * (value / float(1 << 64)) - 1.0)


def distribution_panel(axis, frame: pd.DataFrame, order: list[str], ylabel: str) -> None:
    arrays = [frame.loc[frame["model"] == model, "value"].to_numpy() for model in order]
    violins = axis.violinplot(
        arrays,
        positions=np.arange(len(order)),
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, model in zip(violins["bodies"], order):
        body.set_facecolor(COLORS[model])
        body.set_edgecolor(COLORS[model])
        body.set_alpha(0.18)
        body.set_linewidth(0.7)
    for position, model, values in zip(np.arange(len(order)), order, arrays):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot([position, position], [q1, q3], color=COLORS[model], lw=4.2, solid_capstyle="butt")
        axis.plot([position - 0.12, position + 0.12], [median, median], color="white", lw=1.2)
        subset = frame[frame["model"] == model]
        x = [position + jitter(target) for target in subset["target_id"]]
        axis.scatter(
            x,
            subset["value"],
            s=7,
            facecolors="white",
            edgecolors=COLORS[model],
            linewidths=0.45,
            alpha=0.72,
            rasterized=True,
            zorder=3,
        )
    labels = [frame.loc[frame["model"] == model, "display_name"].iloc[0] for model in order]
    axis.set_xticks(np.arange(len(order)), labels, rotation=34, ha="right")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9D9D9", zorder=0)
    axis.set_xlim(-0.55, len(order) - 0.45)


def save_figure(figure, stem: str) -> list[dict[str, str]]:
    outputs = []
    for extension in ("pdf", "svg", "png"):
        path = BENCHMARK_DIR / "figures" / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append({"path": str(path), "sha256": sha256_file(path)})
    plt.close(figure)
    return outputs


def main() -> None:
    protocol = load_protocol()
    state = load_json(BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json")
    target_path = BENCHMARK_DIR / "results/tables/retrieval_per_target.csv"
    if state.get("retrieval_per_target_sha256") != sha256_file(target_path):
        raise RuntimeError("Target-level retrieval table changed")
    frame = pd.read_csv(target_path)
    frame = frame[frame["included_in_across_target_summary"].astype(str).str.lower() == "true"]
    order = list(protocol["models"]["primary_order"])
    primary_shots = int(protocol["retrieval"]["primary_shots"])
    secondary_shots = int(protocol["retrieval"]["secondary_shots"])
    source_dir = BENCHMARK_DIR / "figures/source-data"
    source_dir.mkdir(parents=True, exist_ok=True)

    panel_a = frame[
        (frame["shots"] == primary_shots)
        & (frame["condition"] == "standard")
        & frame["model"].isin(order)
    ][["model", "display_name", "target_id", "target_class", "ef1_mean"]].rename(
        columns={"ef1_mean": "value"}
    )
    panel_a_path = source_dir / "main_panel_a_five_shot_ef1.csv"
    panel_a.to_csv(panel_a_path, index=False, lineterminator="\n")

    paired = panel_a.pivot(index=["target_id", "target_class"], columns="model", values="value").reset_index()
    panel_b = paired[["target_id", "target_class", "morgan", "gmolai"]].copy()
    panel_b["gmolai_minus_morgan"] = panel_b["gmolai"] - panel_b["morgan"]
    panel_b_path = source_dir / "main_panel_b_gmolai_vs_morgan.csv"
    panel_b.to_csv(panel_b_path, index=False, lineterminator="\n")

    panel_c = frame[
        (frame["shots"] == primary_shots)
        & (frame["condition"] == "scaffold_excluded")
        & frame["model"].isin(order)
    ][["model", "display_name", "target_id", "target_class", "ef1_mean", "eligible_draws"]].rename(
        columns={"ef1_mean": "value"}
    )
    panel_c_path = source_dir / "main_panel_c_scaffold_excluded_ef1.csv"
    panel_c.to_csv(panel_c_path, index=False, lineterminator="\n")

    configure()
    figure, axes = plt.subplots(1, 3, figsize=(11.3, 3.65), constrained_layout=True)
    distribution_panel(axes[0], panel_a, order, "EF1% (mean across anchor draws)")
    axes[0].set_title("A  Five-shot active retrieval", loc="left", fontweight="bold")
    maximum = float(max(panel_b["morgan"].max(), panel_b["gmolai"].max()))
    axes[1].plot([0, maximum], [0, maximum], ls="--", lw=0.9, color="#666666", zorder=0)
    axes[1].scatter(
        panel_b["morgan"],
        panel_b["gmolai"],
        c=[COLORS["gmolai"] if value > 0 else COLORS["morgan"] if value < 0 else "#777777" for value in panel_b["gmolai_minus_morgan"]],
        s=21,
        alpha=0.76,
        edgecolors="white",
        linewidths=0.35,
    )
    wins = int((panel_b["gmolai_minus_morgan"] > 0).sum())
    losses = int((panel_b["gmolai_minus_morgan"] < 0).sum())
    ties = int((panel_b["gmolai_minus_morgan"] == 0).sum())
    axes[1].text(
        0.04,
        0.96,
        f"gMolAI wins {wins}/{len(panel_b)}\nMorgan wins {losses}; ties {ties}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    axes[1].set_xlabel("Morgan EF1%")
    axes[1].set_ylabel("gMolAI EF1%")
    axes[1].set_title("B  Paired target comparison", loc="left", fontweight="bold")
    axes[1].grid(color="#E2E2E2", zorder=0)
    distribution_panel(axes[2], panel_c, order, "Scaffold-excluded EF1%")
    axes[2].set_title("C  Cross-scaffold retrieval", loc="left", fontweight="bold")
    main_outputs = save_figure(figure, "main_lbvs_figure")

    si_parts = []
    definitions = (
        (secondary_shots, "standard", "ef1_mean", "One-shot EF1%"),
        (primary_shots, "standard", "bedroc20_mean", "Five-shot BEDROC (α=20)"),
        (primary_shots, "standard", "roc_auc_mean", "Five-shot ROC-AUC"),
        (primary_shots, "standard", "average_precision_mean", "Five-shot average precision"),
    )
    for shots, condition, column, title in definitions:
        subset = frame[
            (frame["shots"] == shots)
            & (frame["condition"] == condition)
            & frame["model"].isin(order)
        ][["model", "display_name", "target_id", "target_class", column]].copy()
        subset = subset.rename(columns={column: "value"})
        subset["panel"] = title
        si_parts.append(subset)
    si_source = pd.concat(si_parts, ignore_index=True)
    si_path = source_dir / "si_secondary_metrics.csv"
    si_source.to_csv(si_path, index=False, lineterminator="\n")
    si_figure, si_axes = plt.subplots(2, 2, figsize=(8.0, 6.1), constrained_layout=True)
    for axis, (_, _, _, title), label in zip(si_axes.flat, definitions, "ABCD"):
        subset = si_source[si_source["panel"] == title]
        distribution_panel(axis, subset, order, title)
        axis.set_title(f"{label}  {title}", loc="left", fontweight="bold")
    si_outputs = save_figure(si_figure, "si_lbvs_secondary_metrics")

    roc_manifest_path = BENCHMARK_DIR / "audits/roc_curve_manifest.json"
    roc_manifest = load_json(roc_manifest_path)
    if roc_manifest.get("status") != "ok":
        raise RuntimeError("Macro ROC figure is incomplete")
    for artifact in (*roc_manifest["outputs"], *roc_manifest["source_data"]):
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Macro ROC artifact changed: {artifact['path']}")

    manifest = {
        "schema_version": 1,
        "status": "ok",
        "summary_state_sha256": sha256_file(BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json"),
        "main": {
            "outputs": main_outputs,
            "source_data": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (panel_a_path, panel_b_path, panel_c_path)
            ],
        },
        "supplementary": {
            "outputs": si_outputs,
            "source_data": [{"path": str(si_path), "sha256": sha256_file(si_path)}],
        },
        "roc": {
            "manifest": {
                "path": str(roc_manifest_path),
                "sha256": sha256_file(roc_manifest_path),
            },
            "outputs": roc_manifest["outputs"],
            "source_data": roc_manifest["source_data"],
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audits/figure_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

