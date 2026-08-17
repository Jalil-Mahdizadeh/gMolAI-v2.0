#!/usr/bin/env python3
"""Create Step 03 figures and retain every plotted value as CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "navy": "#173F5F",
    "blue": "#20639B",
    "teal": "#3CAEA3",
    "gold": "#F6C85F",
    "red": "#ED553B",
    "gray": "#6B7280",
}


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.step_root.resolve()
    tables = root / "outputs" / "tables"
    plot_data = root / "outputs" / "plot-data"
    figures = root / "figures"
    summary = pd.read_csv(tables / "benchmark_summary.csv")
    batches = pd.read_csv(tables / "per_batch_timings.csv")
    seeds = pd.read_csv(tables / "per_seed_metrics.csv")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )

    headline = summary[
        summary["metric"].isin(
            [
                "raw_proposals_per_second",
                "valid_unique_molecules_per_second",
            ]
        )
    ].copy()
    order = {
        "raw_proposals_per_second": 0,
        "valid_unique_molecules_per_second": 1,
    }
    headline["plot_order"] = headline["metric"].map(order)
    headline = headline.sort_values("plot_order").reset_index(drop=True)
    headline["display_label"] = ["Raw proposals", "Valid unique molecules"]
    atomic_write_csv(plot_data / "headline_throughput.csv", headline)
    values = headline["value"].to_numpy(float)
    lower = values - headline["ci95_lower"].to_numpy(float)
    upper = headline["ci95_upper"].to_numpy(float) - values
    figure, axis = plt.subplots(figsize=(6.3, 4.2))
    bars = axis.bar(
        headline["display_label"],
        values,
        color=[COLORS["blue"], COLORS["teal"]],
        width=0.62,
        yerr=np.vstack([lower, upper]),
        capsize=5,
        error_kw={"elinewidth": 1.3, "capthick": 1.3},
    )
    axis.set_ylabel("Throughput (outputs s$^{-1}$)")
    axis.set_title("Released gMolAI decoder throughput on one GPU")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.text(
        0.01,
        -0.20,
        "100 molecular embeddings × 1,000 draws; bars show batch-bootstrap 95% intervals",
        transform=axis.transAxes,
        color=COLORS["gray"],
        fontsize=8.5,
    )
    save_figure(figure, figures / "decoder_throughput")

    batch_source = batches[
        [
            "batch_index",
            "first_benchmark_seed_index",
            "raw_proposals_per_second",
            "valid_unique_molecules_per_second",
            "generation_seconds",
            "token_decode_seconds",
            "rdkit_validation_seconds",
        ]
    ].copy()
    atomic_write_csv(plot_data / "batch_throughput.csv", batch_source)
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    x = batch_source["batch_index"] + 1
    axis.plot(
        x,
        batch_source["raw_proposals_per_second"],
        color=COLORS["blue"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Raw proposals",
    )
    axis.plot(
        x,
        batch_source["valid_unique_molecules_per_second"],
        color=COLORS["teal"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Valid unique molecules",
    )
    axis.set_xlabel("Measured two-seed batch")
    axis.set_ylabel("Throughput (outputs s$^{-1}$)")
    axis.set_title("Throughput across the measured run")
    axis.grid(color="#E5E7EB", linewidth=0.8)
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, figures / "batch_throughput_trace")

    seed_source = seeds[
        [
            "benchmark_seed_index",
            "seed_heavy_atoms",
            "rdkit_valid_fraction",
            "rdkit_unique_valid_molecules",
            "rdkit_unique_fraction_of_valid",
            "release_policy_accepted_fraction",
            "release_policy_unique_molecules",
        ]
    ].copy()
    atomic_write_csv(plot_data / "per_seed_yield.csv", seed_source)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    axes[0].hist(
        seed_source["rdkit_valid_fraction"] * 100.0,
        bins=14,
        color=COLORS["blue"],
        edgecolor="white",
    )
    axes[0].set_xlabel("RDKit-valid proposals (%)")
    axes[0].set_ylabel("Molecular embeddings")
    axes[0].set_title("Validity by conditioning seed")
    axes[1].hist(
        seed_source["rdkit_unique_valid_molecules"],
        bins=14,
        color=COLORS["teal"],
        edgecolor="white",
    )
    axes[1].set_xlabel("Unique valid molecules / 1,000 draws")
    axes[1].set_ylabel("Molecular embeddings")
    axes[1].set_title("Usable unique yield")
    for axis in axes:
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
    figure.suptitle("Decoder output quality across 100 random molecular embeddings", y=1.02)
    figure.tight_layout()
    save_figure(figure, figures / "per_seed_valid_unique_yield")
    print(f"Wrote figures to {figures} and source data to {plot_data}")


if __name__ == "__main__":
    main()
