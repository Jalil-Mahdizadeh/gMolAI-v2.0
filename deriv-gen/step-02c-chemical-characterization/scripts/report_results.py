#!/usr/bin/env python3
"""Create concise Step 2c figures, tables, and bounded decision report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    STEP_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    protocol,
    resolve_manifest_inputs,
    sha256_file,
    utc_now,
)


CATEGORY_LABELS = {
    "exact_seed_identity": "Exact seed spelling",
    "same_identity_alternative_smiles": "Seed identity, alternate SMILES",
    "one_cut_mmp_derivative": "One-cut MMP",
    "scaffold_preserving_non_mmp_analogue": "Scaffold-preserving non-MMP",
    "acyclic_non_mmp_analogue": "Acyclic non-MMP",
    "scaffold_changing_analogue": "Scaffold-changing",
}

COLORS = {
    "exact_seed_identity": "#4d4d4d",
    "same_identity_alternative_smiles": "#969696",
    "one_cut_mmp_derivative": "#1b9e77",
    "scaffold_preserving_non_mmp_analogue": "#66a61e",
    "acyclic_non_mmp_analogue": "#7570b3",
    "scaffold_changing_analogue": "#d95f02",
}


def pct(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def number(value: float | int | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> list[Path]:
    paths = [directory / f"{stem}.png", directory / f"{stem}.svg"]
    for path in paths:
        figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def category_figure(categories: pd.DataFrame, output: Path) -> list[Path]:
    frame = categories.copy()
    frame["label"] = frame["chemical_category"].map(CATEGORY_LABELS)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(frame))
    fractions = frame["fraction_of_retained_candidates"].to_numpy(dtype=float)
    axis.barh(
        positions,
        fractions,
        color=[COLORS[str(value)] for value in frame["chemical_category"]],
    )
    axis.set_yticks(positions, labels=frame["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Fraction of retained candidates")
    axis.set_xlim(0, max(float(fractions.max()) * 1.18, 0.01))
    for position, value in zip(positions, fractions):
        axis.text(value, position, f" {100 * value:.1f}%", va="center", fontsize=9)
    axis.set_title("Step 2c candidate identities and chemical categories")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, output, "candidate_category_composition")


def yield_figure(seeds: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    bins = np.arange(-0.5, 51.5, 2)
    axes[0].hist(
        seeds["genuine_nonseed_candidate_count"],
        bins=bins,
        color="#3182bd",
        alpha=0.78,
        label="Genuine non-seed",
    )
    axes[0].hist(
        seeds["mmp_derivative_count"],
        bins=bins,
        histtype="step",
        linewidth=2,
        color="#1b9e77",
        label="One-cut MMP",
    )
    axes[0].set_xlabel("Candidates per seed")
    axes[0].set_ylabel("Seeds")
    axes[0].set_title("Per-seed chemical yield")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(
        coverage["minimum_mmp_derivatives"],
        coverage["seed_fraction"],
        marker="o",
        color="#1b9e77",
    )
    axes[1].set_xticks(coverage["minimum_mmp_derivatives"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Minimum one-cut MMP candidates")
    axes[1].set_ylabel("Fraction of all seeds")
    axes[1].set_title("MMP derivative coverage")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, output, "per_seed_yield_and_mmp_coverage")


def similarity_figure(candidates: pd.DataFrame, output: Path) -> list[Path]:
    categories = [
        "one_cut_mmp_derivative",
        "scaffold_preserving_non_mmp_analogue",
        "acyclic_non_mmp_analogue",
        "scaffold_changing_analogue",
    ]
    arrays = [
        candidates.loc[
            candidates["chemical_category"] == category,
            "morgan_similarity_to_seed_recomputed",
        ].to_numpy(dtype=float)
        for category in categories
    ]
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    plot = axis.boxplot(
        arrays,
        labels=[CATEGORY_LABELS[value] for value in categories],
        showfliers=False,
        patch_artist=True,
        whis=(5, 95),
    )
    for patch, category in zip(plot["boxes"], categories):
        patch.set_facecolor(COLORS[category])
        patch.set_alpha(0.75)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Seed-candidate Morgan Tanimoto")
    axis.set_title("Chemical proximity of genuine non-seed candidates")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, output, "seed_candidate_similarity_by_category")


def diversity_figure(
    seeds: pd.DataFrame, histogram: pd.DataFrame, output: Path
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].scatter(
        seeds["unique_canonical_molecular_identities"],
        seeds["distinct_scaffold_keys"],
        s=8,
        alpha=0.25,
        color="#2c7fb8",
    )
    axes[0].set_xlabel("Unique molecular identities per seed")
    axes[0].set_ylabel("Distinct scaffold keys per seed")
    axes[0].set_title("Identity and scaffold exploration")
    axes[0].grid(alpha=0.2)

    for population, label, color in (
        ("all_unique_retained_candidates", "All retained", "#4d4d4d"),
        ("genuine_nonseed_candidates", "Genuine non-seed", "#d95f02"),
    ):
        frame = histogram.loc[histogram["population"] == population]
        centers = (frame["bin_left"] + frame["bin_right"]) / 2
        axes[1].plot(centers, frame["fraction"], label=label, color=color)
    axes[1].set_xlabel("Within-set pairwise Morgan Tanimoto")
    axes[1].set_ylabel("Pair fraction per 0.01 bin")
    axes[1].set_title("Within-set structural redundancy/diversity")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return save_figure(figure, output, "within_set_structural_diversity")


def transformation_figure(transformations: pd.DataFrame, output: Path) -> list[Path]:
    frame = transformations.head(15).iloc[::-1].copy()
    if frame.empty:
        return []
    labels = [
        value if len(value) <= 55 else value[:52] + "..."
        for value in frame["seed_to_candidate_transform"].astype(str)
    ]
    figure, axis = plt.subplots(figsize=(11.0, 6.2))
    axis.barh(np.arange(len(frame)), frame["candidate_count"], color="#1b9e77")
    axis.set_yticks(np.arange(len(frame)), labels=labels, fontsize=8)
    axis.set_xlabel("Candidates assigned this primary directional transform")
    axis.set_title("Most recurrent one-cut MMP transformations")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    return save_figure(figure, output, "recurrent_mmp_transformations")


def markdown_category_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Category | Candidates | Seeds | Retained fraction |",
        "|---|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        rows.append(
            f"| {CATEGORY_LABELS[str(row['chemical_category'])]} | "
            f"{int(row['candidate_count']):,} | {int(row['seed_count']):,} | "
            f"{pct(float(row['fraction_of_retained_candidates']))} |"
        )
    return "\n".join(rows)


def markdown_similarity_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Population | n | Mean | Median | IQR | q10--q90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        label = CATEGORY_LABELS.get(str(row["population"]), str(row["population"]))
        rows.append(
            f"| {label} | {int(row['n']):,} | {number(row['mean'])} | "
            f"{number(row['median'])} | {number(row['q25'])}--{number(row['q75'])} | "
            f"{number(row['q10'])}--{number(row['q90'])} |"
        )
    return "\n".join(rows)


def markdown_coverage_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Minimum MMP derivatives among 50 | Seeds | Fraction of all seeds |",
        "|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        rows.append(
            f"| {int(row['minimum_mmp_derivatives'])} | {int(row['seed_count']):,} | "
            f"{pct(float(row['seed_fraction']))} |"
        )
    return "\n".join(rows)


def markdown_transform_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Primary seed→candidate substituent transform | Class | Candidates | Seeds | Median Morgan |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in frame.head(12).iterrows():
        transform = str(row["seed_to_candidate_transform"]).replace("|", "\\|")
        rows.append(
            f"| `{transform}` | {row['mmp_edit_class']} | "
            f"{int(row['candidate_count']):,} | {int(row['seed_count']):,} | "
            f"{number(row['median_morgan_to_seed'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=STEP_ROOT.parents[2])
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    resolve_manifest_inputs(repo_root, step_root)
    cfg = protocol(step_root)
    analysis_seal = load_json(step_root / "state" / "ANALYSIS_COMPLETE.json")
    if analysis_seal.get("status") != "analysis_complete":
        raise RuntimeError("Step 2c analysis is not sealed")

    summary = load_json(step_root / "outputs" / "study_summary.json")
    categories = pd.read_csv(
        step_root / "outputs" / "tables" / "candidate_category_counts.csv"
    )
    seeds = pd.read_parquet(
        step_root / "outputs" / "tables" / "seed_characterization.parquet"
    )
    candidates = pd.read_parquet(
        step_root / "outputs" / "tables" / "candidate_characterization.parquet",
        columns=["chemical_category", "morgan_similarity_to_seed_recomputed"],
    )
    coverage = pd.read_csv(
        step_root / "outputs" / "tables" / "mmp_seed_coverage.csv"
    )
    similarities = pd.read_csv(
        step_root / "outputs" / "tables" / "seed_candidate_similarity_summary.csv"
    )
    transformations = pd.read_csv(
        step_root / "outputs" / "tables" / "mmp_transformation_counts.csv"
    )
    histogram = pd.read_csv(
        step_root / "outputs" / "tables" / "within_set_pairwise_morgan_histogram.csv"
    )

    figure_dir = step_root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    figure_paths += category_figure(categories, figure_dir)
    figure_paths += yield_figure(seeds, coverage, figure_dir)
    figure_paths += similarity_figure(candidates, figure_dir)
    figure_paths += diversity_figure(seeds, histogram, figure_dir)
    figure_paths += transformation_figure(transformations, figure_dir)

    population = summary["population"]
    validity = summary["validity_and_uniqueness"]
    mmp = summary["mmp_derivatives"]
    scaffold = summary["scaffold_preservation"]
    similarity = summary["seed_candidate_similarity"]["all_genuine_nonseed"]
    diversity = summary["within_set_diversity"]
    decision = summary["bounded_conclusion"]
    classification = decision["classification"]
    supported = classification == cfg["bounded_conclusion"]["label_if_all_gates_pass"]
    limited = classification == cfg["bounded_conclusion"][
        "label_if_yield_and_locality_pass_but_mmp_gate_fails"
    ]
    headline = (
        "SUPPORTED"
        if supported
        else "PARTIALLY SUPPORTED"
        if limited
        else "NOT SUPPORTED"
    )

    category_md = markdown_category_table(categories)
    coverage_md = markdown_coverage_table(coverage)
    similarity_md = markdown_similarity_table(similarities)
    transform_md = markdown_transform_table(transformations)
    results_text = f"""# Step 2c results: chemical characterization of frozen candidate sets

## Bounded conclusion

**{headline}: `{classification}`.** This conclusion applies only to the frozen,
unperturbed Step-2b correct-condition sets. It does not demonstrate controllable
latent edits, MMP-direction generation, novelty, synthesizability, or activity
improvement. No model was executed or trained in Step 2c.

The frozen rule gates were: {', '.join(f'`{key}`={str(value).lower()}' for key, value in decision['gates'].items())}.

## Denominators, validity, and uniqueness

All {population['seeds']:,} fresh-validation seeds were included. One seed can
have fewer than 50 retained candidates because Step 2b filtered invalid and
policy-rejected beam hypotheses and deduplicated canonical molecular identities.

- Persisted retained candidates: {population['retained_candidate_rows']:,};
  independently reparsed valid-SMILES rate
  {pct(validity['retained_rdkit_valid_rate_recomputed'])} and unchanged-policy
  acceptance {pct(validity['retained_policy_accepted_rate_recomputed'])}.
- Filled, valid, unique-identity slots: {pct(validity['retained_valid_unique_identity_slots_fraction_of_nominal_50'])}
  of the nominal {population['nominal_candidate_slots']:,} seed×50 slots; mean
  {validity['mean_unique_canonical_identities_per_seed']:.2f} and median
  {validity['median_unique_canonical_identities_per_seed']:.0f} identities per seed.
- Raw 64-beam hypotheses: valid-SMILES rate
  {pct(validity['raw_beam_rdkit_valid_rate'])}, policy-accepted rate
  {pct(validity['raw_beam_policy_accepted_rate'])}; greedy validity was
  {pct(validity['greedy_rdkit_valid_rate'])}.
- Accepted-beam molecular-identity redundancy was
  {validity['accepted_beam_identity_redundancy_count']:,}
  ({pct(validity['accepted_beam_identity_redundancy_fraction'])}). Individual
  discarded raw strings were not stored, so this aggregate cannot separate
  verbatim duplicate strings from alternative spellings of the same identity.
- A seed identity appeared in {validity['seed_identity_candidate_count']:,}
  sets: {validity['exact_seed_canonical_spelling_count']:,} used the exact
  canonical seed spelling and
  {validity['alternative_smiles_same_seed_identity_count']:,} used an alternate
  raw SMILES spelling. Neither is counted as a derivative.

## Primary chemical classification

{category_md}

Genuine derivatives/analogues are the {mmp['genuine_nonseed_candidates']:,}
non-seed rows; the seed and an alternate SMILES for the seed are excluded.

## One-cut MMP derivatives

The exact hash-bound Step-1b fragmentation code identified
{mmp['one_cut_mmp_candidates']:,} unique non-seed candidates as true one-cut
MMPs ({pct(mmp['global_mmp_fraction_among_genuine_nonseed'])} of genuine
non-seed candidates). At least one MMP was present for
{mmp['seeds_with_at_least_one_mmp']:,} seeds
({pct(mmp['seeds_with_at_least_one_mmp_fraction'])}).

{coverage_md}

All valid MMP explanations are in `mmp_explanations.parquet`.
{mmp['mmp_candidates_with_multiple_explanations']:,} MMP candidates had more
than one valid core explanation. The recurrent table below uses only the frozen
deterministic primary explanation, so its labels are summaries rather than
claims of a uniquely determined chemical edit.

{transform_md}

## Scaffold preservation and chemical proximity

Among genuine non-seed candidates whose seed had a non-empty Bemis-Murcko
scaffold, {pct(scaffold['nonempty_seed_scaffold_retention_rate'])} retained it.
The exact scaffold-key rate including empty keys was
{pct(scaffold['all_nonseed_exact_scaffold_key_rate_including_empty'])}; both seed
and candidate were acyclic in
{pct(scaffold['both_seed_and_candidate_acyclic_rate_among_nonseed'])} of
non-seed rows. The median seed produced
{scaffold['median_distinct_scaffold_keys_per_seed']:.0f} scaffold keys
({scaffold['median_distinct_nonempty_scaffolds_per_seed']:.0f} non-empty).

For all genuine non-seed candidates, seed-candidate Morgan Tanimoto had mean
{number(similarity['mean'])}, median {number(similarity['median'])}, IQR
{number(similarity['q25'])}--{number(similarity['q75'])}, and q10--q90
{number(similarity['q10'])}--{number(similarity['q90'])}.

{similarity_md}

## Within-set diversity and non-MMP graph changes

Across {diversity['genuine_nonseed_pair_count']:,} within-set non-seed pairs,
Morgan Tanimoto had mean {number(diversity['genuine_nonseed']['mean'])}, median
{number(diversity['genuine_nonseed']['median'])}, and IQR
{number(diversity['genuine_nonseed']['q25'])}--{number(diversity['genuine_nonseed']['q75'])}.
Only {pct(diversity['fraction_all_candidate_pairs_tanimoto_ge_0_90'])} of all
distinct-identity within-set pairs had Tanimoto ≥0.90. Tanimoto 1.0 can still
occur for different canonical identities because this Morgan fingerprint does
not encode every stereochemical distinction.

For non-MMP candidates, `non_mmp_graph_delta_summary.csv` and
`non_mmp_descriptor_delta_patterns.csv` report signed heavy-atom, bond, ring,
heteroatom, formal-charge, aromatic, and elemental-count changes. These are
descriptor deltas, not a claimed unique atom mapping or graph-edit path.

## Artifact map

- Candidate-level table: `outputs/tables/candidate_characterization.parquet`
- Seed-level tables: `outputs/tables/seed_characterization.parquet` and `.csv`
- MMP explanations/transforms: `outputs/tables/mmp_explanations.parquet` and
  `mmp_transformation_counts.csv`
- Similarity/diversity: `outputs/tables/seed_candidate_similarity_summary.csv`,
  `within_set_pairwise_morgan_histogram.csv`, and
  `outputs/raw/within_set_pairwise_morgan.npz`
- Figures: `outputs/figures/`

No Step 3 or latent perturbation was performed.
"""
    atomic_write_text(step_root / "RESULTS.md", results_text, step_root)

    decision_payload = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "classification": classification,
        "headline": headline,
        "gate_values": decision["gate_values"],
        "gates": decision["gates"],
        "scope": decision["scope"],
        "model_executed": False,
        "candidate_regeneration": False,
        "latent_perturbation": False,
        "mmp_directed_generation": False,
    }
    atomic_write_json(step_root / "outputs" / "decision.json", decision_payload, step_root)
    decision_text = f"""# Step 2c bounded decision

**{headline}: `{classification}`.**

This decision concerns only whether the frozen unperturbed Step-2b candidate
sets exhibit useful local-analogue chemistry under the prospective descriptive
gates. It is not authorization or evidence for Step 3, latent perturbation, or
MMP-directed generation. See `RESULTS.md` and `outputs/decision.json` for the
gate values and denominators.
"""
    atomic_write_text(step_root / "DECISION.md", decision_text, step_root)
    readme_text = f"""# Step 2c: frozen candidate-set chemical audit

This directory contains the no-training chemical characterization of all
{population['seeds']:,} frozen Step-2b correct-condition seeds. The bounded
outcome is **{headline}** (`{classification}`).

Run `scripts/run_study.sh` from the repository checkout to reproduce the audit
inside the pinned gMolAI container. The runner reads Step-1b/Step-2b artifacts
without modifying them and writes only inside this directory.

- `PROTOCOL.md` and `DESIGN.md`: frozen definitions and denominator discipline.
- `RESULTS.md` and `DECISION.md`: scientific report and bounded conclusion.
- `config/`: frozen analysis settings.
- `inputs/`: SHA-256-bound read-only inputs and source provenance.
- `scripts/`: registration, component tests, audit, reporting, and verification.
- `intermediate/`: independently audited chemistry and exact one-cut fragments.
- `outputs/tables/`: candidate-, seed-, transformation-, and summary-level data.
- `outputs/figures/`: concise PNG and SVG figures.
- `state/`: registration, stage, environment, and completion seals.

No model training/execution, candidate regeneration, latent perturbation, or
MMP-directed generation occurs here.
"""
    atomic_write_text(step_root / "README.md", readme_text, step_root)

    report_outputs = {
        "results": step_root / "RESULTS.md",
        "decision_markdown": step_root / "DECISION.md",
        "readme": step_root / "README.md",
        "decision_json": step_root / "outputs" / "decision.json",
        **{
            f"figure_{index:02d}": path for index, path in enumerate(figure_paths, start=1)
        },
    }
    seal = {
        "schema_version": 1,
        "status": "report_complete",
        "completed_at": utc_now(),
        "classification": classification,
        "outputs": {
            name: {
                "path": str(path.relative_to(step_root)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in report_outputs.items()
        },
    }
    atomic_write_json(step_root / "state" / "REPORT_COMPLETE.json", seal, step_root)
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
