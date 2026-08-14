#!/usr/bin/env python3
"""Audit all frozen Step-2b correct-condition candidate sets chemically."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import rdkit

from audit_core import (
    audit_raw_smiles,
    compute_morgan_and_diversity,
    histogram_table,
    mmp_explanations,
)
from common import (
    STEP_ROOT,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    load_json,
    numeric_summary,
    protocol,
    resolve_manifest_inputs,
    sha256_file,
    utc_now,
)


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
STEP1B_SCRIPTS = REPO_ROOT / "deriv-gen" / "step-01b-scaled-space-selection" / "scripts"
if str(STEP1B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP1B_SCRIPTS))

from scaled_common import fragment_molecules  # noqa: E402


DESCRIPTOR_FIELDS = [
    "atom_count",
    "heavy_atom_count",
    "bond_count",
    "ring_count",
    "heteroatom_count",
    "formal_charge",
    "aromatic_atom_count",
    "aromatic_bond_count",
]

CATEGORY_ORDER = [
    "exact_seed_identity",
    "same_identity_alternative_smiles",
    "one_cut_mmp_derivative",
    "scaffold_preserving_non_mmp_analogue",
    "acyclic_non_mmp_analogue",
    "scaffold_changing_analogue",
]


def assert_equal_series(first: pd.Series, second: pd.Series, message: str) -> None:
    left = first.fillna("").astype(str).to_numpy()
    right = second.fillna("").astype(str).to_numpy()
    mismatches = int(np.sum(left != right))
    if mismatches:
        raise RuntimeError(f"{message}: {mismatches} mismatches")


def element_delta(seed_json: str, candidate_json: str) -> str:
    seed = json.loads(str(seed_json))
    candidate = json.loads(str(candidate_json))
    keys = sorted(set(seed) | set(candidate))
    delta = {
        key: int(candidate.get(key, 0)) - int(seed.get(key, 0))
        for key in keys
        if int(candidate.get(key, 0)) != int(seed.get(key, 0))
    }
    return json.dumps(delta, sort_keys=True, separators=(",", ":"))


def summary_frame(populations: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, values in populations.items():
        rows.append({"population": population, **numeric_summary(values)})
    return pd.DataFrame(rows)


def scalar_metric_rows(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            rows.append({"metric": prefix, "value": value})

    for key in (
        "population",
        "validity_and_uniqueness",
        "mmp_derivatives",
        "scaffold_preservation",
        "within_set_diversity",
        "bounded_conclusion",
    ):
        visit(key, summary[key])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    cfg = protocol(step_root)
    workers = int(args.workers or cfg["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    started_at = utc_now()
    wall_started = time.monotonic()

    registered = step_root / "state" / "REGISTERED.json"
    if not registered.is_file():
        raise RuntimeError("Register Step 2c before analysis")
    paths, input_hashes = resolve_manifest_inputs(repo_root, step_root)
    if cfg["scientific_boundary"]["candidate_regeneration"] is not False:
        raise RuntimeError("Protocol unexpectedly permits candidate regeneration")

    step1_cfg = load_json(paths["external_step1b_protocol"])
    for key in (
        "min_core_heavy_atoms",
        "min_variable_heavy_atoms",
        "max_variable_heavy_atoms",
        "min_core_fraction",
        "max_variable_heavy_atom_delta",
        "max_parent_heavy_atom_delta",
    ):
        if cfg["mmp"][key] != step1_cfg["mmp"][key]:
            raise RuntimeError(f"Step-2c MMP setting differs from Step 1b: {key}")

    print("loading frozen correct-condition candidate sets", flush=True)
    candidates = pd.read_parquet(
        paths["external_step2b_candidates"],
        filters=[
            ("phase", "==", cfg["candidate_population"]["source_phase"]),
            ("policy", "==", cfg["candidate_population"]["source_policy"]),
            ("control", "==", cfg["candidate_population"]["source_control"]),
        ],
    ).sort_values(["query_position", "proposal_rank"], ignore_index=True)
    generation = pd.read_csv(paths["external_step2b_generation_stats"])
    generation = generation.loc[
        (generation["phase"] == cfg["candidate_population"]["source_phase"])
        & (generation["policy"] == cfg["candidate_population"]["source_policy"])
        & (generation["control"] == cfg["candidate_population"]["source_control"])
    ].sort_values("query_position", ignore_index=True)
    panel = pd.read_csv(paths["external_step2b_panel"]).sort_values(
        "query_position", ignore_index=True
    )
    validation = pd.read_parquet(paths["external_validation_molecules"])

    expected_seeds = int(cfg["candidate_population"]["expected_seed_rows"])
    if len(panel) != expected_seeds or len(generation) != expected_seeds:
        raise RuntimeError("Frozen Step-2b seed or generation-stat count changed")
    if panel["query_position"].tolist() != list(range(expected_seeds)):
        raise RuntimeError("Panel query positions are not contiguous")
    if generation["query_position"].tolist() != list(range(expected_seeds)):
        raise RuntimeError("Generation-stat query positions are not contiguous")
    if candidates.empty:
        raise RuntimeError("No correct-condition candidates were found")
    if not set(candidates["query_position"]).issubset(set(range(expected_seeds))):
        raise RuntimeError("Candidate query position lies outside the panel")
    if candidates.duplicated(["query_position", "candidate_hash"]).any():
        raise RuntimeError("Step-2b retained set contains a duplicate molecular identity")
    if candidates.duplicated(["query_position", "proposal_rank"]).any():
        raise RuntimeError("Step-2b retained set contains a duplicate proposal rank")
    if int(candidates.groupby("query_position").size().max()) > 50:
        raise RuntimeError("A Step-2b candidate set exceeds the frozen budget")

    target_indices = panel["target_index"].to_numpy(dtype=np.int64)
    selected_validation = validation.iloc[target_indices].reset_index(drop=True)
    seeds = pd.DataFrame(
        {
            "query_position": panel["query_position"].to_numpy(dtype=np.int64),
            "target_index": target_indices,
            "seed_molecule_index": selected_validation["molecule_index"].to_numpy(
                dtype=np.int64
            ),
            "seed_hash": selected_validation["molecule_hash"].astype(str).to_numpy(),
            "seed_canonical_smiles": selected_validation["canonical_smiles"]
            .astype(str)
            .to_numpy(),
            "seed_scaffold": selected_validation["scaffold"]
            .fillna("")
            .astype(str)
            .to_numpy(),
            "seed_heavy_atoms_source": selected_validation["heavy_atoms"].to_numpy(
                dtype=np.int16
            ),
        }
    )
    assert_equal_series(seeds["seed_hash"], panel["target_hash"], "panel target hash")

    candidate_panel = candidates.merge(
        seeds[
            [
                "query_position",
                "target_index",
                "seed_hash",
                "seed_canonical_smiles",
                "seed_scaffold",
            ]
        ],
        how="left",
        on=["query_position", "target_index"],
        validate="many_to_one",
    )
    if candidate_panel["seed_hash"].isna().any():
        raise RuntimeError("Candidate rows do not align to the frozen seed panel")
    if not np.all(
        candidate_panel["condition_source_index"].to_numpy(dtype=np.int64)
        == candidate_panel["target_index"].to_numpy(dtype=np.int64)
    ):
        raise RuntimeError("Correct-condition rows do not use the target embedding")
    expected_exact = (
        candidate_panel["candidate_hash"].astype(str)
        == candidate_panel["seed_hash"].astype(str)
    ).astype(np.float32)
    if not np.array_equal(
        expected_exact.to_numpy(),
        candidate_panel["exact_target_identity"].to_numpy(dtype=np.float32),
    ):
        raise RuntimeError("Stored exact-target annotation is inconsistent")
    candidates = candidate_panel
    candidates.insert(0, "candidate_row_id", np.arange(len(candidates), dtype=np.int64))

    print("independently reapplying the frozen chemistry policy", flush=True)
    raw_universe = sorted(
        set(candidates["raw_smiles"].astype(str))
        | set(candidates["canonical_smiles"].astype(str))
        | set(seeds["seed_canonical_smiles"].astype(str))
    )
    resolved_config = load_json(paths["external_resolved_config"])
    raw_audit = audit_raw_smiles(
        raw_universe, resolved_config=resolved_config, workers=workers
    )
    atomic_write_parquet(
        step_root / "intermediate" / "raw_smiles_policy_audit.parquet",
        raw_audit,
        step_root,
    )
    candidate_audit = raw_audit.rename(
        columns={column: f"audit_{column}" for column in raw_audit.columns if column != "raw_smiles"}
    )
    candidates = candidates.merge(
        candidate_audit,
        how="left",
        on="raw_smiles",
        validate="many_to_one",
    )
    if candidates["audit_rdkit_valid"].isna().any():
        raise RuntimeError("A retained raw candidate was not independently audited")
    if not candidates["audit_rdkit_valid"].astype(bool).all():
        raise RuntimeError("A retained Step-2b candidate is no longer RDKit-valid")
    if not candidates["audit_policy_accepted"].astype(bool).all():
        raise RuntimeError("A retained Step-2b candidate is no longer policy-accepted")
    assert_equal_series(
        candidates["canonical_smiles"],
        candidates["audit_canonical_smiles"],
        "recomputed canonical SMILES",
    )
    assert_equal_series(
        candidates["candidate_hash"],
        candidates["audit_molecule_hash"],
        "recomputed candidate identity",
    )
    assert_equal_series(
        candidates["candidate_scaffold"],
        candidates["audit_scaffold"],
        "recomputed candidate scaffold",
    )
    if not np.array_equal(
        candidates["candidate_atom_count"].to_numpy(dtype=np.int64),
        candidates["audit_atom_count"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Recomputed candidate atom count differs from Step 2b")

    canonical_universe = set(candidates["canonical_smiles"].astype(str)) | set(
        seeds["seed_canonical_smiles"].astype(str)
    )
    structures = raw_audit.loc[
        raw_audit["raw_smiles"].isin(canonical_universe)
        & raw_audit["raw_equals_canonical"].astype(bool)
    ].copy()
    structures = structures.rename(columns={"raw_smiles": "canonical_input_smiles"})
    if len(structures) != len(canonical_universe):
        raise RuntimeError("Canonical structure audit is incomplete")
    if not structures["policy_accepted"].astype(bool).all():
        raise RuntimeError("A canonical seed/candidate fails the frozen policy")
    if structures["molecule_hash"].duplicated().any():
        raise RuntimeError("Canonical structure identities are not unique")
    structures = structures.sort_values("molecule_hash", ignore_index=True)
    structures.insert(0, "structure_index", np.arange(len(structures), dtype=np.int64))
    atomic_write_parquet(
        step_root / "intermediate" / "unique_molecules.parquet", structures, step_root
    )

    structure_lookup = structures.set_index("molecule_hash")
    candidates["candidate_structure_index"] = candidates["candidate_hash"].map(
        structure_lookup["structure_index"]
    )
    seeds["seed_structure_index"] = seeds["seed_hash"].map(
        structure_lookup["structure_index"]
    )
    if candidates["candidate_structure_index"].isna().any() or seeds[
        "seed_structure_index"
    ].isna().any():
        raise RuntimeError("Structure indices are incomplete")
    candidates["candidate_structure_index"] = candidates[
        "candidate_structure_index"
    ].astype(np.int64)
    seeds["seed_structure_index"] = seeds["seed_structure_index"].astype(np.int64)
    candidates = candidates.merge(
        seeds[["query_position", "seed_structure_index"]],
        on="query_position",
        how="left",
        validate="many_to_one",
    )

    descriptor_lookup = structures.set_index("molecule_hash")
    for field in DESCRIPTOR_FIELDS + ["element_counts_json"]:
        candidates[f"candidate_{field}"] = candidates["candidate_hash"].map(
            descriptor_lookup[field]
        )
        seeds[f"seed_{field}"] = seeds["seed_hash"].map(descriptor_lookup[field])
        candidates[f"seed_{field}"] = candidates["seed_hash"].map(
            descriptor_lookup[field]
        )
    if not np.array_equal(
        seeds["seed_heavy_atom_count"].to_numpy(dtype=np.int64),
        seeds["seed_heavy_atoms_source"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Seed heavy-atom descriptors differ from Step 1b")

    print(f"fragmenting {len(structures):,} unique seed/candidate molecules", flush=True)
    fragments, fragment_heavy, fragment_stats = fragment_molecules(
        structures["canonical_smiles"].astype(str).tolist(),
        settings=cfg["mmp"],
        workers=workers,
        progress_every=50_000,
    )
    if not np.array_equal(
        fragment_heavy.astype(np.int64),
        structures["heavy_atom_count"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Step-1b fragmentation parent sizes differ from descriptors")
    atomic_write_parquet(
        step_root / "intermediate" / "molecule_fragments.parquet", fragments, step_root
    )
    atomic_write_json(
        step_root / "intermediate" / "fragmentation_summary.json",
        fragment_stats,
        step_root,
    )

    candidates["is_seed_identity"] = (
        candidates["candidate_hash"].astype(str) == candidates["seed_hash"].astype(str)
    )
    candidates["is_genuine_nonseed"] = ~candidates["is_seed_identity"]
    mmp_pairs = candidates.loc[
        candidates["is_genuine_nonseed"],
        [
            "candidate_row_id",
            "query_position",
            "seed_structure_index",
            "candidate_structure_index",
        ],
    ].copy()
    print(f"matching one-cut MMPs for {len(mmp_pairs):,} non-seed candidates", flush=True)
    explanations = mmp_explanations(
        mmp_pairs,
        fragments,
        settings=cfg["mmp"],
        threads=int(cfg["execution"]["duckdb_threads"]),
        temporary_dir=step_root / "state" / "duckdb_tmp",
    )
    primary_columns = [
        "candidate_row_id",
        "core",
        "seed_substituent",
        "candidate_substituent",
        "core_heavy_atoms",
        "seed_substituent_heavy_atoms",
        "candidate_substituent_heavy_atoms",
        "seed_parent_heavy_atoms",
        "candidate_parent_heavy_atoms",
        "variable_heavy_atom_delta",
        "parent_heavy_atom_delta",
        "seed_to_candidate_transform",
        "undirected_transform",
        "mmp_edit_class",
        "mmp_explanation_count",
    ]
    if explanations.empty:
        primary = pd.DataFrame(columns=primary_columns)
    else:
        primary = explanations.loc[
            explanations["is_primary_explanation"].astype(bool), primary_columns
        ].copy()
        if primary["candidate_row_id"].duplicated().any():
            raise RuntimeError("Primary MMP explanation is not unique per candidate")
    candidates = candidates.merge(
        primary,
        how="left",
        on="candidate_row_id",
        validate="one_to_one",
    )
    candidates["mmp_explanation_count"] = candidates[
        "mmp_explanation_count"
    ].fillna(0).astype(np.int32)
    candidates["is_one_cut_mmp"] = candidates["mmp_explanation_count"] > 0
    candidates["mmp_explanation_ambiguous"] = candidates["mmp_explanation_count"] > 1

    candidates["seed_has_bemis_murcko_scaffold"] = candidates["seed_scaffold"].astype(
        str
    ) != ""
    candidates["candidate_has_bemis_murcko_scaffold"] = candidates[
        "candidate_scaffold"
    ].fillna("").astype(str) != ""
    candidates["same_scaffold_key"] = (
        candidates["candidate_scaffold"].fillna("").astype(str)
        == candidates["seed_scaffold"].fillna("").astype(str)
    )
    candidates["retains_nonempty_seed_scaffold"] = (
        candidates["seed_has_bemis_murcko_scaffold"]
        & candidates["same_scaffold_key"]
    )
    candidates["both_seed_and_candidate_acyclic"] = (
        ~candidates["seed_has_bemis_murcko_scaffold"]
        & ~candidates["candidate_has_bemis_murcko_scaffold"]
    )

    exact_spelling = (
        candidates["is_seed_identity"]
        & (
            candidates["raw_smiles"].astype(str)
            == candidates["seed_canonical_smiles"].astype(str)
        )
    )
    alternative_spelling = candidates["is_seed_identity"] & ~exact_spelling
    categories = np.full(len(candidates), "", dtype=object)
    categories[exact_spelling.to_numpy()] = "exact_seed_identity"
    categories[alternative_spelling.to_numpy()] = "same_identity_alternative_smiles"
    categories[
        (candidates["is_genuine_nonseed"] & candidates["is_one_cut_mmp"]).to_numpy()
    ] = "one_cut_mmp_derivative"
    categories[
        (
            candidates["is_genuine_nonseed"]
            & ~candidates["is_one_cut_mmp"]
            & candidates["retains_nonempty_seed_scaffold"]
        ).to_numpy()
    ] = "scaffold_preserving_non_mmp_analogue"
    categories[
        (
            candidates["is_genuine_nonseed"]
            & ~candidates["is_one_cut_mmp"]
            & candidates["both_seed_and_candidate_acyclic"]
        ).to_numpy()
    ] = "acyclic_non_mmp_analogue"
    categories[categories == ""] = "scaffold_changing_analogue"
    candidates["chemical_category"] = pd.Categorical(
        categories, categories=CATEGORY_ORDER, ordered=True
    )
    if candidates["chemical_category"].isna().any():
        raise RuntimeError("Candidate classification is incomplete")

    for field in DESCRIPTOR_FIELDS:
        candidates[f"delta_{field}"] = (
            candidates[f"candidate_{field}"].to_numpy(dtype=np.int64)
            - candidates[f"seed_{field}"].to_numpy(dtype=np.int64)
        )
    print("summarizing element-count and graph-descriptor changes", flush=True)
    candidates["element_count_delta_json"] = [
        element_delta(seed_value, candidate_value)
        for seed_value, candidate_value in zip(
            candidates["seed_element_counts_json"],
            candidates["candidate_element_counts_json"],
        )
    ]

    print("computing seed and within-set Morgan similarities", flush=True)
    morgan, diversity_seed, pairwise_arrays, stored_morgan_max_difference = (
        compute_morgan_and_diversity(
            candidates,
            seeds,
            radius=int(cfg["similarity"]["radius"]),
            bits=int(cfg["similarity"]["bits"]),
        )
    )
    candidates["morgan_similarity_to_seed_recomputed"] = morgan
    if stored_morgan_max_difference > 1e-6:
        raise RuntimeError(
            f"Recomputed Morgan similarity differs from Step 2b by {stored_morgan_max_difference}"
        )
    atomic_save_npz(
        step_root / "outputs" / "raw" / "within_set_pairwise_morgan.npz",
        step_root,
        **pairwise_arrays,
    )

    if not explanations.empty:
        explanation_context = candidates[
            [
                "candidate_row_id",
                "seed_hash",
                "seed_canonical_smiles",
                "candidate_hash",
                "canonical_smiles",
                "morgan_similarity_to_seed_recomputed",
            ]
        ]
        explanations = explanations.merge(
            explanation_context,
            how="left",
            on="candidate_row_id",
            validate="many_to_one",
        )
    atomic_write_parquet(
        step_root / "outputs" / "tables" / "mmp_explanations.parquet",
        explanations,
        step_root,
    )

    output_candidate_columns = [
        "candidate_row_id",
        "query_position",
        "target_index",
        "proposal_rank",
        "generator_kind",
        "raw_smiles",
        "canonical_smiles",
        "candidate_hash",
        "seed_canonical_smiles",
        "seed_hash",
        "audit_rdkit_valid",
        "audit_policy_accepted",
        "audit_raw_equals_canonical",
        "is_seed_identity",
        "is_genuine_nonseed",
        "chemical_category",
        "candidate_scaffold",
        "seed_scaffold",
        "seed_has_bemis_murcko_scaffold",
        "candidate_has_bemis_murcko_scaffold",
        "same_scaffold_key",
        "retains_nonempty_seed_scaffold",
        "both_seed_and_candidate_acyclic",
        "is_one_cut_mmp",
        "mmp_explanation_count",
        "mmp_explanation_ambiguous",
        "core",
        "seed_substituent",
        "candidate_substituent",
        "seed_to_candidate_transform",
        "undirected_transform",
        "mmp_edit_class",
        "morgan_similarity_to_seed_recomputed",
        "morgan_similarity_to_target",
        "latent_cosine_to_supplied_condition",
        "latent_l2_to_supplied_condition",
        "latent_relative_l2_to_supplied_condition",
        "element_count_delta_json",
    ]
    for prefix in ("seed", "candidate", "delta"):
        for field in DESCRIPTOR_FIELDS:
            name = f"{prefix}_{field}"
            if name not in output_candidate_columns:
                output_candidate_columns.append(name)
    candidate_output = candidates[output_candidate_columns].copy()
    candidate_output["chemical_category"] = candidate_output[
        "chemical_category"
    ].astype(str)
    atomic_write_parquet(
        step_root / "outputs" / "tables" / "candidate_characterization.parquet",
        candidate_output,
        step_root,
    )

    print("assembling per-seed and aggregate tables", flush=True)
    grouped = candidates.groupby("query_position", sort=False)
    seed_counts = grouped.agg(
        retained_candidate_count=("candidate_row_id", "size"),
        unique_retained_raw_smiles=("raw_smiles", "nunique"),
        unique_canonical_molecular_identities=("candidate_hash", "nunique"),
        exact_seed_identity_count=("is_seed_identity", "sum"),
        genuine_nonseed_candidate_count=("is_genuine_nonseed", "sum"),
        mmp_derivative_count=("is_one_cut_mmp", "sum"),
        ambiguous_mmp_candidate_count=("mmp_explanation_ambiguous", "sum"),
        raw_noncanonical_spelling_count=("audit_raw_equals_canonical", lambda values: int((~values.astype(bool)).sum())),
        distinct_scaffold_keys=("candidate_scaffold", "nunique"),
        same_scaffold_key_nonseed_count=(
            "same_scaffold_key",
            lambda values: int(
                values[candidates.loc[values.index, "is_genuine_nonseed"]].sum()
            ),
        ),
        retained_nonempty_seed_scaffold_nonseed_count=(
            "retains_nonempty_seed_scaffold",
            lambda values: int(
                values[candidates.loc[values.index, "is_genuine_nonseed"]].sum()
            ),
        ),
        both_acyclic_nonseed_count=(
            "both_seed_and_candidate_acyclic",
            lambda values: int(
                values[candidates.loc[values.index, "is_genuine_nonseed"]].sum()
            ),
        ),
    ).reset_index()
    distinct_nonempty_scaffolds = (
        candidates.loc[candidates["candidate_scaffold"].fillna("").astype(str) != ""]
        .groupby("query_position")["candidate_scaffold"]
        .nunique()
        .rename("distinct_nonempty_scaffolds")
        .reset_index()
    )
    categories_by_seed = (
        candidates.groupby(["query_position", "chemical_category"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
        .add_suffix("_count")
        .reset_index()
    )
    seed_output = seeds.merge(seed_counts, how="left", on="query_position").merge(
        distinct_nonempty_scaffolds, how="left", on="query_position"
    ).merge(categories_by_seed, how="left", on="query_position").merge(
        diversity_seed, how="left", on="query_position", validate="one_to_one"
    )
    generation_columns = [
        "query_position",
        "greedy_token_decoded",
        "greedy_rdkit_valid",
        "greedy_policy_accepted",
        "raw_non_greedy_draws",
        "raw_token_decoded",
        "raw_rdkit_valid",
        "raw_policy_accepted",
        "unique_policy_accepted_non_greedy",
        "retained_unique_candidate_count",
    ]
    seed_output = seed_output.merge(
        generation[generation_columns],
        how="left",
        on="query_position",
        validate="one_to_one",
    )
    integer_count_columns = [
        column
        for column in seed_output.columns
        if column.endswith("_count")
        or column
        in {
            "retained_candidate_count",
            "unique_retained_raw_smiles",
            "unique_canonical_molecular_identities",
            "distinct_scaffold_keys",
            "distinct_nonempty_scaffolds",
        }
    ]
    for column in integer_count_columns:
        seed_output[column] = seed_output[column].fillna(0).astype(np.int32)
    seed_output["candidate_capacity_fill_fraction"] = (
        seed_output["retained_candidate_count"]
        / float(cfg["candidate_population"]["requested_candidate_set_size"])
    )
    seed_output["mmp_fraction_among_genuine_nonseed"] = np.divide(
        seed_output["mmp_derivative_count"].to_numpy(dtype=np.float64),
        seed_output["genuine_nonseed_candidate_count"].to_numpy(dtype=np.float64),
        out=np.full(len(seed_output), np.nan, dtype=np.float64),
        where=seed_output["genuine_nonseed_candidate_count"].to_numpy(dtype=np.float64) > 0,
    )
    seed_output["accepted_beam_identity_redundancy_count"] = (
        seed_output["raw_policy_accepted"].astype(np.int64)
        - seed_output["unique_policy_accepted_non_greedy"].astype(np.int64)
    )
    atomic_write_parquet(
        step_root / "outputs" / "tables" / "seed_characterization.parquet",
        seed_output,
        step_root,
    )
    atomic_write_csv(
        step_root / "outputs" / "tables" / "seed_characterization.csv",
        seed_output,
        step_root,
    )

    category_counts = (
        candidates.groupby("chemical_category", observed=False)
        .agg(
            candidate_count=("candidate_row_id", "size"),
            seed_count=("query_position", "nunique"),
            unique_global_molecules=("candidate_hash", "nunique"),
        )
        .reindex(CATEGORY_ORDER, fill_value=0)
        .reset_index()
    )
    category_counts["fraction_of_retained_candidates"] = (
        category_counts["candidate_count"] / len(candidates)
    )
    nonseed_total = int(candidates["is_genuine_nonseed"].sum())
    category_counts["fraction_of_genuine_nonseed"] = np.where(
        category_counts["chemical_category"].isin(
            ["exact_seed_identity", "same_identity_alternative_smiles"]
        ),
        0.0,
        category_counts["candidate_count"] / max(nonseed_total, 1),
    )
    atomic_write_csv(
        step_root / "outputs" / "tables" / "candidate_category_counts.csv",
        category_counts,
        step_root,
    )

    threshold_rows = []
    for threshold in cfg["mmp"]["seed_coverage_thresholds"]:
        count = int((seed_output["mmp_derivative_count"] >= int(threshold)).sum())
        threshold_rows.append(
            {
                "minimum_mmp_derivatives": int(threshold),
                "seed_count": count,
                "seed_fraction": count / len(seed_output),
            }
        )
    mmp_thresholds = pd.DataFrame(threshold_rows)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "mmp_seed_coverage.csv",
        mmp_thresholds,
        step_root,
    )

    if primary.empty:
        transformation_table = pd.DataFrame(
            columns=[
                "seed_to_candidate_transform",
                "mmp_edit_class",
                "candidate_count",
                "seed_count",
            ]
        )
    else:
        mmp_candidates = candidates.loc[candidates["is_one_cut_mmp"]].copy()
        transformation_table = (
            mmp_candidates.groupby(
                [
                    "seed_to_candidate_transform",
                    "seed_substituent",
                    "candidate_substituent",
                    "mmp_edit_class",
                    "variable_heavy_atom_delta",
                ],
                dropna=False,
            )
            .agg(
                candidate_count=("candidate_row_id", "size"),
                seed_count=("query_position", "nunique"),
                unique_candidate_molecules=("candidate_hash", "nunique"),
                ambiguous_explanation_count=("mmp_explanation_ambiguous", "sum"),
                mean_morgan_to_seed=("morgan_similarity_to_seed_recomputed", "mean"),
                median_morgan_to_seed=("morgan_similarity_to_seed_recomputed", "median"),
                median_core_heavy_atoms=("core_heavy_atoms", "median"),
            )
            .reset_index()
            .sort_values(
                ["candidate_count", "seed_count", "seed_to_candidate_transform"],
                ascending=[False, False, True],
                ignore_index=True,
            )
        )
        transformation_table.insert(
            0, "frequency_rank", np.arange(1, len(transformation_table) + 1)
        )
    atomic_write_csv(
        step_root / "outputs" / "tables" / "mmp_transformation_counts.csv",
        transformation_table,
        step_root,
    )

    similarity_populations: dict[str, np.ndarray] = {
        "all_genuine_nonseed": candidates.loc[
            candidates["is_genuine_nonseed"], "morgan_similarity_to_seed_recomputed"
        ].to_numpy(dtype=np.float32)
    }
    for category in CATEGORY_ORDER[2:]:
        similarity_populations[category] = candidates.loc[
            candidates["chemical_category"].astype(str) == category,
            "morgan_similarity_to_seed_recomputed",
        ].to_numpy(dtype=np.float32)
    similarity_table = summary_frame(similarity_populations)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "seed_candidate_similarity_summary.csv",
        similarity_table,
        step_root,
    )

    descriptor_rows: list[dict[str, Any]] = []
    nonmmp = candidates.loc[
        candidates["is_genuine_nonseed"] & ~candidates["is_one_cut_mmp"]
    ]
    for category, frame in [("all_non_mmp", nonmmp)] + [
        (category, nonmmp.loc[nonmmp["chemical_category"].astype(str) == category])
        for category in CATEGORY_ORDER[3:]
    ]:
        for field in DESCRIPTOR_FIELDS:
            descriptor_rows.append(
                {
                    "population": category,
                    "descriptor_delta": field,
                    **numeric_summary(frame[f"delta_{field}"].to_numpy(dtype=np.float64)),
                }
            )
    descriptor_summary = pd.DataFrame(descriptor_rows)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "non_mmp_graph_delta_summary.csv",
        descriptor_summary,
        step_root,
    )
    pattern_fields = [
        "delta_heavy_atom_count",
        "delta_bond_count",
        "delta_ring_count",
        "delta_heteroatom_count",
        "delta_formal_charge",
        "element_count_delta_json",
    ]
    pattern_table = (
        nonmmp.groupby(["chemical_category", *pattern_fields], observed=True)
        .agg(candidate_count=("candidate_row_id", "size"), seed_count=("query_position", "nunique"))
        .reset_index()
        .sort_values(["candidate_count", "seed_count"], ascending=False, ignore_index=True)
    )
    atomic_write_csv(
        step_root / "outputs" / "tables" / "non_mmp_descriptor_delta_patterns.csv",
        pattern_table,
        step_root,
    )

    histogram = pd.concat(
        [
            histogram_table(
                pairwise_arrays["all_candidate_pairwise_morgan"],
                bins=int(cfg["similarity"]["pairwise_histogram_bins"]),
                population="all_unique_retained_candidates",
            ),
            histogram_table(
                pairwise_arrays["nonseed_candidate_pairwise_morgan"],
                bins=int(cfg["similarity"]["pairwise_histogram_bins"]),
                population="genuine_nonseed_candidates",
            ),
        ],
        ignore_index=True,
    )
    atomic_write_csv(
        step_root / "outputs" / "tables" / "within_set_pairwise_morgan_histogram.csv",
        histogram,
        step_root,
    )

    all_candidate_pairs = pairwise_arrays["all_candidate_pairwise_morgan"]
    nonseed_pairs = pairwise_arrays["nonseed_candidate_pairwise_morgan"]
    nonseed_candidates = candidates.loc[candidates["is_genuine_nonseed"]]
    scaffold_eligible = nonseed_candidates.loc[
        nonseed_candidates["seed_has_bemis_murcko_scaffold"]
    ]
    raw_beam_draws = int(generation["raw_non_greedy_draws"].sum())
    raw_beam_accepted = int(generation["raw_policy_accepted"].sum())
    raw_beam_unique_non_greedy = int(
        generation["unique_policy_accepted_non_greedy"].sum()
    )
    retained_seed_identity = candidates.loc[candidates["is_seed_identity"]]
    exact_seed_spelling_count = int(
        (retained_seed_identity["raw_smiles"].astype(str)
         == retained_seed_identity["seed_canonical_smiles"].astype(str)).sum()
    )
    alternative_seed_spelling_count = int(len(retained_seed_identity) - exact_seed_spelling_count)
    nonseed_similarity = nonseed_candidates[
        "morgan_similarity_to_seed_recomputed"
    ].to_numpy(dtype=np.float32)
    scaffold_retention_rate = (
        float(scaffold_eligible["same_scaffold_key"].mean())
        if len(scaffold_eligible)
        else math.nan
    )

    gates_cfg = cfg["bounded_conclusion"]
    gate_values = {
        "fraction_seeds_with_5_genuine_nonseed": float(
            (seed_output["genuine_nonseed_candidate_count"] >= 5).mean()
        ),
        "fraction_seeds_with_1_mmp": float(
            (seed_output["mmp_derivative_count"] >= 1).mean()
        ),
        "nonempty_seed_scaffold_retention": scaffold_retention_rate,
        "global_nonseed_median_morgan": float(np.median(nonseed_similarity)),
        "median_unique_nonseed_per_seed": float(
            np.median(seed_output["genuine_nonseed_candidate_count"])
        ),
        "global_pairwise_mean_morgan": float(np.mean(nonseed_pairs)),
    }
    gates = {
        "genuine_nonseed_yield": gate_values["fraction_seeds_with_5_genuine_nonseed"]
        >= float(gates_cfg["minimum_fraction_seeds_with_5_genuine_nonseed"]),
        "mmp_seed_coverage": gate_values["fraction_seeds_with_1_mmp"]
        >= float(gates_cfg["minimum_fraction_seeds_with_1_mmp"]),
        "scaffold_locality": gate_values["nonempty_seed_scaffold_retention"]
        >= float(gates_cfg["minimum_nonempty_seed_scaffold_retention"]),
        "seed_similarity_local_not_trivial": (
            float(gates_cfg["minimum_global_nonseed_median_morgan"])
            <= gate_values["global_nonseed_median_morgan"]
            <= float(gates_cfg["maximum_global_nonseed_median_morgan"])
        ),
        "unique_nonseed_yield": gate_values["median_unique_nonseed_per_seed"]
        >= float(gates_cfg["minimum_median_unique_nonseed_per_seed"]),
        "within_set_nontrivial_diversity": gate_values["global_pairwise_mean_morgan"]
        <= float(gates_cfg["maximum_global_pairwise_mean_morgan"]),
    }
    if all(gates.values()):
        conclusion = gates_cfg["label_if_all_gates_pass"]
    elif all(value for key, value in gates.items() if key != "mmp_seed_coverage"):
        conclusion = gates_cfg["label_if_yield_and_locality_pass_but_mmp_gate_fails"]
    else:
        conclusion = gates_cfg["label_otherwise"]

    summary = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "status": "analysis_complete",
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_time_seconds": float(time.monotonic() - wall_started),
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
            "workers": workers,
            "gpu_used": False,
            "model_executed": False,
        },
        "scientific_boundary": cfg["scientific_boundary"],
        "population": {
            "seeds": len(seed_output),
            "seeds_with_at_least_one_retained_candidate": int(
                (seed_output["retained_candidate_count"] > 0).sum()
            ),
            "seeds_without_retained_candidate": int(
                (seed_output["retained_candidate_count"] == 0).sum()
            ),
            "retained_candidate_rows": len(candidates),
            "nominal_candidate_slots": int(len(seed_output) * 50),
            "globally_unique_candidate_molecules": int(candidates["candidate_hash"].nunique()),
            "globally_unique_seed_or_candidate_molecules": len(structures),
        },
        "validity_and_uniqueness": {
            "retained_rdkit_valid_rate_recomputed": float(
                candidates["audit_rdkit_valid"].astype(bool).mean()
            ),
            "retained_policy_accepted_rate_recomputed": float(
                candidates["audit_policy_accepted"].astype(bool).mean()
            ),
            "retained_valid_unique_identity_slots_fraction_of_nominal_50": float(
                len(candidates) / (len(seed_output) * 50)
            ),
            "mean_unique_canonical_identities_per_seed": float(
                seed_output["unique_canonical_molecular_identities"].mean()
            ),
            "median_unique_canonical_identities_per_seed": float(
                seed_output["unique_canonical_molecular_identities"].median()
            ),
            "mean_unique_raw_smiles_per_seed_in_retained_sets": float(
                seed_output["unique_retained_raw_smiles"].mean()
            ),
            "raw_beam_hypotheses": raw_beam_draws,
            "raw_beam_rdkit_valid_rate": float(
                generation["raw_rdkit_valid"].sum() / raw_beam_draws
            ),
            "raw_beam_policy_accepted_rate": float(raw_beam_accepted / raw_beam_draws),
            "greedy_rdkit_valid_rate": float(generation["greedy_rdkit_valid"].mean()),
            "greedy_policy_accepted_rate": float(
                generation["greedy_policy_accepted"].mean()
            ),
            "accepted_beam_identity_redundancy_count": int(
                raw_beam_accepted - raw_beam_unique_non_greedy
            ),
            "accepted_beam_identity_redundancy_fraction": float(
                (raw_beam_accepted - raw_beam_unique_non_greedy)
                / max(raw_beam_accepted, 1)
            ),
            "seed_identity_candidate_count": len(retained_seed_identity),
            "exact_seed_canonical_spelling_count": exact_seed_spelling_count,
            "alternative_smiles_same_seed_identity_count": alternative_seed_spelling_count,
            "individual_discarded_raw_strings_available": False,
            "raw_string_duplicate_vs_alternative_spelling_limitation": cfg[
                "candidate_population"
            ]["raw_hypothesis_limitation"],
        },
        "category_counts": {
            str(row["chemical_category"]): int(row["candidate_count"])
            for _, row in category_counts.iterrows()
        },
        "mmp_derivatives": {
            "genuine_nonseed_candidates": nonseed_total,
            "one_cut_mmp_candidates": int(candidates["is_one_cut_mmp"].sum()),
            "global_mmp_fraction_among_genuine_nonseed": float(
                candidates["is_one_cut_mmp"].sum() / max(nonseed_total, 1)
            ),
            "seeds_with_at_least_one_mmp": int(
                (seed_output["mmp_derivative_count"] >= 1).sum()
            ),
            "seeds_with_at_least_one_mmp_fraction": float(
                (seed_output["mmp_derivative_count"] >= 1).mean()
            ),
            "mmp_candidates_with_multiple_explanations": int(
                candidates["mmp_explanation_ambiguous"].sum()
            ),
            "distinct_primary_directional_transformations": int(
                candidates.loc[
                    candidates["is_one_cut_mmp"], "seed_to_candidate_transform"
                ].nunique()
            ),
            "fragmentation": fragment_stats,
        },
        "scaffold_preservation": {
            "genuine_nonseed_with_nonempty_seed_scaffold": len(scaffold_eligible),
            "nonempty_seed_scaffold_retention_rate": scaffold_retention_rate,
            "all_nonseed_exact_scaffold_key_rate_including_empty": float(
                nonseed_candidates["same_scaffold_key"].mean()
            ),
            "both_seed_and_candidate_acyclic_rate_among_nonseed": float(
                nonseed_candidates["both_seed_and_candidate_acyclic"].mean()
            ),
            "median_distinct_scaffold_keys_per_seed": float(
                seed_output["distinct_scaffold_keys"].median()
            ),
            "median_distinct_nonempty_scaffolds_per_seed": float(
                seed_output["distinct_nonempty_scaffolds"].median()
            ),
        },
        "seed_candidate_similarity": {
            "all_genuine_nonseed": numeric_summary(nonseed_similarity),
            "by_category_table": "outputs/tables/seed_candidate_similarity_summary.csv",
        },
        "within_set_diversity": {
            "all_candidate_pair_count": len(all_candidate_pairs),
            "genuine_nonseed_pair_count": len(nonseed_pairs),
            "all_candidates": numeric_summary(all_candidate_pairs),
            "genuine_nonseed": numeric_summary(nonseed_pairs),
            "fraction_all_candidate_pairs_tanimoto_ge_0_90": float(
                np.mean(all_candidate_pairs >= 0.90)
            ),
            "fraction_distinct_identity_pairs_tanimoto_eq_1": float(
                np.mean(all_candidate_pairs == 1.0)
            ),
        },
        "morgan_recomputation_max_absolute_difference": stored_morgan_max_difference,
        "bounded_conclusion": {
            "classification": conclusion,
            "gate_values": gate_values,
            "gates": gates,
            "scope": "unperturbed frozen Step-2b correct-condition candidate sets only",
            "does_not_establish": [
                "controllable latent edits",
                "MMP-direction generation",
                "novelty or synthesizability",
                "activity improvement",
            ],
        },
        "input_sha256": input_hashes,
    }
    atomic_write_json(step_root / "outputs" / "study_summary.json", summary, step_root)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "summary_metrics.csv",
        scalar_metric_rows(summary),
        step_root,
    )

    output_paths = {
        "candidate_characterization": step_root
        / "outputs"
        / "tables"
        / "candidate_characterization.parquet",
        "seed_characterization": step_root
        / "outputs"
        / "tables"
        / "seed_characterization.parquet",
        "mmp_explanations": step_root
        / "outputs"
        / "tables"
        / "mmp_explanations.parquet",
        "mmp_transformations": step_root
        / "outputs"
        / "tables"
        / "mmp_transformation_counts.csv",
        "pairwise_values": step_root
        / "outputs"
        / "raw"
        / "within_set_pairwise_morgan.npz",
        "study_summary": step_root / "outputs" / "study_summary.json",
    }
    resolve_manifest_inputs(repo_root, step_root)
    seal = {
        "schema_version": 1,
        "status": "analysis_complete",
        "completed_at": summary["finished_at"],
        "seed_rows": len(seed_output),
        "candidate_rows": len(candidates),
        "model_executed": False,
        "candidate_regeneration": False,
        "latent_perturbation": False,
        "mmp_directed_generation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
        "input_sha256": input_hashes,
        "outputs": {
            name: {
                "path": str(path.relative_to(step_root)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in output_paths.items()
        },
    }
    atomic_write_json(step_root / "state" / "ANALYSIS_COMPLETE.json", seal, step_root)
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
