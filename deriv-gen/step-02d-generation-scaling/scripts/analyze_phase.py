#!/usr/bin/env python3
"""Chemically characterize nested candidate budgets for one Step-2d phase."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import (
    STEP1B_ROOT,
    STEP2C_ROOT,
    STEP_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    load_json,
    numeric_summary,
    protocol,
    resolve_manifest_inputs,
    sha256_file,
    stable_digest,
    utc_now,
)


for source in (STEP1B_ROOT / "scripts", STEP2C_ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from audit_core import mmp_explanations  # noqa: E402
from scaled_common import fragment_molecules  # noqa: E402
from safe_policy_audit import audit_raw_smiles  # noqa: E402


RDLogger.DisableLog("rdApp.*")
CATEGORY_ORDER = [
    "exact_seed_identity",
    "one_cut_mmp_derivative",
    "scaffold_preserving_non_mmp_analogue",
    "acyclic_non_mmp_analogue",
    "scaffold_changing_analogue",
]
COUNT_FIELDS = [
    "unique_accepted_identity_count",
    "genuine_nonseed_count",
    "mmp_derivative_count",
    "same_scaffold_non_mmp_count",
    "retains_nonempty_seed_scaffold_count",
    "scaffold_changing_count",
    "acyclic_non_mmp_count",
    "novel_identity_count",
    "novel_genuine_nonseed_count",
    "novel_mmp_count",
    "novel_same_scaffold_non_mmp_count",
    "novel_scaffold_changing_count",
    "novel_useful_local_count",
    "distinct_scaffold_count",
    "distinct_nonempty_scaffold_count",
]


def safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def phase_strategies(cfg: dict[str, Any], root: Path, phase: str) -> list[str]:
    if phase == "development":
        return [value["name"] for value in cfg["generation"]["development_strategies"]]
    frozen = load_json(root / "state" / "STRATEGY_FROZEN.json")
    return [str(frozen["selected_strategy"]["name"])]


def proposal_view(connection: duckdb.DuckDBPyConnection, root: Path, phase: str) -> None:
    pattern = str(root / "outputs" / "raw" / phase / "proposals-*.parquet").replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE VIEW proposals AS SELECT * FROM read_parquet('{pattern}')"
    )


def audit_and_extract(
    root: Path,
    phase: str,
    panel: pd.DataFrame,
    budgets: list[int],
    resolved_config: dict[str, Any],
    workers: int,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={threads}")
        connection.execute("SET memory_limit='250GB'")
        temporary = str(root / "state" / "duckdb_tmp").replace("'", "''")
        (root / "state" / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory='{temporary}'")
        proposal_view(connection, root, phase)
        connection.register(
            "seed_strings",
            panel[["seed_canonical_smiles"]].rename(
                columns={"seed_canonical_smiles": "raw_smiles"}
            ),
        )
        raw = connection.execute(
            """
            SELECT raw_smiles FROM proposals
            UNION
            SELECT raw_smiles FROM seed_strings
            ORDER BY raw_smiles
            """
        ).fetchdf()
        values = raw["raw_smiles"].fillna("").astype(str).tolist()
        print(f"  {phase}: auditing {len(values):,} unique raw strings", flush=True)
        audit = audit_raw_smiles(
            values, resolved_config=resolved_config, workers=int(workers)
        )
        audit_path = root / "intermediate" / f"{phase}_raw_smiles_policy_audit.parquet"
        atomic_write_parquet(audit_path, audit, root)
        connection.register("policy_audit", audit)
        connection.register("budgets", pd.DataFrame({"budget": budgets}))
        raw_metrics = connection.execute(
            """
            SELECT
                p.strategy,
                p.query_position::INTEGER AS query_position,
                b.budget::INTEGER AS budget,
                count(*)::BIGINT AS raw_slot_count,
                count(*) FILTER (WHERE p.token_error = '')::BIGINT
                    AS token_decoded_count,
                count(*) FILTER (WHERE a.rdkit_valid)::BIGINT
                    AS rdkit_valid_count,
                count(*) FILTER (WHERE a.policy_accepted)::BIGINT
                    AS policy_accepted_count,
                count(DISTINCT p.raw_smiles) FILTER (WHERE p.token_error = '')::BIGINT
                    AS unique_decoded_raw_smiles_count,
                count(DISTINCT p.raw_smiles) FILTER (WHERE a.rdkit_valid)::BIGINT
                    AS unique_valid_raw_smiles_count,
                count(DISTINCT p.raw_smiles) FILTER (WHERE a.policy_accepted)::BIGINT
                    AS unique_policy_accepted_raw_smiles_count,
                count(DISTINCT a.molecule_hash) FILTER (WHERE a.policy_accepted)::BIGINT
                    AS unique_policy_accepted_identity_count_raw,
                count(*) FILTER (
                    WHERE a.policy_accepted AND a.molecule_hash = p.target_hash
                )::BIGINT AS seed_identity_raw_count,
                count(*) FILTER (
                    WHERE a.policy_accepted AND NOT a.raw_equals_canonical
                )::BIGINT AS noncanonical_accepted_raw_count
            FROM proposals AS p
            CROSS JOIN budgets AS b
            LEFT JOIN policy_audit AS a USING (raw_smiles)
            WHERE p.proposal_rank <= b.budget
            GROUP BY p.strategy, p.query_position, b.budget
            ORDER BY p.strategy, p.query_position, b.budget
            """
        ).fetchdf()
        candidates = connection.execute(
            """
            SELECT
                p.strategy,
                p.query_position::INTEGER AS query_position,
                p.target_index::BIGINT AS target_index,
                p.target_hash,
                p.proposal_rank::INTEGER AS first_proposal_rank,
                p.source_kind AS first_source_kind,
                p.source_rank::INTEGER AS first_source_rank,
                p.raw_smiles AS first_raw_smiles,
                a.canonical_smiles,
                a.molecule_hash AS candidate_hash,
                a.scaffold AS candidate_scaffold,
                a.atom_count::INTEGER AS candidate_atom_count,
                a.heavy_atom_count::INTEGER AS candidate_heavy_atom_count,
                a.raw_equals_canonical AS first_raw_equals_canonical
            FROM proposals AS p
            INNER JOIN policy_audit AS a USING (raw_smiles)
            WHERE a.policy_accepted
            QUALIFY row_number() OVER (
                PARTITION BY p.strategy, p.query_position, a.molecule_hash
                ORDER BY p.proposal_rank, p.source_kind, p.source_rank
            ) = 1
            ORDER BY p.strategy, p.query_position, p.proposal_rank, a.molecule_hash
            """
        ).fetchdf()
    finally:
        connection.close()
    if not (raw_metrics["raw_slot_count"] == raw_metrics["budget"]).all():
        raise RuntimeError("Nested raw-budget cardinality changed")
    return audit, raw_metrics, candidates


def characterize_pairs(
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    audit: pd.DataFrame,
    cfg: dict[str, Any],
    root: Path,
    phase: str,
    workers: int,
    threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Any]]:
    seed_columns = [
        "query_position",
        "target_hash",
        "seed_canonical_smiles",
        "seed_scaffold",
        "seed_heavy_atoms",
    ]
    seeds = panel[seed_columns].rename(columns={"target_hash": "seed_hash"}).copy()
    if candidates.empty:
        raise RuntimeError("No policy-accepted candidates")
    candidates = candidates.merge(seeds, on="query_position", how="left", validate="many_to_one")
    if not (candidates["target_hash"] == candidates["seed_hash"]).all():
        raise RuntimeError("Candidate/seed identity alignment changed")

    candidate_structures = candidates[
        [
            "candidate_hash",
            "canonical_smiles",
            "candidate_scaffold",
            "candidate_heavy_atom_count",
        ]
    ].rename(
        columns={
            "candidate_hash": "molecule_hash",
            "candidate_scaffold": "scaffold",
            "candidate_heavy_atom_count": "heavy_atom_count",
        }
    )
    seed_structures = seeds[
        ["seed_hash", "seed_canonical_smiles", "seed_scaffold", "seed_heavy_atoms"]
    ].rename(
        columns={
            "seed_hash": "molecule_hash",
            "seed_canonical_smiles": "canonical_smiles",
            "seed_scaffold": "scaffold",
            "seed_heavy_atoms": "heavy_atom_count",
        }
    )
    structures = pd.concat([candidate_structures, seed_structures], ignore_index=True)
    consistency = structures.groupby("molecule_hash").agg(
        smiles_values=("canonical_smiles", "nunique"),
        scaffold_values=("scaffold", "nunique"),
        heavy_values=("heavy_atom_count", "nunique"),
    )
    if int(consistency.to_numpy().max()) != 1:
        raise RuntimeError("A molecular identity has inconsistent chemistry metadata")
    structures = (
        structures.drop_duplicates("molecule_hash")
        .sort_values("molecule_hash", ignore_index=True)
    )
    structures.insert(0, "structure_index", np.arange(len(structures), dtype=np.int64))
    atomic_write_parquet(
        root / "intermediate" / f"{phase}_unique_molecules.parquet", structures, root
    )
    print(f"  {phase}: fragmenting {len(structures):,} unique molecules", flush=True)
    fragments, parent_heavy, fragmentation_stats = fragment_molecules(
        structures["canonical_smiles"].astype(str).tolist(),
        settings=cfg["mmp"],
        workers=int(workers),
        progress_every=50_000,
    )
    if not np.array_equal(
        parent_heavy.astype(np.int64), structures["heavy_atom_count"].to_numpy(dtype=np.int64)
    ):
        raise RuntimeError("Step-1b fragmentation heavy-atom counts changed")
    atomic_write_parquet(
        root / "intermediate" / f"{phase}_molecule_fragments.parquet", fragments, root
    )
    atomic_write_json(
        root / "intermediate" / f"{phase}_fragmentation_summary.json",
        fragmentation_stats,
        root,
    )

    structure_index = structures.set_index("molecule_hash")["structure_index"]
    pair_chemistry = candidates[
        [
            "query_position",
            "candidate_hash",
            "canonical_smiles",
            "candidate_scaffold",
            "candidate_heavy_atom_count",
            "seed_hash",
            "seed_canonical_smiles",
            "seed_scaffold",
        ]
    ].drop_duplicates(["query_position", "candidate_hash"])
    pair_chemistry = pair_chemistry.sort_values(
        ["query_position", "candidate_hash"], ignore_index=True
    )
    pair_chemistry.insert(0, "candidate_row_id", np.arange(len(pair_chemistry), dtype=np.int64))
    pair_chemistry["candidate_structure_index"] = pair_chemistry["candidate_hash"].map(structure_index).astype(np.int64)
    pair_chemistry["seed_structure_index"] = pair_chemistry["seed_hash"].map(structure_index).astype(np.int64)
    pair_chemistry["is_seed_identity"] = pair_chemistry["candidate_hash"] == pair_chemistry["seed_hash"]
    mmp_pairs = pair_chemistry.loc[
        ~pair_chemistry["is_seed_identity"],
        [
            "candidate_row_id",
            "query_position",
            "seed_structure_index",
            "candidate_structure_index",
        ],
    ]
    print(f"  {phase}: matching {len(mmp_pairs):,} unique seed-candidate pairs", flush=True)
    explanations = mmp_explanations(
        mmp_pairs,
        fragments,
        settings=cfg["mmp"],
        threads=int(threads),
        temporary_dir=root / "state" / "duckdb_tmp" / f"mmp_{phase}",
    )
    primary_columns = [
        "candidate_row_id",
        "core",
        "seed_substituent",
        "candidate_substituent",
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
    pair_chemistry = pair_chemistry.merge(
        primary, on="candidate_row_id", how="left", validate="one_to_one"
    )
    pair_chemistry["mmp_explanation_count"] = pair_chemistry[
        "mmp_explanation_count"
    ].fillna(0).astype(np.int32)
    pair_chemistry["is_one_cut_mmp"] = pair_chemistry["mmp_explanation_count"] > 0

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg["similarity"]["radius"]),
        fpSize=int(cfg["similarity"]["bits"]),
        includeChirality=False,
    )
    fingerprints = []
    for smiles in structures["canonical_smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Canonical molecule failed to parse: {smiles}")
        fingerprints.append(generator.GetFingerprint(molecule))
    pair_similarity = np.full(len(pair_chemistry), np.nan, dtype=np.float32)
    for query_position, group in pair_chemistry.groupby("query_position", sort=False):
        seed_index = int(group["seed_structure_index"].iloc[0])
        candidate_indices = group["candidate_structure_index"].to_numpy(dtype=np.int64)
        pair_similarity[group.index.to_numpy()] = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[seed_index], [fingerprints[index] for index in candidate_indices]
            ),
            dtype=np.float32,
        )
    pair_chemistry["morgan_similarity_to_seed"] = pair_similarity
    pair_chemistry["seed_has_scaffold"] = pair_chemistry["seed_scaffold"].fillna("").astype(str) != ""
    pair_chemistry["candidate_has_scaffold"] = pair_chemistry["candidate_scaffold"].fillna("").astype(str) != ""
    pair_chemistry["retains_nonempty_seed_scaffold"] = (
        pair_chemistry["seed_has_scaffold"]
        & (pair_chemistry["candidate_scaffold"].fillna("").astype(str) == pair_chemistry["seed_scaffold"].fillna("").astype(str))
    )
    pair_chemistry["both_acyclic"] = (
        ~pair_chemistry["seed_has_scaffold"] & ~pair_chemistry["candidate_has_scaffold"]
    )
    categories = np.full(len(pair_chemistry), "scaffold_changing_analogue", dtype=object)
    categories[pair_chemistry["is_seed_identity"].to_numpy()] = "exact_seed_identity"
    categories[(~pair_chemistry["is_seed_identity"] & pair_chemistry["is_one_cut_mmp"]).to_numpy()] = "one_cut_mmp_derivative"
    categories[(~pair_chemistry["is_seed_identity"] & ~pair_chemistry["is_one_cut_mmp"] & pair_chemistry["retains_nonempty_seed_scaffold"]).to_numpy()] = "scaffold_preserving_non_mmp_analogue"
    categories[(~pair_chemistry["is_seed_identity"] & ~pair_chemistry["is_one_cut_mmp"] & pair_chemistry["both_acyclic"]).to_numpy()] = "acyclic_non_mmp_analogue"
    pair_chemistry["chemical_category"] = categories
    atomic_write_parquet(
        root / "outputs" / "tables" / f"{phase}_mmp_explanations.parquet",
        explanations,
        root,
    )
    return candidates, pair_chemistry, structures, fingerprints


def sampled_pairwise(
    indices: np.ndarray,
    fingerprints: list[Any],
    maximum_pairs: int,
    seed: int,
) -> tuple[np.ndarray, bool, int]:
    n = len(indices)
    total = n * (n - 1) // 2
    if total <= 0:
        return np.empty(0, dtype=np.float32), True, total
    if total <= maximum_pairs:
        pairs = [(first, second) for first in range(n - 1) for second in range(first + 1, n)]
        exact = True
    else:
        rng = np.random.default_rng(seed)
        selected: set[tuple[int, int]] = set()
        while len(selected) < maximum_pairs:
            draws = rng.integers(0, n, size=(maximum_pairs, 2))
            for first, second in draws:
                if first == second:
                    continue
                if first > second:
                    first, second = second, first
                selected.add((int(first), int(second)))
                if len(selected) == maximum_pairs:
                    break
        pairs = sorted(selected)
        exact = False
    grouped: dict[int, list[int]] = defaultdict(list)
    for first, second in pairs:
        grouped[first].append(second)
    values: list[float] = []
    for first, seconds in grouped.items():
        values.extend(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[int(indices[first])],
                [fingerprints[int(indices[second])] for second in seconds],
            )
        )
    return np.asarray(values, dtype=np.float32), exact, total


def seed_budget_tables(
    candidates: pd.DataFrame,
    pair_chemistry: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    training_hashes: set[str],
    fingerprints: list[Any],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotation_columns = [
        "query_position",
        "candidate_hash",
        "candidate_structure_index",
        "is_seed_identity",
        "is_one_cut_mmp",
        "retains_nonempty_seed_scaffold",
        "seed_has_scaffold",
        "both_acyclic",
        "chemical_category",
        "morgan_similarity_to_seed",
        "core",
        "seed_substituent",
        "candidate_substituent",
        "seed_to_candidate_transform",
        "undirected_transform",
        "mmp_edit_class",
        "mmp_explanation_count",
    ]
    candidates = candidates.merge(
        pair_chemistry[annotation_columns],
        on=["query_position", "candidate_hash"],
        how="left",
        validate="many_to_one",
    )
    if candidates["chemical_category"].isna().any():
        raise RuntimeError("Candidate chemistry annotation is incomplete")
    candidates["is_genuine_nonseed"] = ~candidates["is_seed_identity"].astype(bool)
    candidates["is_novel_to_decoder_training"] = ~candidates["candidate_hash"].isin(training_hashes)
    candidates["is_useful_local"] = candidates["is_genuine_nonseed"] & (
        candidates["is_one_cut_mmp"].astype(bool)
        | candidates["retains_nonempty_seed_scaffold"].astype(bool)
    )
    candidates["is_novel_useful_local"] = (
        candidates["is_novel_to_decoder_training"] & candidates["is_useful_local"]
    )
    candidates = candidates.sort_values(
        ["strategy", "query_position", "first_proposal_rank"], ignore_index=True
    )

    groups = {
        (str(strategy), int(query)): group.reset_index(drop=True)
        for (strategy, query), group in candidates.groupby(
            ["strategy", "query_position"], sort=False
        )
    }
    metric_rows: list[dict[str, Any]] = []
    histogram_counts: dict[tuple[str, int], np.ndarray] = defaultdict(
        lambda: np.zeros(int(cfg["similarity"]["histogram_bins"]), dtype=np.int64)
    )
    for raw in raw_metrics.itertuples(index=False):
        strategy = str(raw.strategy)
        query = int(raw.query_position)
        budget = int(raw.budget)
        group = groups.get((strategy, query))
        if group is None:
            current = candidates.iloc[0:0]
        else:
            stop = int(np.searchsorted(group["first_proposal_rank"].to_numpy(), budget, side="right"))
            current = group.iloc[:stop]
        genuine = current["is_genuine_nonseed"].to_numpy(dtype=bool)
        mmp = current["is_one_cut_mmp"].to_numpy(dtype=bool) & genuine
        retained = current["retains_nonempty_seed_scaffold"].to_numpy(dtype=bool) & genuine
        seed_has = current["seed_has_scaffold"].to_numpy(dtype=bool) & genuine
        novel = current["is_novel_to_decoder_training"].to_numpy(dtype=bool)
        same_nonmmp = retained & ~mmp
        scaffold_changing = (
            current["chemical_category"].astype(str).to_numpy() == "scaffold_changing_analogue"
        )
        acyclic = (
            current["chemical_category"].astype(str).to_numpy() == "acyclic_non_mmp_analogue"
        )
        useful = current["is_useful_local"].to_numpy(dtype=bool)
        structure_indices = current["candidate_structure_index"].to_numpy(dtype=np.int64)
        diversity, exact, possible_pairs = sampled_pairwise(
            structure_indices,
            fingerprints,
            int(cfg["similarity"]["maximum_pairwise_comparisons_per_seed_budget"]),
            int(stable_digest(cfg["seed"], strategy, query, budget, "within")[:16], 16),
        )
        if len(diversity):
            histogram_counts[(strategy, budget)] += np.histogram(
                diversity,
                bins=int(cfg["similarity"]["histogram_bins"]),
                range=(0.0, 1.0),
            )[0]
        diversity_summary = numeric_summary(diversity)
        nonseed_similarity = current.loc[
            current["is_genuine_nonseed"], "morgan_similarity_to_seed"
        ].to_numpy(dtype=np.float64)
        row = raw._asdict()
        row.update(
            {
                "unique_accepted_identity_count": len(current),
                "accepted_identity_redundancy_count": int(raw.policy_accepted_count) - len(current),
                "seed_identity_unique_count": int(current["is_seed_identity"].sum()),
                "genuine_nonseed_count": int(genuine.sum()),
                "mmp_derivative_count": int(mmp.sum()),
                "same_scaffold_non_mmp_count": int(same_nonmmp.sum()),
                "retains_nonempty_seed_scaffold_count": int(retained.sum()),
                "scaffold_eligible_nonseed_count": int(seed_has.sum()),
                "scaffold_changing_count": int(scaffold_changing.sum()),
                "acyclic_non_mmp_count": int(acyclic.sum()),
                "novel_identity_count": int(novel.sum()),
                "novel_genuine_nonseed_count": int((novel & genuine).sum()),
                "novel_mmp_count": int((novel & mmp).sum()),
                "novel_same_scaffold_non_mmp_count": int((novel & same_nonmmp).sum()),
                "novel_scaffold_changing_count": int((novel & scaffold_changing).sum()),
                "novel_useful_local_count": int((novel & useful).sum()),
                "distinct_scaffold_count": int(current["candidate_scaffold"].fillna("").astype(str).nunique()),
                "distinct_nonempty_scaffold_count": int(current.loc[current["candidate_scaffold"].fillna("").astype(str) != "", "candidate_scaffold"].nunique()),
                "raw_valid_fraction": safe_fraction(raw.rdkit_valid_count, raw.raw_slot_count),
                "raw_policy_acceptance_fraction": safe_fraction(raw.policy_accepted_count, raw.raw_slot_count),
                "canonical_identity_yield_fraction": safe_fraction(len(current), raw.raw_slot_count),
                "raw_seed_identity_fraction": safe_fraction(raw.seed_identity_raw_count, raw.policy_accepted_count),
                "mmp_fraction_among_genuine_nonseed": safe_fraction(int(mmp.sum()), int(genuine.sum())),
                "same_scaffold_fraction_among_scaffold_eligible_nonseed": safe_fraction(int(retained.sum()), int(seed_has.sum())),
                "novel_fraction_among_genuine_nonseed": safe_fraction(int((novel & genuine).sum()), int(genuine.sum())),
                "seed_candidate_morgan_mean_nonseed": float(np.mean(nonseed_similarity)) if len(nonseed_similarity) else math.nan,
                "seed_candidate_morgan_median_nonseed": float(np.median(nonseed_similarity)) if len(nonseed_similarity) else math.nan,
                "within_pairwise_possible_count": possible_pairs,
                "within_pairwise_evaluated_count": len(diversity),
                "within_pairwise_exact": exact,
                "within_pairwise_sum": float(diversity.sum(dtype=np.float64)),
                "within_pairwise_mean": diversity_summary["mean"],
                "within_pairwise_median": diversity_summary["median"],
                "within_pairwise_q25": diversity_summary["q25"],
                "within_pairwise_q75": diversity_summary["q75"],
                "within_pairwise_q10": diversity_summary["q10"],
                "within_pairwise_q90": diversity_summary["q90"],
            }
        )
        metric_rows.append(row)
    seed_metrics = pd.DataFrame(metric_rows).sort_values(
        ["strategy", "query_position", "budget"], ignore_index=True
    )

    similarity_rows: list[dict[str, Any]] = []
    budgets = sorted(seed_metrics["budget"].unique())
    populations = {
        "all_genuine_nonseed": lambda frame: frame["is_genuine_nonseed"],
        "one_cut_mmp_derivative": lambda frame: frame["chemical_category"] == "one_cut_mmp_derivative",
        "same_scaffold_non_mmp": lambda frame: frame["chemical_category"] == "scaffold_preserving_non_mmp_analogue",
        "scaffold_changing": lambda frame: frame["chemical_category"] == "scaffold_changing_analogue",
        "acyclic_non_mmp": lambda frame: frame["chemical_category"] == "acyclic_non_mmp_analogue",
    }
    for strategy, strategy_frame in candidates.groupby("strategy", sort=False):
        for budget in budgets:
            current = strategy_frame.loc[strategy_frame["first_proposal_rank"] <= budget]
            for population, selector in populations.items():
                values = current.loc[
                    selector(current), "morgan_similarity_to_seed"
                ].to_numpy(dtype=np.float64)
                similarity_rows.append(
                    {"strategy": strategy, "budget": budget, "population": population, **numeric_summary(values)}
                )
    similarity = pd.DataFrame(similarity_rows)

    histogram_rows = []
    bins = int(cfg["similarity"]["histogram_bins"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    for (strategy, budget), counts in sorted(histogram_counts.items()):
        total = int(counts.sum())
        for index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "strategy": strategy,
                    "budget": budget,
                    "bin_left": edges[index],
                    "bin_right": edges[index + 1],
                    "evaluated_pair_count": int(count),
                    "fraction": safe_fraction(int(count), total),
                }
            )
    return candidates, seed_metrics, similarity, pd.DataFrame(histogram_rows)


def aggregate_tables(
    seed_metrics: pd.DataFrame, similarity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    similarity_lookup = similarity.loc[
        similarity["population"] == "all_genuine_nonseed"
    ].set_index(["strategy", "budget"])
    for (strategy, budget), group in seed_metrics.groupby(["strategy", "budget"], sort=False):
        raw_slots = int(group["raw_slot_count"].sum())
        accepted = int(group["policy_accepted_count"].sum())
        genuine = int(group["genuine_nonseed_count"].sum())
        mmp = int(group["mmp_derivative_count"].sum())
        novel_genuine = int(group["novel_genuine_nonseed_count"].sum())
        pair_count = int(group["within_pairwise_evaluated_count"].sum())
        sim = similarity_lookup.loc[(strategy, budget)]
        summary_rows.append(
            {
                "strategy": strategy,
                "budget": int(budget),
                "seed_rows": len(group),
                "raw_slot_count": raw_slots,
                "raw_valid_fraction": safe_fraction(group["rdkit_valid_count"].sum(), raw_slots),
                "raw_policy_acceptance_fraction": safe_fraction(accepted, raw_slots),
                "raw_duplicate_or_redundant_fraction": safe_fraction(accepted - group["unique_accepted_identity_count"].sum(), accepted),
                "mean_unique_accepted_identities": float(group["unique_accepted_identity_count"].mean()),
                "median_unique_accepted_identities": float(group["unique_accepted_identity_count"].median()),
                "mean_genuine_nonseed": float(group["genuine_nonseed_count"].mean()),
                "mean_mmp_derivatives": float(group["mmp_derivative_count"].mean()),
                "mmp_fraction_among_genuine_nonseed": safe_fraction(mmp, genuine),
                "seed_fraction_with_1_mmp": float((group["mmp_derivative_count"] >= 1).mean()),
                "seed_fraction_with_5_mmp": float((group["mmp_derivative_count"] >= 5).mean()),
                "seed_fraction_with_10_mmp": float((group["mmp_derivative_count"] >= 10).mean()),
                "mean_same_scaffold_non_mmp": float(group["same_scaffold_non_mmp_count"].mean()),
                "same_scaffold_fraction_among_scaffold_eligible_nonseed": safe_fraction(group["retains_nonempty_seed_scaffold_count"].sum(), group["scaffold_eligible_nonseed_count"].sum()),
                "mean_distinct_scaffolds": float(group["distinct_scaffold_count"].mean()),
                "median_distinct_scaffolds": float(group["distinct_scaffold_count"].median()),
                "mean_novel_genuine_nonseed": float(group["novel_genuine_nonseed_count"].mean()),
                "novel_fraction_among_genuine_nonseed": safe_fraction(novel_genuine, genuine),
                "mean_novel_mmp": float(group["novel_mmp_count"].mean()),
                "mean_novel_same_scaffold_non_mmp": float(group["novel_same_scaffold_non_mmp_count"].mean()),
                "mean_novel_scaffold_changing": float(group["novel_scaffold_changing_count"].mean()),
                "mean_novel_useful_local": float(group["novel_useful_local_count"].mean()),
                "median_novel_useful_local": float(group["novel_useful_local_count"].median()),
                "seed_fraction_with_1_novel_useful_local": float((group["novel_useful_local_count"] >= 1).mean()),
                "seed_fraction_with_5_novel_useful_local": float((group["novel_useful_local_count"] >= 5).mean()),
                "seed_fraction_with_10_novel_useful_local": float((group["novel_useful_local_count"] >= 10).mean()),
                "median_seed_candidate_morgan_nonseed": float(sim["median"]),
                "q25_seed_candidate_morgan_nonseed": float(sim["q25"]),
                "q75_seed_candidate_morgan_nonseed": float(sim["q75"]),
                "within_pairwise_morgan_weighted_mean": safe_fraction(group["within_pairwise_sum"].sum(), pair_count),
                "within_pairwise_evaluated_pairs": pair_count,
                "within_pairwise_approximate_seed_fraction": float((~group["within_pairwise_exact"].astype(bool)).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["strategy", "budget"], ignore_index=True)

    incremental_rows: list[dict[str, Any]] = []
    increment_fields = [
        "unique_accepted_identity_count",
        "mmp_derivative_count",
        "retains_nonempty_seed_scaffold_count",
        "novel_genuine_nonseed_count",
        "novel_useful_local_count",
        "distinct_scaffold_count",
    ]
    for (strategy, query), group in seed_metrics.groupby(["strategy", "query_position"], sort=False):
        group = group.sort_values("budget")
        previous_budget = 0
        previous = {field: 0 for field in increment_fields}
        for row in group.itertuples(index=False):
            record = {
                "strategy": strategy,
                "query_position": int(query),
                "budget": int(row.budget),
                "interval_start_exclusive": previous_budget,
                "interval_raw_proposals": int(row.budget) - previous_budget,
            }
            for field in increment_fields:
                value = int(getattr(row, field))
                delta = value - previous[field]
                if delta < 0:
                    raise RuntimeError(f"Nested yield decreased: {field}")
                record[f"new_{field}"] = delta
                record[f"rate_{field}"] = delta / record["interval_raw_proposals"]
                previous[field] = value
            incremental_rows.append(record)
            previous_budget = int(row.budget)
    incremental_seed = pd.DataFrame(incremental_rows)
    rate_columns = [column for column in incremental_seed.columns if column.startswith("rate_")]
    new_columns = [column for column in incremental_seed.columns if column.startswith("new_")]
    incremental = (
        incremental_seed.groupby(["strategy", "budget", "interval_start_exclusive", "interval_raw_proposals"], as_index=False)
        .agg(
            **{f"mean_{column}": (column, "mean") for column in new_columns},
            **{f"mean_{column}": (column, "mean") for column in rate_columns},
        )
        .sort_values(["strategy", "budget"], ignore_index=True)
    )
    return summary, incremental


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    phase = args.phase
    state_path = root / "state" / f"{phase.upper()}_ANALYSIS_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    generation_seal = root / "state" / f"{phase.upper()}_GENERATION_COMPLETE.json"
    if not generation_seal.is_file():
        raise RuntimeError("Seal raw generation before analysis")
    cfg = protocol(root)
    workers = int(args.workers or cfg["execution"]["analysis_workers"])
    threads = int(cfg["execution"]["analysis_workers"])
    paths, _ = resolve_manifest_inputs(repo_root, root)
    panel_name = "development_panel.csv" if phase == "development" else "fresh_validation_panel.csv"
    panel = pd.read_csv(root / "prepared" / panel_name)
    strategies = phase_strategies(cfg, root, phase)
    generation = load_json(generation_seal)
    if generation["strategies"] != strategies:
        raise RuntimeError("Generation strategy set differs from analysis strategy set")
    started = time.monotonic()
    resolved = load_json(paths["gmolai_resolved_config"])
    audit, raw_metrics, raw_candidates = audit_and_extract(
        root,
        phase,
        panel,
        [int(value) for value in cfg["budgets"]],
        resolved,
        workers,
        threads,
    )
    candidates, pair_chemistry, structures, fingerprints = characterize_pairs(
        raw_candidates, panel, audit, cfg, root, phase, workers, threads
    )
    novelty_frame = pd.read_parquet(root / "prepared" / "decoder_training_identities.parquet")
    training_hashes = set(novelty_frame["molecule_hash"].astype(str))
    if len(training_hashes) != 980_000:
        raise RuntimeError("Decoder-training novelty reference changed")
    characterized, seed_metrics, similarity, diversity_histogram = seed_budget_tables(
        candidates,
        pair_chemistry,
        raw_metrics,
        training_hashes,
        fingerprints,
        cfg,
    )
    summary, incremental = aggregate_tables(seed_metrics, similarity)

    table_root = root / "outputs" / "tables"
    outputs = {
        "candidate_characterization": table_root / f"{phase}_candidate_characterization.parquet",
        "seed_budget_metrics": table_root / f"{phase}_seed_budget_metrics.parquet",
        "budget_summary": table_root / f"{phase}_budget_summary.csv",
        "similarity": table_root / f"{phase}_similarity_by_category_budget.csv",
        "diversity_histogram": table_root / f"{phase}_within_set_similarity_histogram.csv",
        "incremental": table_root / f"{phase}_incremental_yield.csv",
    }
    atomic_write_parquet(outputs["candidate_characterization"], characterized, root)
    atomic_write_parquet(outputs["seed_budget_metrics"], seed_metrics, root)
    atomic_write_csv(outputs["budget_summary"], summary, root)
    atomic_write_csv(outputs["similarity"], similarity, root)
    atomic_write_csv(outputs["diversity_histogram"], diversity_histogram, root)
    atomic_write_csv(outputs["incremental"], incremental, root)
    phase_summary = {
        "schema_version": 1,
        "phase": phase,
        "strategies": strategies,
        "seed_rows": len(panel),
        "raw_rows": int(generation["raw_rows"]),
        "policy_audit_unique_raw_strings": len(audit),
        "unique_strategy_seed_candidate_rows": len(characterized),
        "unique_seed_candidate_chemistry_pairs": len(pair_chemistry),
        "unique_molecules_fragmented": len(structures),
        "budgets": [int(value) for value in cfg["budgets"]],
        "novelty_reference_rows": len(training_hashes),
        "output_sha256": {role: sha256_file(path) for role, path in outputs.items()},
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    summary_path = root / "outputs" / f"{phase}_analysis_summary.json"
    atomic_write_json(summary_path, phase_summary, root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "phase": phase,
        "completed_at": utc_now(),
        "summary_sha256": sha256_file(summary_path),
        "output_sha256": phase_summary["output_sha256"],
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
