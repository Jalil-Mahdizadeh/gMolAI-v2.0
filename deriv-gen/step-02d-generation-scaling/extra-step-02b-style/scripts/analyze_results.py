#!/usr/bin/env python3
"""Re-encode-aligned Step 2d candidate scoring and paired bootstrap analysis."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analysis_core import MetricSpec, paired_bootstrap, select_reranked
from common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    input_paths,
    load_config,
    load_json,
    require_analysis_root,
    require_repo_root,
    sha256_file,
    validate_inputs,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} changed or is incompatible: {observed!r} != {expected!r}")


def validate_embedding_artifact(
    root: Path,
    config: dict,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    output = root / "intermediate" / "candidate_embeddings.npz"
    metadata_path = output.with_suffix(".metadata.json")
    rejections_path = output.with_suffix(".rejections.csv")
    for path in (output, metadata_path, rejections_path):
        if not path.is_file():
            raise RuntimeError(f"Missing re-encoding output: {path}")

    metadata = load_json(metadata_path)
    execution = metadata["execution"]
    requested = config["reencoding"]
    expected_backend = (
        "optimized_gine_v1"
        if requested["backend"] == "optimized"
        else requested["backend"]
    )
    require_equal(execution["backend"], expected_backend, "encoder backend")
    require_equal(int(execution["batch_size"]), int(requested["batch_size"]), "encoder batch size")
    require_equal(int(execution["node_budget"]), int(requested["node_budget"]), "encoder node budget")
    require_equal(int(execution["workers"]), int(requested["workers"]), "encoder workers")
    require_equal(int(execution["threads"]), int(requested["threads"]), "encoder threads")
    require_equal(execution["invalid_policy"], "error", "encoder invalid policy")
    if not str(execution["device"]).startswith("cuda"):
        raise RuntimeError(f"Expected CUDA re-encoding, observed {execution['device']!r}")
    require_equal(int(metadata["rows"]["total"]), expected_rows, "encoder input rows")
    require_equal(int(metadata["rows"]["accepted"]), expected_rows, "encoder accepted rows")
    require_equal(int(metadata["rows"]["rejected"]), 0, "encoder rejected rows")
    require_equal(int(metadata["embedding"]["dimensions"]), int(requested["embedding_dimensions"]), "embedding dimensions")
    require_equal(metadata["embedding"]["space"], "released_hybrid_w3", "embedding space")
    require_equal(metadata["input"]["sha256"], sha256_file(root / "inputs" / "encoder_input.csv"), "encoder input hash")
    require_equal(metadata["outputs"][output.name]["sha256"], sha256_file(output), "embedding output hash")

    with np.load(output, allow_pickle=False) as payload:
        required = {"embeddings", "input_id", "input_row", "molecule_hash", "embedding_dimensions", "embedding_space", "input_sha256"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"Embedding bundle lacks arrays: {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        input_ids = np.asarray(payload["input_id"], dtype=str)
        input_rows = np.asarray(payload["input_row"], dtype=np.int64)
        molecule_hashes = np.asarray(payload["molecule_hash"], dtype=str)
        require_equal(tuple(embeddings.shape), (expected_rows, int(requested["embedding_dimensions"])), "embedding matrix shape")
        if not np.array_equal(input_rows, np.arange(1, expected_rows + 1, dtype=np.int64)):
            raise RuntimeError("Encoded input_row is not the exact one-based prepared order")
        mismatch_indices = np.flatnonzero(input_ids != molecule_hashes)
        audit = pd.DataFrame(
            {
                "structure_index": mismatch_indices.astype(np.int64),
                "retained_step2d_molecule_hash": input_ids[mismatch_indices],
                "encoder_recanonicalized_molecule_hash": molecule_hashes[mismatch_indices],
                "input_smiles": np.asarray(payload["input_smiles"], dtype=str)[mismatch_indices],
                "encoder_canonical_smiles": np.asarray(payload["canonical_smiles"], dtype=str)[mismatch_indices],
            }
        )
        atomic_write_csv(
            root / "intermediate" / "reencoding_canonicalization_audit.csv",
            audit,
            root,
        )
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding matrix contains non-finite values")
    return embeddings, input_ids, len(mismatch_indices)


def latent_metrics(
    candidates: pd.DataFrame,
    embeddings: np.ndarray,
    conditions: np.ndarray,
    *,
    chunk_rows: int = 20_000,
) -> None:
    structure_indices = candidates["candidate_structure_index"].to_numpy(dtype=np.int64)
    query_positions = candidates["query_position"].to_numpy(dtype=np.int64)
    count = len(candidates)
    l2 = np.empty(count, dtype=np.float32)
    relative_l2 = np.empty(count, dtype=np.float32)
    cosine = np.empty(count, dtype=np.float32)
    for start in range(0, count, chunk_rows):
        stop = min(start + chunk_rows, count)
        candidate_values = embeddings[structure_indices[start:stop]].astype(np.float64)
        supplied_values = conditions[query_positions[start:stop]].astype(np.float64)
        delta = candidate_values - supplied_values
        current_l2 = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        supplied_norm = np.sqrt(np.einsum("ij,ij->i", supplied_values, supplied_values))
        candidate_norm = np.sqrt(np.einsum("ij,ij->i", candidate_values, candidate_values))
        denominator = np.maximum(candidate_norm * supplied_norm, 1e-12)
        current_cosine = np.einsum("ij,ij->i", candidate_values, supplied_values) / denominator
        l2[start:stop] = current_l2.astype(np.float32)
        relative_l2[start:stop] = (current_l2 / np.maximum(supplied_norm, 1e-12)).astype(np.float32)
        cosine[start:stop] = current_cosine.astype(np.float32)
    candidates["latent_l2_to_seed_condition"] = l2
    candidates["latent_relative_l2_to_seed_condition"] = relative_l2
    candidates["latent_cosine_to_seed_condition"] = cosine


def greedy_rows(repo_root: Path, root: Path, strategy: str) -> pd.DataFrame:
    step = repo_root / "deriv-gen" / "step-02d-generation-scaling"
    raw_glob = str(step / "outputs" / "raw" / "final" / "*.parquet")
    audit = str(step / "intermediate" / "final_raw_smiles_policy_audit.parquet")
    characterization = str(step / "outputs" / "tables" / "final_candidate_characterization.parquet")
    temporary = root / "state" / "duckdb_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET temp_directory = ?", [str(temporary)])
    query = """
        WITH greedy AS (
            SELECT query_position, target_hash, proposal_rank, raw_smiles,
                   CASE WHEN coalesce(token_error, '') = '' AND length(raw_smiles) > 0
                        THEN TRUE ELSE FALSE END AS token_decoded
            FROM read_parquet(?)
            WHERE phase = 'final' AND strategy = ? AND source_kind = 'greedy'
        ),
        audit AS (
            SELECT raw_smiles, rdkit_valid, policy_accepted, canonical_smiles,
                   molecule_hash, scaffold
            FROM read_parquet(?)
        ),
        candidates AS (
            SELECT query_position, candidate_hash, seed_scaffold,
                   morgan_similarity_to_seed
            FROM read_parquet(?)
            WHERE strategy = ?
        )
        SELECT g.query_position,
               g.proposal_rank AS greedy_proposal_rank,
               g.raw_smiles AS greedy_raw_smiles,
               CAST(g.token_decoded AS DOUBLE) AS greedy_token_decoded,
               CAST(coalesce(a.rdkit_valid, FALSE) AS DOUBLE) AS greedy_rdkit_valid,
               CAST(coalesce(a.policy_accepted, FALSE) AS DOUBLE) AS greedy_policy_accepted,
               a.canonical_smiles AS greedy_canonical_smiles,
               a.molecule_hash AS greedy_candidate_hash,
               CAST(CASE WHEN coalesce(a.policy_accepted, FALSE)
                              AND a.molecule_hash = g.target_hash
                         THEN 1.0 ELSE 0.0 END AS DOUBLE) AS greedy_exact_seed_identity,
               CAST(CASE WHEN coalesce(a.policy_accepted, FALSE)
                              AND coalesce(a.scaffold, '') = coalesce(c.seed_scaffold, '')
                         THEN 1.0 ELSE 0.0 END AS DOUBLE) AS greedy_seed_scaffold_recovery,
               CAST(CASE WHEN coalesce(a.policy_accepted, FALSE)
                         THEN coalesce(c.morgan_similarity_to_seed, 0.0)
                         ELSE 0.0 END AS DOUBLE) AS greedy_morgan_similarity_to_seed,
               CAST(CASE WHEN coalesce(a.policy_accepted, FALSE)
                         THEN c.morgan_similarity_to_seed ELSE NULL END AS DOUBLE)
                   AS greedy_morgan_similarity_to_seed_conditional
        FROM greedy AS g
        LEFT JOIN audit AS a USING (raw_smiles)
        LEFT JOIN candidates AS c
          ON c.query_position = g.query_position
         AND c.candidate_hash = a.molecule_hash
        ORDER BY g.query_position
    """
    result = connection.execute(
        query,
        [raw_glob, strategy, audit, characterization, strategy],
    ).df()
    connection.close()
    if len(result) != 10_000 or result["query_position"].nunique() != 10_000:
        raise RuntimeError("Expected exactly one retained greedy proposal for every seed")
    return result


def merge_selected(
    base: pd.DataFrame,
    selected: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    fields = [
        "query_position",
        "first_proposal_rank",
        "first_source_kind",
        "first_source_rank",
        "canonical_smiles",
        "candidate_hash",
        "is_seed_identity",
        "seed_scaffold_recovery",
        "morgan_similarity_to_seed",
        "latent_l2_to_seed_condition",
        "latent_relative_l2_to_seed_condition",
        "latent_cosine_to_seed_condition",
    ]
    renamed = selected[fields].rename(
        columns={name: f"{prefix}_{name}" for name in fields if name != "query_position"}
    )
    return base.merge(renamed, how="left", on="query_position", validate="one_to_one")


def finish_top1_columns(frame: pd.DataFrame, prefix: str) -> None:
    hash_column = f"{prefix}_candidate_hash"
    frame[f"{prefix}_available"] = frame[hash_column].notna().astype(np.float64)
    frame[f"{prefix}_exact_seed_identity"] = (
        frame[f"{prefix}_is_seed_identity"].fillna(False).astype(np.float64)
    )
    frame[f"{prefix}_seed_scaffold_recovery"] = (
        frame[f"{prefix}_seed_scaffold_recovery"].fillna(False).astype(np.float64)
    )
    conditional = frame[f"{prefix}_morgan_similarity_to_seed"].where(
        frame[f"{prefix}_available"] == 1.0
    )
    frame[f"{prefix}_morgan_similarity_to_seed_conditional"] = conditional
    frame[f"{prefix}_morgan_similarity_to_seed"] = conditional.fillna(0.0).astype(np.float64)


def metric_specs() -> list[MetricSpec]:
    mean = lambda name, column=None, unit="fraction", denominator="all seeds": MetricSpec(
        name=name,
        kind="mean",
        column=column or name,
        unit=unit,
        denominator_description=denominator,
    )
    nanmean = lambda name, column=None, unit="value", denominator="available selected top-1 candidates": MetricSpec(
        name=name,
        kind="nanmean",
        column=column or name,
        unit=unit,
        denominator_description=denominator,
    )
    return [
        mean("raw_token_decode_fraction"),
        mean("raw_rdkit_valid_fraction", "raw_valid_fraction"),
        mean("raw_policy_acceptance_fraction"),
        mean("unique_identity_yield_per_raw_slot", "canonical_identity_yield_fraction"),
        mean("mean_unique_accepted_identities", "unique_accepted_identity_count", unit="count"),
        mean("candidate_availability_rate", "candidate_set_nonempty"),
        mean("greedy_rdkit_valid_top1_rate", "greedy_rdkit_valid"),
        mean("greedy_policy_accepted_top1_rate", "greedy_policy_accepted"),
        mean("greedy_exact_seed_identity_at_1", "greedy_exact_seed_identity"),
        mean("greedy_seed_scaffold_recovery"),
        mean("greedy_mean_morgan_to_seed", "greedy_morgan_similarity_to_seed", unit="similarity"),
        nanmean("greedy_conditional_mean_morgan_to_seed", "greedy_morgan_similarity_to_seed_conditional", unit="similarity", denominator="policy-accepted greedy top-1 candidates"),
        mean("generator_order_valid_top1_rate", "generator_order_available"),
        mean("generator_order_exact_seed_identity_at_1", "generator_order_exact_seed_identity"),
        mean("generator_order_seed_scaffold_recovery"),
        mean("generator_order_mean_morgan_to_seed", "generator_order_morgan_similarity_to_seed", unit="similarity"),
        nanmean("generator_order_conditional_mean_morgan_to_seed", "generator_order_morgan_similarity_to_seed_conditional", unit="similarity", denominator="available generator-order top-1 candidates"),
        mean("exact_seed_oracle_recall_at_budget", "oracle_exact_seed_recall"),
        mean("reranked_valid_top1_rate", "reranked_available"),
        mean("reranked_exact_seed_identity_at_1", "reranked_exact_seed_identity"),
        MetricSpec(
            name="rerank_selection_efficiency_given_oracle_presence",
            kind="ratio",
            numerator="reranked_exact_seed_identity",
            denominator="oracle_exact_seed_recall",
            unit="fraction",
            denominator_description="seeds with exact seed present in accepted candidate prefix",
        ),
        mean("reranked_identity_gain_over_greedy", "reranked_identity_gain_over_greedy", unit="fraction_point"),
        mean("reranked_seed_scaffold_recovery"),
        mean("reranked_mean_morgan_to_seed", "reranked_morgan_similarity_to_seed", unit="similarity"),
        nanmean("reranked_conditional_mean_morgan_to_seed", "reranked_morgan_similarity_to_seed_conditional", unit="similarity"),
        nanmean("reranked_mean_latent_l2", "reranked_latent_l2_to_seed_condition", unit="distance"),
        nanmean("reranked_mean_latent_relative_l2", "reranked_latent_relative_l2_to_seed_condition", unit="relative_distance"),
        nanmean("reranked_mean_latent_cosine", "reranked_latent_cosine_to_seed_condition", unit="cosine"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = require_repo_root(args.repo_root)
    root = require_analysis_root(args.analysis_root)
    config = load_config(root)
    paths = input_paths(repo_root)
    validate_inputs(paths)
    complete_path = root / "state" / "ANALYSIS_COMPLETE.json"
    if complete_path.exists():
        print(json.dumps(load_json(complete_path), sort_keys=True))
        return

    started = time.monotonic()
    step = repo_root / "deriv-gen" / "step-02d-generation-scaling"
    final_summary = load_json(step / "outputs" / "final_analysis_summary.json")
    expected_seeds = int(final_summary["seed_rows"])
    expected_unique = int(final_summary["unique_molecules_fragmented"])
    expected_candidates = int(final_summary["unique_strategy_seed_candidate_rows"])
    require_equal(expected_seeds, 10_000, "Step 2d seed rows")
    require_equal(config["budgets"], final_summary["budgets"], "analysis budgets")

    unique = pd.read_parquet(
        Path(paths["unique_molecules"]),
        columns=["structure_index", "molecule_hash"],
    ).sort_values("structure_index", ignore_index=True)
    require_equal(len(unique), expected_unique, "unique molecule rows")
    if not np.array_equal(unique["structure_index"].to_numpy(dtype=np.int64), np.arange(expected_unique, dtype=np.int64)):
        raise RuntimeError("Unique molecule structure_index is not contiguous")
    embeddings, encoded_ids, recanonicalized_rows = validate_embedding_artifact(
        root, config, expected_unique
    )
    unique_ids = unique["molecule_hash"].astype(str).to_numpy()
    if not np.array_equal(encoded_ids, unique_ids):
        raise RuntimeError("Re-encoded molecule IDs are not exactly aligned to Step 2d structure_index")
    del encoded_ids, unique_ids, unique
    gc.collect()

    conditions = np.load(Path(paths["conditions"]), allow_pickle=False)
    require_equal(tuple(conditions.shape), (expected_seeds, int(config["reencoding"]["embedding_dimensions"])), "condition matrix shape")
    if not np.isfinite(conditions).all():
        raise RuntimeError("Seed-condition matrix contains non-finite values")

    candidate_columns = [
        "strategy",
        "query_position",
        "target_index",
        "target_hash",
        "first_proposal_rank",
        "first_source_kind",
        "first_source_rank",
        "canonical_smiles",
        "candidate_hash",
        "candidate_scaffold",
        "seed_scaffold",
        "candidate_structure_index",
        "is_seed_identity",
        "morgan_similarity_to_seed",
    ]
    candidates = pd.read_parquet(Path(paths["candidate_characterization"]), columns=candidate_columns)
    require_equal(len(candidates), expected_candidates, "candidate rows")
    strategy = str(config["candidate_scope"]["strategy"])
    require_equal(candidates["strategy"].drop_duplicates().tolist(), [strategy], "candidate strategy")
    if candidates.duplicated(["query_position", "candidate_hash"]).any():
        raise RuntimeError("Candidate characterization is not unique per seed and identity")
    indices = candidates["candidate_structure_index"].to_numpy(dtype=np.int64)
    if indices.min() < 0 or indices.max() >= expected_unique:
        raise RuntimeError("Candidate structure index is outside the embedding matrix")
    candidates["seed_scaffold_recovery"] = (
        candidates["candidate_scaffold"].fillna("").astype(str).to_numpy()
        == candidates["seed_scaffold"].fillna("").astype(str).to_numpy()
    )
    latent_metrics(candidates, embeddings, conditions)
    del embeddings, conditions, indices
    gc.collect()

    latent_table_columns = [
        "strategy",
        "query_position",
        "target_index",
        "target_hash",
        "first_proposal_rank",
        "first_source_kind",
        "first_source_rank",
        "canonical_smiles",
        "candidate_hash",
        "candidate_structure_index",
        "is_seed_identity",
        "seed_scaffold_recovery",
        "morgan_similarity_to_seed",
        "latent_l2_to_seed_condition",
        "latent_relative_l2_to_seed_condition",
        "latent_cosine_to_seed_condition",
    ]
    candidate_latent_path = root / "outputs" / "tables" / "candidate_latent_metrics.parquet"
    atomic_write_parquet(candidate_latent_path, candidates[latent_table_columns], root)

    greedy = greedy_rows(repo_root, root, strategy)
    seed_columns = [
        "strategy",
        "query_position",
        "budget",
        "raw_slot_count",
        "token_decoded_count",
        "rdkit_valid_count",
        "policy_accepted_count",
        "unique_policy_accepted_identity_count_raw",
        "unique_accepted_identity_count",
        "seed_identity_unique_count",
        "raw_valid_fraction",
        "raw_policy_acceptance_fraction",
        "canonical_identity_yield_fraction",
    ]
    seed_metrics = pd.read_parquet(Path(paths["seed_budget_metrics"]), columns=seed_columns)
    require_equal(len(seed_metrics), expected_seeds * len(config["budgets"]), "seed-budget rows")
    require_equal(sorted(seed_metrics["budget"].unique().tolist()), config["budgets"], "seed-budget values")
    seed_metrics["raw_token_decode_fraction"] = (
        seed_metrics["token_decoded_count"] / seed_metrics["raw_slot_count"]
    )

    generator_first = candidates.sort_values(
        ["query_position", "first_proposal_rank", "canonical_smiles"],
        kind="mergesort",
    ).drop_duplicates("query_position", keep="first")

    per_budget: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    specs = metric_specs()
    bootstrap_cfg = config["bootstrap"]
    for budget in config["budgets"]:
        query = seed_metrics.loc[seed_metrics["budget"] == int(budget)].copy()
        query = query.sort_values("query_position", ignore_index=True)
        if not np.array_equal(query["query_position"].to_numpy(dtype=np.int64), np.arange(expected_seeds, dtype=np.int64)):
            raise RuntimeError(f"Budget {budget} lacks the exact seed population/order")
        query["candidate_set_nonempty"] = (query["unique_accepted_identity_count"] > 0).astype(np.float64)
        query["oracle_exact_seed_recall"] = (query["seed_identity_unique_count"] > 0).astype(np.float64)
        query = query.merge(greedy, how="left", on="query_position", validate="one_to_one")

        generator_selected = generator_first.loc[
            generator_first["first_proposal_rank"] <= int(budget)
        ]
        query = merge_selected(query, generator_selected, "generator_order")
        finish_top1_columns(query, "generator_order")
        if not np.array_equal(
            query["candidate_set_nonempty"].to_numpy(),
            query["generator_order_available"].to_numpy(),
        ):
            raise RuntimeError(f"Generator-order availability disagrees with candidate count at budget {budget}")

        reranked_selected = select_reranked(candidates, int(budget))
        query = merge_selected(query, reranked_selected, "reranked")
        finish_top1_columns(query, "reranked")
        if not np.array_equal(
            query["candidate_set_nonempty"].to_numpy(),
            query["reranked_available"].to_numpy(),
        ):
            raise RuntimeError(f"Reranked availability disagrees with candidate count at budget {budget}")
        query["reranked_identity_gain_over_greedy"] = (
            query["reranked_exact_seed_identity"] - query["greedy_exact_seed_identity"]
        )
        if (query["reranked_exact_seed_identity"] > query["oracle_exact_seed_recall"]).any():
            raise RuntimeError(f"Reranked recovery exceeds exact-seed oracle at budget {budget}")

        candidate_oracle = (
            candidates.loc[
                (candidates["first_proposal_rank"] <= int(budget))
                & candidates["is_seed_identity"]
            ]["query_position"]
            .drop_duplicates()
            .to_numpy(dtype=np.int64)
        )
        expected_oracle = np.zeros(expected_seeds, dtype=np.float64)
        expected_oracle[candidate_oracle] = 1.0
        if not np.array_equal(query["oracle_exact_seed_recall"].to_numpy(), expected_oracle):
            raise RuntimeError(f"Seed-budget oracle and candidate oracle disagree at budget {budget}")

        ci = paired_bootstrap(
            query,
            specs,
            resamples=int(bootstrap_cfg["resamples"]),
            confidence_level=float(bootstrap_cfg["confidence_level"]),
            seed=int(config["seed"]),
        )
        ci.insert(0, "budget", int(budget))
        ci.insert(0, "strategy", strategy)
        ci_frames.append(ci)
        per_budget.append(query)

    per_seed = pd.concat(per_budget, ignore_index=True)
    confidence_intervals = pd.concat(ci_frames, ignore_index=True)
    per_seed_path = root / "outputs" / "tables" / "per_seed_budget_metrics.parquet"
    ci_path = root / "outputs" / "tables" / "bootstrap_cis.csv"
    atomic_write_parquet(per_seed_path, per_seed, root)
    atomic_write_csv(ci_path, confidence_intervals, root)

    summary = confidence_intervals.pivot(index="budget", columns="metric", values="estimate").reset_index()
    summary.columns.name = None
    summary.insert(0, "strategy", strategy)
    summary.insert(2, "seed_rows", expected_seeds)
    descriptive = per_seed.groupby("budget", sort=True).agg(
        median_unique_accepted_identities=("unique_accepted_identity_count", "median"),
        median_reranked_latent_l2=("reranked_latent_l2_to_seed_condition", "median"),
        median_reranked_latent_relative_l2=("reranked_latent_relative_l2_to_seed_condition", "median"),
        median_reranked_latent_cosine=("reranked_latent_cosine_to_seed_condition", "median"),
    ).reset_index()
    summary = summary.merge(descriptive, how="left", on="budget", validate="one_to_one")
    summary_path = root / "outputs" / "tables" / "summary_metrics_by_budget.csv"
    atomic_write_csv(summary_path, summary, root)

    state = {
        "schema_version": 1,
        "status": "complete",
        "study_id": config["study_id"],
        "completed_at": utc_now(),
        "seed_rows": expected_seeds,
        "candidate_rows": expected_candidates,
        "unique_reencoded_molecules": expected_unique,
        "reencoding_recanonicalized_rows": recanonicalized_rows,
        "budgets": config["budgets"],
        "bootstrap_resamples": int(bootstrap_cfg["resamples"]),
        "outputs": {
            "candidate_latent_metrics": {"path": str(candidate_latent_path.relative_to(root)), "sha256": sha256_file(candidate_latent_path)},
            "per_seed_budget_metrics": {"path": str(per_seed_path.relative_to(root)), "sha256": sha256_file(per_seed_path)},
            "bootstrap_cis": {"path": str(ci_path.relative_to(root)), "sha256": sha256_file(ci_path)},
            "summary_metrics_by_budget": {"path": str(summary_path.relative_to(root)), "sha256": sha256_file(summary_path)},
        },
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(complete_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
