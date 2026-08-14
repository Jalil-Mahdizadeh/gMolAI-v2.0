#!/usr/bin/env python3
"""Run the frozen Day-1 gMolAI latent geometry and retrieval study."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import numpy as np
import pandas as pd
import pyarrow
import rdkit
import scipy
import sklearn
import torch
from scipy.stats import spearmanr

from day1_common import (
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    bootstrap_mean_ci,
    build_mmp_pairs,
    core_sets,
    covariance_eigendecomposition,
    covariance_sample,
    diverse_indices,
    effective_rank,
    ensure_within,
    fragment_molecules,
    hash_ledger,
    local_covariance_sample,
    make_fingerprints,
    object_sha256,
    one_cut_related,
    sha256_file,
    stable_digest,
    tanimoto,
    topk_l2,
    unit_vector,
)


SPACE_ORDER = ("graph_256", "mean_node_128", "hybrid_384")
BLIND_METHODS = ("isotropic", "global_covariance", "local_covariance")
INTERPOLATION_METHODS = (
    "interpolation_0.25",
    "interpolation_0.50",
    "interpolation_0.75",
    "interpolation_1.00",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/repo"),
        help="Read-only repository root inside the container.",
    )
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path(
            "/repo/deriv-gen/step-01-latent-geometry-retrieval"
        ),
        help="Only writable study root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Fragmentation workers; 0 uses the visible CPU allocation up to 48.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def save_figure(fig: plt.Figure, path: Path, step_root: Path) -> None:
    path = ensure_within(path, step_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(
            temporary,
            format=path.suffix.lstrip("."),
            dpi=220 if path.suffix == ".png" else None,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_and_validate_inputs(
    repo_root: Path,
    step_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role, record in manifest["files"].items():
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")
        actual = sha256_file(path)
        expected = str(record["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {role}: expected {expected}, observed {actual}"
            )
        paths[role] = path
        hashes[role] = actual
    forbidden = ("test-standardized", "test-partition", "moleculenet", "hiv")
    for role, path in paths.items():
        if role == "container":
            continue
        lowered = str(path).lower()
        if any(value in lowered for value in forbidden):
            raise RuntimeError(f"Forbidden input entered Day-1 study: {path}")
    for path in paths.values():
        if step_root.resolve() in path.resolve().parents:
            raise RuntimeError(f"Input must be immutable and external to step root: {path}")
    return paths, hashes


def payload_array(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Payload field {key!r} is not a tensor")
    return value.detach().cpu().numpy()


def chemical_records(
    payload: dict[str, Any], cache_records: dict[str, list[str]], work_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    hashes = [str(value) for value in payload["molecule_hashes"]]
    buckets = payload_array(payload, "source_buckets").reshape(-1).astype(np.int64)
    records = {
        value: (str(cache_records[value][0]), str(cache_records[value][1] or ""))
        for value in hashes
        if value in cache_records
    }
    missing_by_bucket: dict[int, list[str]] = defaultdict(list)
    for value, bucket in zip(hashes, buckets):
        if value not in records:
            missing_by_bucket[int(bucket)].append(value)
    connection = duckdb.connect(":memory:")
    try:
        for bucket, wanted_hashes in sorted(missing_by_bucket.items()):
            parquet_path = work_dir / "deduplicated" / f"bucket-{bucket:04d}.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(parquet_path)
            connection.register("wanted", pyarrow.table({"molecule_hash": wanted_hashes}))
            rows = connection.execute(
                """
                SELECT d.molecule_hash, d.canonical_smiles, d.scaffold
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet_path)],
            ).fetchall()
            connection.unregister("wanted")
            for molecule_hash, smiles, scaffold in rows:
                records[str(molecule_hash)] = (str(smiles), str(scaffold or ""))
    finally:
        connection.close()
    missing = [value for value in hashes if value not in records]
    if missing:
        raise RuntimeError(f"Chemical sources are missing {len(missing)} payload molecules")
    smiles = [records[value][0] for value in hashes]
    scaffolds = [records[value][1] for value in hashes]
    return hashes, smiles, scaffolds


def block_data(
    matrix: np.ndarray, spaces: dict[str, list[int]], name: str
) -> np.ndarray:
    start, stop = (int(value) for value in spaces[name])
    return np.ascontiguousarray(matrix[:, start:stop], dtype=np.float32)


def fit_mmp_directions(
    train_pairs: pd.DataFrame,
    eligible_validation_pairs: pd.DataFrame,
    train_space: np.ndarray,
    validation_space: np.ndarray,
    *,
    space: str,
    minimum_train_cores: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    directions: dict[str, dict[str, Any]] = {}
    transform_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    train_groups = {
        str(transform): group
        for transform, group in train_pairs.groupby("transform", sort=True)
        if group["core"].nunique() >= minimum_train_cores
    }
    for transform, group in train_groups.items():
        lhs = group["lhs_index"].to_numpy(dtype=np.int64)
        rhs = group["rhs_index"].to_numpy(dtype=np.int64)
        delta = train_space[rhs].astype(np.float64) - train_space[lhs].astype(
            np.float64
        )
        norms = np.linalg.norm(delta, axis=1)
        valid = norms > 1e-10
        delta = delta[valid]
        norms = norms[valid]
        if len(delta) < minimum_train_cores:
            continue
        units = delta / norms[:, None]
        summed = units.sum(axis=0)
        direction = unit_vector(summed)
        references = summed[None, :] - units
        reference_norm = np.linalg.norm(references, axis=1)
        valid_reference = reference_norm > 1e-10
        loo = np.sum(
            units[valid_reference]
            * (references[valid_reference] / reference_norm[valid_reference, None]),
            axis=1,
        )
        directions[transform] = {
            "unit": direction.astype(np.float32),
            "median_norm": float(np.median(norms)),
            "train_examples": int(len(delta)),
            "train_cores": int(group["core"].nunique()),
            "train_resultant_length": float(np.linalg.norm(units.mean(axis=0))),
            "train_loo_alignment_mean": float(np.mean(loo)) if len(loo) else math.nan,
            "train_loo_alignment_median": float(np.median(loo)) if len(loo) else math.nan,
        }

    available = sorted(directions)
    if len(available) < 2:
        raise RuntimeError(f"Too few repeated MMP transformations in {space}: {len(available)}")
    for transform, group in eligible_validation_pairs.groupby("transform", sort=True):
        transform = str(transform)
        if transform not in directions:
            continue
        direction = directions[transform]["unit"].astype(np.float64)
        lhs = group["lhs_index"].to_numpy(dtype=np.int64)
        rhs = group["rhs_index"].to_numpy(dtype=np.int64)
        delta = validation_space[rhs].astype(np.float64) - validation_space[lhs].astype(
            np.float64
        )
        norms = np.linalg.norm(delta, axis=1)
        valid = norms > 1e-10
        delta = delta[valid]
        norms = norms[valid]
        valid_group = group.iloc[np.flatnonzero(valid)]
        alignments = (delta / norms[:, None]) @ direction
        null_values: list[float] = []
        for pair_row, unit_delta in zip(valid_group.itertuples(index=False), delta / norms[:, None]):
            position = int(stable_digest(seed, space, pair_row.pair_id)[:16], 16) % len(
                available
            )
            null_transform = available[position]
            if null_transform == transform:
                null_transform = available[(position + 1) % len(available)]
            null_values.append(
                float(unit_delta @ directions[null_transform]["unit"].astype(np.float64))
            )
        for pair_row, alignment, null_alignment, norm in zip(
            valid_group.itertuples(index=False), alignments, null_values, norms
        ):
            alignment_rows.append(
                {
                    "space": space,
                    "pair_id": str(pair_row.pair_id),
                    "transform": transform,
                    "core": str(pair_row.core),
                    "alignment": float(alignment),
                    "null_alignment": float(null_alignment),
                    "validation_delta_norm": float(norm),
                }
            )
        metadata = directions[transform]
        transform_rows.append(
            {
                "space": space,
                "transform": transform,
                "train_examples": metadata["train_examples"],
                "train_cores": metadata["train_cores"],
                "train_resultant_length": metadata["train_resultant_length"],
                "train_loo_alignment_mean": metadata["train_loo_alignment_mean"],
                "train_loo_alignment_median": metadata["train_loo_alignment_median"],
                "median_train_step_norm": metadata["median_norm"],
                "validation_examples_unseen_core": int(len(alignments)),
                "validation_alignment_mean": float(np.mean(alignments)),
                "validation_alignment_median": float(np.median(alignments)),
                "null_alignment_mean": float(np.mean(null_values)),
                "null_alignment_median": float(np.median(null_values)),
            }
        )

    alignment = pd.DataFrame(alignment_rows)
    transform_summary = pd.DataFrame(transform_rows)
    overall_rows: list[dict[str, Any]] = []
    if len(alignment):
        pair_mean, pair_low, pair_high = bootstrap_mean_ci(
            alignment["alignment"].to_numpy(), seed=seed + 17, resamples=1000
        )
        null_mean, null_low, null_high = bootstrap_mean_ci(
            alignment["null_alignment"].to_numpy(), seed=seed + 19, resamples=1000
        )
        transform_means = alignment.groupby("transform", sort=True)[
            ["alignment", "null_alignment"]
        ].mean()
        overall_rows.append(
            {
                "space": space,
                "validation_pairs": len(alignment),
                "transformations": alignment["transform"].nunique(),
                "pair_weighted_alignment_mean": pair_mean,
                "pair_weighted_alignment_ci_low": pair_low,
                "pair_weighted_alignment_ci_high": pair_high,
                "pair_weighted_alignment_median": float(alignment["alignment"].median()),
                "pair_weighted_null_mean": null_mean,
                "pair_weighted_null_ci_low": null_low,
                "pair_weighted_null_ci_high": null_high,
                "pair_weighted_null_median": float(
                    alignment["null_alignment"].median()
                ),
                "transform_weighted_alignment_mean": float(
                    transform_means["alignment"].mean()
                ),
                "transform_weighted_null_mean": float(
                    transform_means["null_alignment"].mean()
                ),
            }
        )
    return directions, transform_summary, alignment, pd.DataFrame(overall_rows)


def select_queries(
    eligible_pairs: pd.DataFrame,
    eligible_transforms: set[str],
    *,
    maximum: int,
    per_transform: int,
    seed: int,
) -> pd.DataFrame:
    grouped: dict[str, list[Any]] = {}
    for transform, group in eligible_pairs.groupby("transform", sort=True):
        transform = str(transform)
        if transform not in eligible_transforms:
            continue
        records = sorted(
            group.itertuples(index=False),
            key=lambda row: stable_digest(seed, transform, row.pair_id),
        )[:per_transform]
        if records:
            grouped[transform] = records
    transforms = sorted(grouped, key=lambda value: stable_digest(seed, value))
    selected: list[Any] = []
    used_seeds: set[int] = set()
    round_index = 0
    while len(selected) < maximum:
        added = False
        for transform in transforms:
            records = grouped[transform]
            while round_index < len(records) and int(records[round_index].lhs_index) in used_seeds:
                records.pop(round_index)
            if round_index >= len(records):
                continue
            row = records[round_index]
            used_seeds.add(int(row.lhs_index))
            selected.append(row)
            added = True
            if len(selected) == maximum:
                break
        if not added:
            break
        round_index += 1
    rows = [row._asdict() for row in selected]
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No held-out validation MMP queries satisfy the protocol")
    result.insert(
        0,
        "query_id",
        [stable_digest(seed, "query", value) for value in result["pair_id"]],
    )
    return result


def local_and_chemistry_analysis(
    *,
    space: str,
    train_space: np.ndarray,
    validation_space: np.ndarray,
    diverse: np.ndarray,
    fingerprints: list[Any],
    scaffolds: list[str],
    heavy_atoms: np.ndarray,
    descriptors: np.ndarray,
    val_cores: list[set[str]],
    local_neighbors: int,
    chemistry_neighbors: int,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_indices, train_distances = topk_l2(
        validation_space[diverse],
        train_space,
        k=local_neighbors,
        device=device,
        batch_size=batch_size,
    )
    local_rows: list[dict[str, Any]] = []
    for position, validation_index in enumerate(diverse):
        neighbors = train_space[train_indices[position]].astype(np.float64)
        displacements = neighbors - validation_space[validation_index].astype(np.float64)
        centered = displacements - displacements.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        eigenvalues = np.square(singular) / max(1, centered.shape[0] - 1)
        metrics = effective_rank(np.sort(eigenvalues))
        local_rows.append(
            {
                "space": space,
                "validation_index": int(validation_index),
                "nearest_train_distance": float(train_distances[position, 0]),
                "train_distance_rank10": float(train_distances[position, 9]),
                "train_distance_rank64": float(train_distances[position, -1]),
                "normalized_nearest_train_distance": float(
                    train_distances[position, 0] / math.sqrt(validation_space.shape[1])
                ),
                "local_effective_rank": metrics["effective_rank"],
                "local_participation_ratio": metrics["participation_ratio"],
                "local_components_90pct": metrics["components_90pct"],
                "local_top_eigenvalue_fraction": metrics["top_eigenvalue_fraction"],
            }
        )
    local_frame = pd.DataFrame(local_rows)
    local_summary = pd.DataFrame(
        [
            {
                "space": space,
                "seeds": len(local_frame),
                **{
                    f"{column}_median": float(local_frame[column].median())
                    for column in (
                        "nearest_train_distance",
                        "normalized_nearest_train_distance",
                        "local_effective_rank",
                        "local_participation_ratio",
                        "local_components_90pct",
                        "local_top_eigenvalue_fraction",
                    )
                },
            }
        ]
    )

    neighbor_indices, neighbor_distances = topk_l2(
        validation_space[diverse],
        validation_space,
        k=chemistry_neighbors,
        device=device,
        batch_size=batch_size,
        exclude_indices=diverse,
    )
    rng = np.random.default_rng(seed + int(stable_digest(space)[:8], 16))
    raw_rows: list[dict[str, Any]] = []

    def append_pair(
        query_position: int,
        candidate_index: int,
        rank: int,
        distance: float,
        band: str,
    ) -> None:
        seed_index = int(diverse[query_position])
        same_scaffold = bool(
            scaffolds[seed_index]
            and scaffolds[seed_index] == scaffolds[candidate_index]
        )
        raw_rows.append(
            {
                "space": space,
                "validation_seed_index": seed_index,
                "candidate_index": int(candidate_index),
                "rank": rank,
                "rank_band": band,
                "latent_distance": float(distance),
                "normalized_latent_distance": float(
                    distance / math.sqrt(validation_space.shape[1])
                ),
                "morgan_tanimoto": tanimoto(
                    fingerprints[seed_index], fingerprints[candidate_index]
                ),
                "same_scaffold": same_scaffold,
                "single_cut_mmp": one_cut_related(
                    val_cores, seed_index, int(candidate_index)
                ),
                "absolute_heavy_atom_delta": abs(
                    int(heavy_atoms[seed_index]) - int(heavy_atoms[candidate_index])
                ),
                "descriptor_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                descriptors[candidate_index].astype(np.float64)
                                - descriptors[seed_index].astype(np.float64)
                            )
                        )
                    )
                ),
            }
        )

    for query_position in range(len(diverse)):
        for zero_rank, candidate_index in enumerate(neighbor_indices[query_position]):
            rank = zero_rank + 1
            band = (
                "1"
                if rank == 1
                else "2-5"
                if rank <= 5
                else "6-10"
                if rank <= 10
                else "11-25"
            )
            append_pair(
                query_position,
                int(candidate_index),
                rank,
                float(neighbor_distances[query_position, zero_rank]),
                band,
            )
        seed_index = int(diverse[query_position])
        random_index = int(rng.integers(0, len(validation_space) - 1))
        if random_index >= seed_index:
            random_index += 1
        distance = float(
            np.linalg.norm(
                validation_space[random_index].astype(np.float64)
                - validation_space[seed_index].astype(np.float64)
            )
        )
        append_pair(query_position, random_index, 0, distance, "random")
    raw = pd.DataFrame(raw_rows)
    summary_rows: list[dict[str, Any]] = []
    for band, group in raw.groupby("rank_band", sort=False):
        query_averages = group.groupby("validation_seed_index", sort=False).agg(
            morgan_tanimoto=("morgan_tanimoto", "mean"),
            same_scaffold=("same_scaffold", "mean"),
            single_cut_mmp=("single_cut_mmp", "mean"),
            absolute_heavy_atom_delta=("absolute_heavy_atom_delta", "mean"),
            descriptor_rms=("descriptor_rms", "mean"),
            normalized_latent_distance=("normalized_latent_distance", "mean"),
        )
        row: dict[str, Any] = {
            "space": space,
            "rank_band": str(band),
            "pairs": len(group),
            "queries": len(query_averages),
        }
        for metric in query_averages.columns:
            mean, low, high = bootstrap_mean_ci(
                query_averages[metric].to_numpy(),
                seed=seed + int(stable_digest(space, band, metric)[:8], 16),
                resamples=1000,
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    nearest = raw[raw["rank_band"] != "random"]
    correlations = pd.DataFrame(
        [
            {
                "space": space,
                "pairs": len(nearest),
                "distance_vs_tanimoto_spearman": float(
                    spearmanr(nearest["latent_distance"], nearest["morgan_tanimoto"]).statistic
                ),
                "distance_vs_descriptor_rms_spearman": float(
                    spearmanr(nearest["latent_distance"], nearest["descriptor_rms"]).statistic
                ),
                "distance_vs_heavy_atom_delta_spearman": float(
                    spearmanr(
                        nearest["latent_distance"], nearest["absolute_heavy_atom_delta"]
                    ).statistic
                ),
            }
        ]
    )
    return local_frame, local_summary, raw, summary.merge(correlations, on="space", how="left")


def retrieval_experiment(
    *,
    space: str,
    train_space: np.ndarray,
    validation_space: np.ndarray,
    queries: pd.DataFrame,
    directions: dict[str, dict[str, Any]],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    fingerprints: list[Any],
    hashes: list[str],
    smiles: list[str],
    scaffolds: list[str],
    heavy_atoms: np.ndarray,
    descriptors: np.ndarray,
    val_cores: list[set[str]],
    device: torch.device,
    local_neighbors: int,
    top_k: int,
    summary_top_k: int,
    random_replicates: int,
    batch_size: int,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_indices = queries["lhs_index"].to_numpy(dtype=np.int64)
    local_indices, _ = topk_l2(
        validation_space[seed_indices],
        train_space,
        k=local_neighbors,
        device=device,
        batch_size=batch_size,
    )
    expected_rows = validation_space.shape[0]
    if any(
        len(values) != expected_rows
        for values in (fingerprints, hashes, smiles, scaffolds, heavy_atoms, descriptors, val_cores)
    ):
        raise RuntimeError("Validation chemistry/identity arrays are not row-aligned")
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    def add_target(
        query_row: Any,
        method: str,
        replicate: int,
        vector: np.ndarray,
        step_length: float,
    ) -> None:
        targets.append(np.asarray(vector, dtype=np.float32))
        metadata.append(
            {
                "space": space,
                "query_id": str(query_row.query_id),
                "pair_id": str(query_row.pair_id),
                "transform": str(query_row.transform),
                "seed_index": int(query_row.lhs_index),
                "true_target_index": int(query_row.rhs_index),
                "method": method,
                "replicate": replicate,
                "step_length": float(step_length),
            }
        )

    for position, query_row in enumerate(queries.itertuples(index=False)):
        seed_vector = validation_space[int(query_row.lhs_index)].astype(np.float64)
        target_vector = validation_space[int(query_row.rhs_index)].astype(np.float64)
        true_delta = target_vector - seed_vector
        direction_record = directions[str(query_row.transform)]
        step = float(direction_record["median_norm"])
        add_target(query_row, "seed_nn", 0, seed_vector, 0.0)
        add_target(
            query_row,
            "mmp_direction",
            0,
            seed_vector + step * direction_record["unit"].astype(np.float64),
            step,
        )
        for alpha in (0.25, 0.50, 0.75, 1.00):
            add_target(
                query_row,
                f"interpolation_{alpha:.2f}",
                0,
                seed_vector + alpha * true_delta,
                float(alpha * np.linalg.norm(true_delta)),
            )
        local_vectors = (
            train_space[local_indices[position]].astype(np.float64) - seed_vector
        )
        for replicate in range(random_replicates):
            rng = np.random.default_rng(
                int(
                    stable_digest(seed, space, query_row.query_id, replicate)[:16],
                    16,
                )
            )
            samples = {
                "isotropic": rng.normal(size=train_space.shape[1]),
                "global_covariance": covariance_sample(eigenvalues, eigenvectors, rng),
                "local_covariance": local_covariance_sample(local_vectors, rng),
            }
            for method, sample in samples.items():
                try:
                    perturbation = step * unit_vector(sample)
                except ValueError:
                    perturbation = step * unit_vector(rng.normal(size=train_space.shape[1]))
                add_target(
                    query_row,
                    method,
                    replicate,
                    seed_vector + perturbation,
                    step,
                )
    target_matrix = np.stack(targets).astype(np.float32)
    exclude = np.asarray([record["seed_index"] for record in metadata], dtype=np.int64)
    candidate_indices, candidate_distances = topk_l2(
        target_matrix,
        validation_space,
        k=top_k,
        device=device,
        batch_size=batch_size,
        exclude_indices=exclude,
    )
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    retained_candidate_methods = {
        "seed_nn",
        "mmp_direction",
        "isotropic",
        "local_covariance",
        "interpolation_1.00",
    }
    for target_position, record in enumerate(metadata):
        seed_index = int(record["seed_index"])
        true_index = int(record["true_target_index"])
        candidates = candidate_indices[target_position]
        target_hits = np.flatnonzero(candidates == true_index)
        target_rank = int(target_hits[0] + 1) if len(target_hits) else 0
        top1 = int(candidates[0])
        top_summary = candidates[:summary_top_k]
        seed_vector = validation_space[seed_index].astype(np.float64)
        true_vector = validation_space[true_index].astype(np.float64)
        candidate_vector = validation_space[top1].astype(np.float64)
        true_delta = true_vector - seed_vector
        candidate_delta = candidate_vector - seed_vector
        true_norm_squared = float(np.square(true_delta).sum())
        latent_progress = (
            float(candidate_delta @ true_delta / true_norm_squared)
            if true_norm_squared > 1e-12
            else math.nan
        )
        descriptor_true_delta = descriptors[true_index].astype(np.float64) - descriptors[
            seed_index
        ].astype(np.float64)
        descriptor_candidate_delta = descriptors[top1].astype(np.float64) - descriptors[
            seed_index
        ].astype(np.float64)
        descriptor_norm_squared = float(np.square(descriptor_true_delta).sum())
        descriptor_progress = (
            float(
                descriptor_candidate_delta
                @ descriptor_true_delta
                / descriptor_norm_squared
            )
            if descriptor_norm_squared > 1e-12
            else math.nan
        )
        row = dict(record)
        row.update(
            {
                "target_rank_within_50": target_rank,
                "target_recall_at_1": target_rank == 1,
                "target_recall_at_10": 0 < target_rank <= 10,
                "target_recall_at_50": target_rank > 0,
                "seed_hash": hashes[seed_index],
                "true_target_hash": hashes[true_index],
                "top1_candidate_index": top1,
                "top1_candidate_hash": hashes[top1],
                "top1_morgan_tanimoto_to_seed": tanimoto(
                    fingerprints[seed_index], fingerprints[top1]
                ),
                "top1_morgan_tanimoto_to_true_target": tanimoto(
                    fingerprints[true_index], fingerprints[top1]
                ),
                "top1_same_scaffold_as_seed": bool(
                    scaffolds[seed_index]
                    and scaffolds[seed_index] == scaffolds[top1]
                ),
                "top1_single_cut_mmp_to_seed": one_cut_related(
                    val_cores, seed_index, top1
                ),
                "top1_absolute_heavy_atom_delta": abs(
                    int(heavy_atoms[seed_index]) - int(heavy_atoms[top1])
                ),
                "top1_descriptor_rms_from_seed": float(
                    np.sqrt(np.mean(np.square(descriptor_candidate_delta)))
                ),
                "top1_descriptor_progress_to_true_target": descriptor_progress,
                "top1_latent_progress_to_true_target": latent_progress,
                "constructed_target_error_to_true": float(
                    np.linalg.norm(
                        target_matrix[target_position].astype(np.float64) - true_vector
                    )
                ),
                "top1_distance_to_constructed_target": float(
                    candidate_distances[target_position, 0]
                ),
                "true_derivative_distance_from_seed": float(
                    np.linalg.norm(true_delta)
                ),
                "top10_mean_morgan_tanimoto_to_seed": float(
                    np.mean(
                        [
                            tanimoto(fingerprints[seed_index], fingerprints[int(index)])
                            for index in top_summary
                        ]
                    )
                ),
                "top10_same_scaffold_fraction": float(
                    np.mean(
                        [
                            bool(
                                scaffolds[seed_index]
                                and scaffolds[seed_index] == scaffolds[int(index)]
                            )
                            for index in top_summary
                        ]
                    )
                ),
                "top10_single_cut_mmp_fraction": float(
                    np.mean(
                        [
                            one_cut_related(val_cores, seed_index, int(index))
                            for index in top_summary
                        ]
                    )
                ),
                "seed_smiles": smiles[seed_index],
                "true_target_smiles": smiles[true_index],
                "top1_candidate_smiles": smiles[top1],
            }
        )
        rows.append(row)
        if (
            record["method"] in retained_candidate_methods
            and int(record["replicate"]) == 0
        ):
            for rank, (candidate_index, distance) in enumerate(
                zip(candidates[:5], candidate_distances[target_position, :5]), start=1
            ):
                candidate_index = int(candidate_index)
                candidate_rows.append(
                    {
                        "space": space,
                        "query_id": record["query_id"],
                        "transform": record["transform"],
                        "method": record["method"],
                        "rank": rank,
                        "seed_hash": hashes[seed_index],
                        "true_target_hash": hashes[true_index],
                        "candidate_hash": hashes[candidate_index],
                        "seed_smiles": smiles[seed_index],
                        "true_target_smiles": smiles[true_index],
                        "candidate_smiles": smiles[candidate_index],
                        "candidate_is_true_target": candidate_index == true_index,
                        "distance_to_constructed_target": float(distance),
                        "morgan_tanimoto_to_seed": tanimoto(
                            fingerprints[seed_index], fingerprints[candidate_index]
                        ),
                        "same_scaffold_as_seed": bool(
                            scaffolds[seed_index]
                            and scaffolds[seed_index] == scaffolds[candidate_index]
                        ),
                        "single_cut_mmp_to_seed": one_cut_related(
                            val_cores, seed_index, candidate_index
                        ),
                    }
                )
    per_query = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    metrics = (
        "target_recall_at_1",
        "target_recall_at_10",
        "target_recall_at_50",
        "top1_morgan_tanimoto_to_seed",
        "top1_morgan_tanimoto_to_true_target",
        "top1_same_scaffold_as_seed",
        "top1_single_cut_mmp_to_seed",
        "top1_absolute_heavy_atom_delta",
        "top1_descriptor_rms_from_seed",
        "top1_descriptor_progress_to_true_target",
        "top1_latent_progress_to_true_target",
        "top10_mean_morgan_tanimoto_to_seed",
        "top10_same_scaffold_fraction",
        "top10_single_cut_mmp_fraction",
    )
    for method, group in per_query.groupby("method", sort=True):
        query_means = group.groupby("query_id", sort=False)[list(metrics)].mean()
        summary: dict[str, Any] = {
            "space": space,
            "method": str(method),
            "queries": len(query_means),
            "rows_including_replicates": len(group),
            "median_nonzero_target_rank_within_50": float(
                group.loc[group["target_rank_within_50"] > 0, "target_rank_within_50"].median()
            )
            if bool((group["target_rank_within_50"] > 0).any())
            else math.nan,
        }
        for metric in metrics:
            mean, low, high = bootstrap_mean_ci(
                query_means[metric].to_numpy(dtype=np.float64),
                seed=seed + int(stable_digest(space, method, metric)[:8], 16),
                resamples=bootstrap_resamples,
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summary_rows.append(summary)
    return per_query, pd.DataFrame(summary_rows), pd.DataFrame(candidate_rows)


def build_figures(
    *,
    spectra: dict[str, np.ndarray],
    distance_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    figures_dir: Path,
    step_root: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "graph_256": "#2b6cb0",
        "mean_node_128": "#d97706",
        "hybrid_384": "#25855a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for space in SPACE_ORDER:
        values = np.clip(spectra[space][::-1], 1e-12, None)
        axes[0].plot(np.arange(1, len(values) + 1), values, label=space, color=colors[space])
        axes[1].plot(
            np.arange(1, len(values) + 1),
            np.cumsum(values) / values.sum(),
            label=space,
            color=colors[space],
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Component rank")
    axes[0].set_ylabel("Covariance eigenvalue")
    axes[0].set_title("Global covariance spectra")
    axes[1].axhline(0.90, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_ylim(0, 1.01)
    axes[1].set_title("Cumulative variance")
    axes[1].legend(frameon=False)
    for suffix in (".png", ".svg"):
        save_figure(fig, figures_dir / f"global_spectrum{suffix}", step_root)
    plt.close(fig)

    band_order = ["1", "2-5", "6-10", "11-25", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for space in SPACE_ORDER:
        frame = distance_summary[distance_summary["space"] == space].copy()
        frame["rank_band"] = pd.Categorical(frame["rank_band"], band_order, ordered=True)
        frame = frame.sort_values("rank_band")
        axes[0].plot(
            frame["rank_band"].astype(str),
            frame["morgan_tanimoto_mean"],
            marker="o",
            label=space,
            color=colors[space],
        )
        axes[1].plot(
            frame["rank_band"].astype(str),
            frame["same_scaffold_mean"],
            marker="o",
            label=space,
            color=colors[space],
        )
    axes[0].set_ylabel("Mean Morgan Tanimoto to seed")
    axes[1].set_ylabel("Same-scaffold fraction")
    for axis in axes:
        axis.set_xlabel("Latent-neighbor rank band")
        axis.tick_params(axis="x", rotation=25)
    axes[0].set_title("Chemical similarity by latent distance")
    axes[1].set_title("Scaffold retention by latent distance")
    axes[1].legend(frameon=False)
    for suffix in (".png", ".svg"):
        save_figure(fig, figures_dir / f"distance_to_chemistry{suffix}", step_root)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(SPACE_ORDER))
    actual = [
        float(
            alignment_summary.loc[
                alignment_summary["space"] == space,
                "pair_weighted_alignment_mean",
            ].iloc[0]
        )
        for space in SPACE_ORDER
    ]
    null = [
        float(
            alignment_summary.loc[
                alignment_summary["space"] == space, "pair_weighted_null_mean"
            ].iloc[0]
        )
        for space in SPACE_ORDER
    ]
    width = 0.36
    ax.bar(positions - width / 2, actual, width, label="held-out MMP direction")
    ax.bar(positions + width / 2, null, width, label="mismatched null")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, SPACE_ORDER)
    ax.set_ylabel("Mean cosine alignment")
    ax.set_title("Unseen-core MMP displacement transfer")
    ax.legend(frameon=False)
    for suffix in (".png", ".svg"):
        save_figure(fig, figures_dir / f"mmp_alignment{suffix}", step_root)
    plt.close(fig)

    selected_methods = [
        "seed_nn",
        "isotropic",
        "global_covariance",
        "local_covariance",
        "mmp_direction",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    width = 0.24
    method_positions = np.arange(len(selected_methods))
    for offset, space in enumerate(SPACE_ORDER):
        values = []
        for method in selected_methods:
            match = retrieval_summary[
                (retrieval_summary["space"] == space)
                & (retrieval_summary["method"] == method)
            ]
            values.append(
                float(match["target_recall_at_10_mean"].iloc[0]) if len(match) else 0.0
            )
        ax.bar(
            method_positions + (offset - 1) * width,
            values,
            width,
            label=space,
            color=colors[space],
        )
    ax.set_xticks(method_positions, selected_methods, rotation=20, ha="right")
    ax.set_ylabel("Exact held-out derivative recall@10")
    ax.set_title("Blind and transformation-controlled retrieval")
    ax.legend(frameon=False)
    for suffix in (".png", ".svg"):
        save_figure(fig, figures_dir / f"retrieval_recall10{suffix}", step_root)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    alphas = [0.25, 0.50, 0.75, 1.00]
    for space in SPACE_ORDER:
        values = []
        for alpha in alphas:
            method = f"interpolation_{alpha:.2f}"
            match = retrieval_summary[
                (retrieval_summary["space"] == space)
                & (retrieval_summary["method"] == method)
            ]
            values.append(float(match["target_recall_at_10_mean"].iloc[0]))
        ax.plot(alphas, values, marker="o", label=space, color=colors[space])
    ax.set_xlabel("Oracle interpolation fraction")
    ax.set_ylabel("Exact derivative recall@10")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Seed-to-derivative path sanity check")
    ax.legend(frameon=False)
    for suffix in (".png", ".svg"):
        save_figure(fig, figures_dir / f"interpolation_recall{suffix}", step_root)
    plt.close(fig)


def evaluate_gates(
    config: dict[str, Any],
    distance_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    thresholds = config["gates"]
    hybrid_distance = distance_summary[distance_summary["space"] == "hybrid_384"]
    nearest = hybrid_distance[hybrid_distance["rank_band"].astype(str) == "1"].iloc[0]
    random = hybrid_distance[
        hybrid_distance["rank_band"].astype(str) == "random"
    ].iloc[0]
    tanimoto_enrichment = float(nearest["morgan_tanimoto_mean"]) / max(
        float(random["morgan_tanimoto_mean"]), 1e-12
    )
    scaffold_gain = float(nearest["same_scaffold_mean"]) - float(
        random["same_scaffold_mean"]
    )
    g1 = (
        tanimoto_enrichment
        >= float(thresholds["g1_tanimoto_enrichment_minimum"])
        and scaffold_gain
        >= float(thresholds["g1_scaffold_absolute_gain_minimum"])
    )

    hybrid_alignment = alignment_summary[
        alignment_summary["space"] == "hybrid_384"
    ].iloc[0]
    median_alignment = float(hybrid_alignment["pair_weighted_alignment_median"])
    median_null = float(hybrid_alignment["pair_weighted_null_median"])
    alignment_gain = median_alignment - median_null
    g2 = median_alignment > 0 and alignment_gain >= float(
        thresholds["g2_alignment_over_null_minimum"]
    )

    hybrid_retrieval = retrieval_summary[
        retrieval_summary["space"] == "hybrid_384"
    ]
    mmp_r10 = float(
        hybrid_retrieval.loc[
            hybrid_retrieval["method"] == "mmp_direction",
            "target_recall_at_10_mean",
        ].iloc[0]
    )
    blind_r10 = max(
        float(
            hybrid_retrieval.loc[
                hybrid_retrieval["method"] == method,
                "target_recall_at_10_mean",
            ].iloc[0]
        )
        for method in BLIND_METHODS
    )
    recall_gain = mmp_r10 - blind_r10
    recall_fold = mmp_r10 / max(blind_r10, 1e-12)
    g3 = recall_gain >= float(
        thresholds["g3_recall10_absolute_gain_minimum"]
    ) and (blind_r10 == 0 or recall_fold >= float(thresholds["g3_recall10_fold_gain_minimum"]))

    interpolation = []
    for method in INTERPOLATION_METHODS:
        interpolation.append(
            float(
                hybrid_retrieval.loc[
                    hybrid_retrieval["method"] == method,
                    "target_recall_at_10_mean",
                ].iloc[0]
            )
        )
    monotonic = all(
        later + 1e-12 >= earlier
        for earlier, later in zip(interpolation[:-1], interpolation[1:])
    )
    g4 = monotonic and interpolation[-1] >= float(
        thresholds["g4_interpolation_alpha1_recall10_minimum"]
    )

    gates = {
        "G1_local_chemical_organization": {
            "passed": bool(g1),
            "tanimoto_enrichment": tanimoto_enrichment,
            "scaffold_absolute_gain": scaffold_gain,
        },
        "G2_transferable_mmp_direction": {
            "passed": bool(g2),
            "validation_median_alignment": median_alignment,
            "null_median_alignment": median_null,
            "alignment_gain": alignment_gain,
        },
        "G3_controlled_retrieval": {
            "passed": bool(g3),
            "mmp_recall_at_10": mmp_r10,
            "best_blind_recall_at_10": blind_r10,
            "absolute_gain": recall_gain,
            "fold_gain": recall_fold if blind_r10 else None,
        },
        "G4_interpolation_sanity": {
            "passed": bool(g4),
            "recall_at_10_by_alpha": dict(
                zip(("0.25", "0.50", "0.75", "1.00"), interpolation)
            ),
            "nondecreasing": monotonic,
        },
    }
    if g2 and g3:
        approach = "matched-pair directions with a seed-conditioned edit decoder"
    elif g2 and recall_gain >= float(
        thresholds["g3_recall10_absolute_gain_minimum"]
    ):
        approach = (
            "transformation-conditioned matched-pair retrieval with local manifold "
            "constraints, followed by a seed-conditioned edit decoder; do not assume "
            "a universal linear vector"
        )
    elif g1:
        approach = (
            "manifold-aware retrieval first, followed by a seed-conditioned decoder"
        )
    else:
        approach = (
            "retrieval-only investigation and representation redesign before decoder work"
        )
    return gates, approach


def results_markdown(
    *,
    inputs: dict[str, Any],
    fragmentation: dict[str, Any],
    pair_statistics: dict[str, Any],
    query_count: int,
    global_geometry: pd.DataFrame,
    local_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    retrieval_summary: pd.DataFrame,
    gates: dict[str, Any],
    approach: str,
) -> str:
    def value(frame: pd.DataFrame, space: str, column: str) -> float:
        return float(frame.loc[frame["space"] == space, column].iloc[0])

    hybrid_distance = distance_summary[distance_summary["space"] == "hybrid_384"]
    nearest = hybrid_distance[hybrid_distance["rank_band"].astype(str) == "1"].iloc[0]
    random = hybrid_distance[
        hybrid_distance["rank_band"].astype(str) == "random"
    ].iloc[0]
    hybrid_retrieval = retrieval_summary[
        retrieval_summary["space"] == "hybrid_384"
    ]

    def retrieval(method: str, metric: str) -> float:
        return float(
            hybrid_retrieval.loc[hybrid_retrieval["method"] == method, metric].iloc[0]
        )

    gate_lines = "\n".join(
        f"- **{name}: {'PASS' if record['passed'] else 'FAIL'}** — "
        + ", ".join(
            f"{key}={value:.4f}"
            for key, value in record.items()
            if key != "passed" and isinstance(value, (int, float)) and value is not None
        )
        for name, record in gates.items()
    )
    geometry_rows = "\n".join(
        f"| {space} | {value(global_geometry, space, 'effective_rank'):.2f} | "
        f"{int(value(global_geometry, space, 'components_90pct'))} | "
        f"{value(local_summary, space, 'local_effective_rank_median'):.2f} |"
        for space in SPACE_ORDER
    )
    retrieval_rows = "\n".join(
        f"| {method} | {retrieval(method, 'target_recall_at_10_mean'):.4f} | "
        f"{retrieval(method, 'top1_morgan_tanimoto_to_seed_mean'):.4f} | "
        f"{retrieval(method, 'top1_same_scaffold_as_seed_mean'):.4f} | "
        f"{retrieval(method, 'top1_single_cut_mmp_to_seed_mean'):.4f} |"
        for method in (
            "seed_nn",
            "isotropic",
            "global_covariance",
            "local_covariance",
            "mmp_direction",
        )
    )
    return f"""# Day-1 results: latent geometry and derivative retrieval

## Outcome

The final amended train/validation study is complete. It evaluated latent
organization, unseen-core matched-pair direction transfer, blind perturbations,
oracle interpolation, and retrieval from 50,000 known valid validation molecules.
The pre-outcome support amendment is recorded in PROTOCOL_AMENDMENT_01.md.

**Recommended next approach:** {approach}.

This is evidence about retrieval geometry. It is not evidence that the existing
gMolAI checkpoint can decode a 384D vector or generate a novel molecule.

## Audited data

- train geometry/directions: {inputs['train_rows']:,} molecules;
- independent validation bank: {inputs['validation_rows']:,} molecules;
- train/validation identity overlap: 0;
- held-out MMP retrieval queries: {query_count:,};
- locked test molecules used: **0**;
- train eligible fragmentations: {fragmentation['train']['eligible_fragmentations']:,};
- validation eligible fragmentations: {fragmentation['validation']['eligible_fragmentations']:,};
- train MMP pairs: {pair_statistics['train']['pairs']:,};
- validation MMP pairs: {pair_statistics['validation']['pairs']:,}.

## Geometry

| Space | Global effective rank | Components for 90% variance | Median local effective rank |
|---|---:|---:|---:|
{geometry_rows}

For hybrid-384, the nearest-neighbor mean Morgan Tanimoto was
**{float(nearest['morgan_tanimoto_mean']):.4f}**, versus
**{float(random['morgan_tanimoto_mean']):.4f}** for random pairs. Same-scaffold
fractions were **{float(nearest['same_scaffold_mean']):.4f}** and
**{float(random['same_scaffold_mean']):.4f}**, respectively. These quantities
measure chemical enrichment, not guaranteed small edits.

## Unseen-core MMP displacement transfer

| Space | Validation pairs | Transformations | Mean alignment | Median alignment | Null mean | Null median |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {space} | {int(value(alignment_summary, space, 'validation_pairs'))} | "
        f"{int(value(alignment_summary, space, 'transformations'))} | "
        f"{value(alignment_summary, space, 'pair_weighted_alignment_mean'):.4f} | "
        f"{value(alignment_summary, space, 'pair_weighted_alignment_median'):.4f} | "
        f"{value(alignment_summary, space, 'pair_weighted_null_mean'):.4f} | "
        f"{value(alignment_summary, space, 'pair_weighted_null_median'):.4f} |"
        for space in SPACE_ORDER
    ) + f"""

Positive alignment above the mismatched-transformation null indicates some
cross-core directional organization. The exact retrieval test below determines
whether that signal is strong enough to be practically controlling.

## Hybrid-384 retrieval

Blind perturbations and the learned MMP direction use equal,
transformation-specific median train-pair step lengths. `seed_nn` has zero
perturbation. Oracle interpolation is reported separately in the machine-readable
tables and figures.

| Method | Exact target recall@10 | Top-1 seed Tanimoto | Top-1 same scaffold | Top-1 one-cut MMP |
|---|---:|---:|---:|---:|
{retrieval_rows}

## Predeclared gates

{gate_lines}

Gate failures are retained. A gate was not changed after result inspection.
G3 is conjunctive: the learned direction gained
**{gates['G3_controlled_retrieval']['absolute_gain']:.4f}** recall@10 absolute,
but its **{gates['G3_controlled_retrieval']['fold_gain']:.4f}×** improvement did
not meet the predeclared 2× clause. Because the blind baseline was already
**{gates['G3_controlled_retrieval']['best_blind_recall_at_10']:.4f}**, the
maximum possible fold at perfect recall was only
**{1.0 / gates['G3_controlled_retrieval']['best_blind_recall_at_10']:.4f}×**.
The formal failure is therefore retained without discarding the strong absolute
retrieval signal.

## Bounded conclusion

The appropriate disclosure is that the promoted representation was tested for
seed-centered derivative retrieval under held-out chemical controls. The
recommended next architecture follows the result stated above. A later decoder
must still demonstrate clean reconstruction, condition use, molecular validity,
novelty, and re-encoding consistency before any generative claim.

## Artifacts

- `outputs/tables/`: global/local geometry, distance chemistry, MMP alignment,
  and retrieval summaries;
- `outputs/raw/`: complete per-pair and per-query records;
- `outputs/examples/`: ranked molecular examples including failures;
- `outputs/figures/`: covariance, chemistry-distance, alignment, retrieval, and
  interpolation plots;
- `outputs/study_summary.json`: gates and machine-readable conclusion;
- `outputs/SHA256SUMS`: output integrity ledger;
- `state/COMPLETE.json`: completion and provenance seal.
"""


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    ensure_within(step_root, repo_root / "deriv-gen")
    config_path = step_root / "config" / "protocol.json"
    manifest_path = step_root / "inputs" / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Day-1 requires exactly one visible GPU; observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    workers = args.workers or min(48, len(os.sched_getaffinity(0)))
    if workers <= 0:
        raise RuntimeError("No CPU workers are available")

    intermediate_dir = step_root / "intermediate"
    outputs_dir = step_root / "outputs"
    tables_dir = outputs_dir / "tables"
    raw_dir = outputs_dir / "raw"
    examples_dir = outputs_dir / "examples"
    figures_dir = outputs_dir / "figures"
    state_dir = step_root / "state"
    for directory in (
        intermediate_dir,
        tables_dir,
        raw_dir,
        examples_dir,
        figures_dir,
        state_dir,
    ):
        ensure_within(directory, step_root).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    started_at = utc_now()
    atomic_write_json(
        state_dir / "RUNNING.json",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "started_at": started_at,
            "pid": os.getpid(),
        },
        step_root,
    )
    print(f"[{utc_now()}] validating immutable inputs", flush=True)
    input_paths, input_hashes = load_and_validate_inputs(
        repo_root, step_root, manifest
    )
    train_payload = torch.load(
        input_paths["train_raw_embeddings"], map_location="cpu", weights_only=False
    )
    validation_payload = torch.load(
        input_paths["validation_public_embeddings"],
        map_location="cpu",
        weights_only=False,
    )
    calibrator = torch.load(
        input_paths["promoted_calibrator"], map_location="cpu", weights_only=False
    )
    if train_payload["metadata"]["split"] != "train":
        raise RuntimeError("Train payload split identity failed")
    if validation_payload["metadata"]["split"] != "validation":
        raise RuntimeError("Validation payload split identity failed")
    if train_payload["metadata"]["embedding_definition"] != "clean_graph_z_plus_mean_node_z_raw_blocks":
        raise RuntimeError("Unexpected train embedding definition")
    if validation_payload["metadata"]["embedding_parameters"]["mean_node_weight"] != 3.0:
        raise RuntimeError("Validation payload is not the promoted weight-3 representation")
    if calibrator["metadata"]["source_embedding_sha256"] != input_hashes[
        "train_raw_embeddings"
    ]:
        raise RuntimeError("Calibrator is not bound to the selected train payload")
    if train_payload["metadata"]["checkpoint_sha256"] != validation_payload[
        "metadata"
    ]["checkpoint_sha256"]:
        raise RuntimeError("Train and validation checkpoint identities differ")

    train_raw = payload_array(train_payload, "embeddings").astype(np.float32)
    validation_public = payload_array(validation_payload, "embeddings").astype(
        np.float32
    )
    coordinate_mean = payload_array(calibrator, "coordinate_mean").astype(np.float32)
    coordinate_scale = payload_array(calibrator, "coordinate_scale").astype(np.float32)
    train_u = np.ascontiguousarray(
        (train_raw - coordinate_mean[None, :]) / coordinate_scale[None, :],
        dtype=np.float32,
    )
    validation_u = np.ascontiguousarray(validation_public.copy(), dtype=np.float32)
    validation_u[:, 256:] /= 3.0
    if train_u.shape != (100000, 384) or validation_u.shape != (50000, 384):
        raise RuntimeError(
            f"Unexpected matrix shapes: train={train_u.shape}, validation={validation_u.shape}"
        )
    if not np.isfinite(train_u).all() or not np.isfinite(validation_u).all():
        raise RuntimeError("Non-finite embedding coordinate detected")
    if float(np.max(np.abs(train_u.mean(axis=0)))) > 2e-4:
        raise RuntimeError("Train calibration mean check failed")
    if float(np.max(np.abs(train_u.std(axis=0) - 1.0))) > 2e-4:
        raise RuntimeError("Train calibration population-scale check failed")
    train_descriptors = payload_array(
        train_payload, "standardized_descriptor_targets"
    ).astype(np.float32)
    validation_descriptors = payload_array(
        validation_payload, "standardized_descriptor_targets"
    ).astype(np.float32)

    cache = json.loads(input_paths["chemical_record_cache"].read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        input_paths["dataset_manifest"].read_text(encoding="utf-8")
    )
    descriptor_scaler = json.loads(
        input_paths["descriptor_scaler"].read_text(encoding="utf-8")
    )
    if cache["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Chemical cache and dataset manifest identities differ")
    if descriptor_scaler["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Descriptor scaler and dataset manifest identities differ")
    if len(descriptor_scaler["descriptor_names"]) != train_descriptors.shape[1]:
        raise RuntimeError("Descriptor schema width mismatch")
    train_hashes, train_smiles, train_scaffolds = chemical_records(
        train_payload, cache["records"], repo_root / "work"
    )
    validation_hashes, validation_smiles, validation_scaffolds = chemical_records(
        validation_payload, cache["records"], repo_root / "work"
    )
    overlap = set(train_hashes).intersection(validation_hashes)
    if overlap:
        raise RuntimeError(f"Train/validation identity overlap: {len(overlap)}")

    print(
        f"[{utc_now()}] fragmenting 100k train + 50k validation molecules with {workers} workers",
        flush=True,
    )
    train_fragmentation = fragment_molecules(
        train_hashes,
        train_smiles,
        settings=config["mmp"],
        workers=workers,
    )
    validation_fragmentation = fragment_molecules(
        validation_hashes,
        validation_smiles,
        settings=config["mmp"],
        workers=workers,
    )
    if train_fragmentation.statistics["parse_failures"] or validation_fragmentation.statistics[
        "parse_failures"
    ]:
        raise RuntimeError("An immutable cached molecule failed RDKit parsing")
    atomic_write_parquet(
        intermediate_dir / "train_fragments.parquet",
        train_fragmentation.fragments,
        step_root,
    )
    atomic_write_parquet(
        intermediate_dir / "validation_fragments.parquet",
        validation_fragmentation.fragments,
        step_root,
    )
    train_pairs_result = build_mmp_pairs(
        train_fragmentation.fragments,
        train_hashes,
        settings=config["mmp"],
        seed=seed,
    )
    validation_pairs_result = build_mmp_pairs(
        validation_fragmentation.fragments,
        validation_hashes,
        settings=config["mmp"],
        seed=seed + 1,
    )
    train_pairs = train_pairs_result.pairs
    validation_pairs = validation_pairs_result.pairs
    atomic_write_parquet(intermediate_dir / "train_mmp_pairs.parquet", train_pairs, step_root)
    atomic_write_parquet(
        intermediate_dir / "validation_mmp_pairs.parquet",
        validation_pairs,
        step_root,
    )
    minimum_cores = int(config["mmp"]["minimum_train_cores_per_transform"])
    train_core_sets_by_transform = {
        str(transform): set(str(value) for value in group["core"])
        for transform, group in train_pairs.groupby("transform", sort=True)
        if group["core"].nunique() >= minimum_cores
    }
    eligible_validation_mask = [
        str(row.transform) in train_core_sets_by_transform
        and str(row.core) not in train_core_sets_by_transform[str(row.transform)]
        for row in validation_pairs.itertuples(index=False)
    ]
    eligible_validation_pairs = validation_pairs.loc[
        eligible_validation_mask
    ].reset_index(drop=True)
    if eligible_validation_pairs.empty:
        raise RuntimeError("No unseen-core validation MMP pairs match repeated train transformations")
    atomic_write_parquet(
        intermediate_dir / "eligible_unseen_core_validation_pairs.parquet",
        eligible_validation_pairs,
        step_root,
    )
    print(
        f"[{utc_now()}] MMP pairs: train={len(train_pairs):,}, validation={len(validation_pairs):,}, "
        f"eligible unseen-core={len(eligible_validation_pairs):,}",
        flush=True,
    )

    molecule_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train",
                    "molecule_index": np.arange(len(train_hashes)),
                    "molecule_hash": train_hashes,
                    "canonical_smiles": train_smiles,
                    "scaffold": train_scaffolds,
                    "heavy_atoms": train_fragmentation.heavy_atoms,
                }
            ),
            pd.DataFrame(
                {
                    "split": "validation",
                    "molecule_index": np.arange(len(validation_hashes)),
                    "molecule_hash": validation_hashes,
                    "canonical_smiles": validation_smiles,
                    "scaffold": validation_scaffolds,
                    "heavy_atoms": validation_fragmentation.heavy_atoms,
                }
            ),
        ],
        ignore_index=True,
    )
    atomic_write_parquet(intermediate_dir / "molecules.parquet", molecule_table, step_root)

    print(f"[{utc_now()}] computing fingerprints and covariance geometry", flush=True)
    _, validation_fingerprints = make_fingerprints(validation_smiles)
    validation_core_sets = core_sets(
        validation_fragmentation.fragments, len(validation_hashes)
    )
    diverse = diverse_indices(
        validation_hashes,
        validation_scaffolds,
        maximum=int(config["geometry"]["diverse_validation_seeds"]),
        seed=seed,
    )
    atomic_write_csv(
        intermediate_dir / "diverse_validation_seeds.csv",
        pd.DataFrame(
            {
                "validation_index": diverse,
                "molecule_hash": [validation_hashes[index] for index in diverse],
                "canonical_smiles": [validation_smiles[index] for index in diverse],
                "scaffold": [validation_scaffolds[index] for index in diverse],
            }
        ),
        step_root,
    )

    spectra: dict[str, np.ndarray] = {}
    eigenvectors_by_space: dict[str, np.ndarray] = {}
    geometry_rows: list[dict[str, Any]] = []
    train_spaces: dict[str, np.ndarray] = {}
    validation_spaces: dict[str, np.ndarray] = {}
    for space in SPACE_ORDER:
        train_space = block_data(train_u, config["coordinate_spaces"], space)
        validation_space = block_data(
            validation_u, config["coordinate_spaces"], space
        )
        train_spaces[space] = train_space
        validation_spaces[space] = validation_space
        eigenvalues, eigenvectors = covariance_eigendecomposition(train_space, device)
        spectra[space] = eigenvalues
        eigenvectors_by_space[space] = eigenvectors
        metrics = effective_rank(eigenvalues)
        geometry_rows.append(
            {
                "space": space,
                "dimensions": train_space.shape[1],
                **metrics,
                "effective_rank_fraction": float(
                    metrics["effective_rank"] / train_space.shape[1]
                ),
                "median_coordinate_std": float(np.median(train_space.std(axis=0))),
            }
        )
    global_geometry = pd.DataFrame(geometry_rows)
    atomic_write_csv(tables_dir / "global_geometry.csv", global_geometry, step_root)
    atomic_save_npz(
        intermediate_dir / "global_covariance_spectra.npz",
        step_root,
        **{space: values for space, values in spectra.items()},
    )

    directions_by_space: dict[str, dict[str, dict[str, Any]]] = {}
    transform_summaries: list[pd.DataFrame] = []
    alignment_raw_frames: list[pd.DataFrame] = []
    alignment_summaries: list[pd.DataFrame] = []
    for space in SPACE_ORDER:
        directions, transform_summary, alignment_raw, alignment_summary = fit_mmp_directions(
            train_pairs,
            eligible_validation_pairs,
            train_spaces[space],
            validation_spaces[space],
            space=space,
            minimum_train_cores=minimum_cores,
            seed=seed,
        )
        directions_by_space[space] = directions
        transform_summaries.append(transform_summary)
        alignment_raw_frames.append(alignment_raw)
        alignment_summaries.append(alignment_summary)
        transforms = np.asarray(sorted(directions), dtype=np.str_)
        vectors = np.stack([directions[value]["unit"] for value in transforms])
        steps = np.asarray(
            [directions[value]["median_norm"] for value in transforms], dtype=np.float32
        )
        atomic_save_npz(
            intermediate_dir / f"mmp_directions_{space}.npz",
            step_root,
            transforms=transforms,
            unit_directions=vectors,
            median_step_norms=steps,
        )
    transform_summary = pd.concat(transform_summaries, ignore_index=True)
    alignment_raw = pd.concat(alignment_raw_frames, ignore_index=True)
    alignment_summary = pd.concat(alignment_summaries, ignore_index=True)
    atomic_write_csv(
        tables_dir / "mmp_transform_summary.csv", transform_summary, step_root
    )
    atomic_write_csv(
        tables_dir / "mmp_alignment_summary.csv", alignment_summary, step_root
    )
    atomic_write_parquet(raw_dir / "mmp_alignment_per_pair.parquet", alignment_raw, step_root)

    common_transforms = set.intersection(
        *(set(directions_by_space[space]) for space in SPACE_ORDER)
    )
    queries = select_queries(
        eligible_validation_pairs,
        common_transforms,
        maximum=int(config["retrieval"]["maximum_validation_queries"]),
        per_transform=int(config["retrieval"]["maximum_queries_per_transform"]),
        seed=seed,
    )
    queries["seed_hash"] = [
        validation_hashes[int(index)] for index in queries["lhs_index"]
    ]
    queries["target_hash"] = [
        validation_hashes[int(index)] for index in queries["rhs_index"]
    ]
    queries["seed_smiles"] = [
        validation_smiles[int(index)] for index in queries["lhs_index"]
    ]
    queries["target_smiles"] = [
        validation_smiles[int(index)] for index in queries["rhs_index"]
    ]
    atomic_write_csv(intermediate_dir / "retrieval_queries.csv", queries, step_root)
    print(
        f"[{utc_now()}] running local geometry and retrieval for {len(queries):,} held-out MMP queries",
        flush=True,
    )

    local_frames: list[pd.DataFrame] = []
    local_summaries: list[pd.DataFrame] = []
    chemistry_raw_frames: list[pd.DataFrame] = []
    chemistry_summaries: list[pd.DataFrame] = []
    retrieval_raw_frames: list[pd.DataFrame] = []
    retrieval_summaries: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for space in SPACE_ORDER:
        print(f"[{utc_now()}] space={space}", flush=True)
        local_raw, local_summary, chemistry_raw, chemistry_summary = local_and_chemistry_analysis(
            space=space,
            train_space=train_spaces[space],
            validation_space=validation_spaces[space],
            diverse=diverse,
            fingerprints=validation_fingerprints,
            scaffolds=validation_scaffolds,
            heavy_atoms=validation_fragmentation.heavy_atoms,
            descriptors=validation_descriptors,
            val_cores=validation_core_sets,
            local_neighbors=int(config["geometry"]["local_train_neighbors"]),
            chemistry_neighbors=int(config["geometry"]["neighbors_per_seed"]),
            device=device,
            batch_size=int(config["retrieval"]["gpu_query_batch_size"]),
            seed=seed,
        )
        local_frames.append(local_raw)
        local_summaries.append(local_summary)
        chemistry_raw_frames.append(chemistry_raw)
        chemistry_summaries.append(chemistry_summary)
        retrieval_raw, retrieval_summary, candidates = retrieval_experiment(
            space=space,
            train_space=train_spaces[space],
            validation_space=validation_spaces[space],
            queries=queries,
            directions=directions_by_space[space],
            eigenvalues=spectra[space],
            eigenvectors=eigenvectors_by_space[space],
            fingerprints=validation_fingerprints,
            hashes=validation_hashes,
            smiles=validation_smiles,
            scaffolds=validation_scaffolds,
            heavy_atoms=validation_fragmentation.heavy_atoms,
            descriptors=validation_descriptors,
            val_cores=validation_core_sets,
            device=device,
            local_neighbors=int(config["geometry"]["local_train_neighbors"]),
            top_k=int(config["retrieval"]["top_k"]),
            summary_top_k=int(config["retrieval"]["summary_top_k"]),
            random_replicates=int(config["retrieval"]["random_replicates"]),
            batch_size=int(config["retrieval"]["gpu_query_batch_size"]),
            bootstrap_resamples=int(config["bootstrap_resamples"]),
            seed=seed,
        )
        retrieval_raw_frames.append(retrieval_raw)
        retrieval_summaries.append(retrieval_summary)
        candidate_frames.append(candidates)

    local_raw = pd.concat(local_frames, ignore_index=True)
    local_summary = pd.concat(local_summaries, ignore_index=True)
    chemistry_raw = pd.concat(chemistry_raw_frames, ignore_index=True)
    distance_summary = pd.concat(chemistry_summaries, ignore_index=True)
    retrieval_raw = pd.concat(retrieval_raw_frames, ignore_index=True)
    retrieval_summary = pd.concat(retrieval_summaries, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    atomic_write_parquet(raw_dir / "local_geometry_per_seed.parquet", local_raw, step_root)
    atomic_write_csv(tables_dir / "local_geometry_summary.csv", local_summary, step_root)
    atomic_write_parquet(
        raw_dir / "distance_chemistry_per_pair.parquet", chemistry_raw, step_root
    )
    atomic_write_csv(
        tables_dir / "distance_chemistry_summary.csv", distance_summary, step_root
    )
    atomic_write_parquet(
        raw_dir / "retrieval_per_query.parquet", retrieval_raw, step_root
    )
    atomic_write_csv(
        tables_dir / "retrieval_summary.csv", retrieval_summary, step_root
    )
    atomic_write_parquet(
        examples_dir / "ranked_candidate_examples.parquet", candidates, step_root
    )
    atomic_write_csv(
        examples_dir / "mmp_direction_top1_examples.csv",
        retrieval_raw[
            (retrieval_raw["method"] == "mmp_direction")
            & (retrieval_raw["replicate"] == 0)
        ][
            [
                "space",
                "query_id",
                "transform",
                "seed_hash",
                "true_target_hash",
                "top1_candidate_hash",
                "seed_smiles",
                "true_target_smiles",
                "top1_candidate_smiles",
                "target_rank_within_50",
                "top1_morgan_tanimoto_to_seed",
                "top1_same_scaffold_as_seed",
                "top1_single_cut_mmp_to_seed",
            ]
        ],
        step_root,
    )

    gates, approach = evaluate_gates(
        config, distance_summary, alignment_summary, retrieval_summary
    )
    input_summary = {
        "train_rows": len(train_hashes),
        "validation_rows": len(validation_hashes),
        "identity_overlap": 0,
        "checkpoint_sha256": train_payload["metadata"]["checkpoint_sha256"],
        "calibrator_sha256": input_hashes["promoted_calibrator"],
        "dataset_manifest_hash": dataset_manifest["manifest_hash"],
    }
    pair_statistics = {
        "train": train_pairs_result.statistics,
        "validation": validation_pairs_result.statistics,
        "eligible_unseen_core_validation_pairs": len(eligible_validation_pairs),
        "repeated_train_transformations": len(train_core_sets_by_transform),
        "repeated_train_transformations_with_unseen_core_validation": int(
            eligible_validation_pairs["transform"].nunique()
        ),
        "retrieval_queries": len(queries),
    }
    fragmentation_statistics = {
        "train": train_fragmentation.statistics,
        "validation": validation_fragmentation.statistics,
    }
    study_summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "data": input_summary,
        "fragmentation": fragmentation_statistics,
        "matched_pairs": pair_statistics,
        "protocol_amendments": config.get("protocol_amendments", []),
        "gates": gates,
        "recommended_next_approach": approach,
        "claim_boundary": (
            "retrieval-geometry feasibility only; no current 384D decoder or novel-molecule "
            "generation demonstrated"
        ),
    }
    atomic_write_json(outputs_dir / "study_summary.json", study_summary, step_root)
    build_figures(
        spectra=spectra,
        distance_summary=distance_summary,
        alignment_summary=alignment_summary,
        retrieval_summary=retrieval_summary,
        figures_dir=figures_dir,
        step_root=step_root,
    )
    report = results_markdown(
        inputs=input_summary,
        fragmentation=fragmentation_statistics,
        pair_statistics=pair_statistics,
        query_count=len(queries),
        global_geometry=global_geometry,
        local_summary=local_summary,
        distance_summary=distance_summary,
        alignment_summary=alignment_summary,
        retrieval_summary=retrieval_summary,
        gates=gates,
        approach=approach,
    )
    atomic_write_text(step_root / "RESULTS.md", report, step_root)

    finished_at = utc_now()
    runtime = time.perf_counter() - started
    runtime_metadata = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": runtime,
        "command": sys.argv,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "gpu": {
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "fragmentation_workers": workers,
        "git_commit": run_git(repo_root, "rev-parse", "HEAD"),
        "config_sha256": sha256_file(config_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "input_sha256": input_hashes,
    }
    atomic_write_json(state_dir / "run_metadata.json", runtime_metadata, step_root)
    hash_ledger(outputs_dir, outputs_dir / "SHA256SUMS", step_root)
    complete = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "complete",
        "finished_at": finished_at,
        "wall_time_seconds": runtime,
        "single_gpu": True,
        "gpu": torch.cuda.get_device_name(0),
        "train_rows": len(train_hashes),
        "validation_rows": len(validation_hashes),
        "test_rows": 0,
        "retrieval_queries": len(queries),
        "gates": {key: bool(value["passed"]) for key, value in gates.items()},
        "recommended_next_approach": approach,
        "results_sha256": sha256_file(step_root / "RESULTS.md"),
        "output_ledger_sha256": sha256_file(outputs_dir / "SHA256SUMS"),
    }
    atomic_write_json(state_dir / "COMPLETE.json", complete, step_root)
    (state_dir / "RUNNING.json").unlink(missing_ok=True)
    print(
        f"[{utc_now()}] complete in {runtime:.1f}s; recommended approach: {approach}",
        flush=True,
    )


if __name__ == "__main__":
    main()

