"""Analysis primitives for the scaled gMolAI latent-space study."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from scaled_common import (
    covariance_sample,
    local_covariance_sample,
    one_cut_related,
    stable_digest,
    tanimoto,
    topk_l2,
    unit_vector,
)

SPACE_ORDER = (
    "graph_256",
    "mean_node_128",
    "hybrid_w1",
    "released_hybrid_w3",
    "hybrid_w6",
)
METHOD_ORDER = (
    "seed_nn",
    "isotropic",
    "global_covariance",
    "local_covariance",
    "mismatched_mmp_direction",
    "mmp_direction",
    "interpolation_0.25",
    "interpolation_0.50",
    "interpolation_0.75",
    "interpolation_1.00",
)
RETRIEVAL_METRICS = (
    "exact_derivative_recall_at_1",
    "exact_derivative_recall_at_10",
    "reciprocal_rank_within_50",
    "scaffold_retention",
    "mmp_consistency",
    "exact_requested_transform",
    "seed_retrieved_tanimoto",
)


def space_matrix(base: np.ndarray, space: str) -> np.ndarray:
    graph = base[:, :256]
    mean_node = base[:, 256:]
    if space == "graph_256":
        return np.ascontiguousarray(graph, dtype=np.float32)
    if space == "mean_node_128":
        return np.ascontiguousarray(mean_node, dtype=np.float32)
    weight = {
        "hybrid_w1": 1.0,
        "released_hybrid_w3": 3.0,
        "hybrid_w6": 6.0,
    }[space]
    result = np.empty((len(base), 384), dtype=np.float32)
    result[:, :256] = graph
    result[:, 256:] = mean_node * weight
    return result


def add_observation_ids(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(
        0,
        "observation_id",
        [
            stable_digest(
                seed,
                "mmp-observation",
                row.core,
                row.transform,
                int(row.lhs_index),
                int(row.rhs_index),
            )
            for row in result.itertuples(index=False)
        ],
    )
    if result["observation_id"].duplicated().any():
        raise RuntimeError("MMP observation identities are not unique")
    return result


def fit_directions(
    train_observations: pd.DataFrame,
    train_space: np.ndarray,
    *,
    minimum_cores: int,
) -> dict[str, dict[str, Any]]:
    directions: dict[str, dict[str, Any]] = {}
    for transform, group in train_observations.groupby("transform", sort=True):
        lhs = group["lhs_index"].to_numpy(dtype=np.int64)
        rhs = group["rhs_index"].to_numpy(dtype=np.int64)
        delta = (
            train_space[rhs].astype(np.float64)
            - train_space[lhs].astype(np.float64)
        )
        norms = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(norms) & (norms > 1e-10)
        if int(valid.sum()) < minimum_cores:
            continue
        delta = delta[valid]
        norms = norms[valid]
        units = delta / norms[:, None]
        resultant = units.mean(axis=0)
        try:
            direction = unit_vector(resultant)
        except ValueError:
            continue
        summed = units.sum(axis=0)
        references = summed[None, :] - units
        reference_norms = np.linalg.norm(references, axis=1)
        valid_references = reference_norms > 1e-10
        loo = np.full(len(units), np.nan, dtype=np.float64)
        if valid_references.any():
            loo[valid_references] = np.sum(
                units[valid_references]
                * (
                    references[valid_references]
                    / reference_norms[valid_references, None]
                ),
                axis=1,
            )
        directions[str(transform)] = {
            "unit": direction.astype(np.float32),
            "median_norm": float(np.median(norms)),
            "train_cores": int(valid.sum()),
            "train_resultant_length": float(np.linalg.norm(resultant)),
            "train_loo_alignment_mean": float(np.nanmean(loo)),
            "train_loo_alignment_median": float(np.nanmedian(loo)),
        }
    return directions


def assign_mismatched_transforms(
    observations: pd.DataFrame,
    common_transforms: set[str],
    support_by_transform: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    result = observations.loc[
        observations["transform"].astype(str).isin(common_transforms)
    ].copy()
    tiers: dict[str, list[str]] = defaultdict(list)
    for transform in sorted(common_transforms):
        support = int(support_by_transform[transform])
        tier = "20+" if support >= 20 else "10-19" if support >= 10 else "5-9" if support >= 5 else "2-4"
        tiers[tier].append(transform)
    all_transforms = sorted(common_transforms)
    assigned: list[str] = []
    for row in result.itertuples(index=False):
        candidates = [
            value
            for value in tiers[str(row.support_tier)]
            if value != str(row.transform)
        ]
        if not candidates:
            candidates = [
                value for value in all_transforms if value != str(row.transform)
            ]
        if not candidates:
            raise RuntimeError("No mismatched transformation control is available")
        position = int(
            stable_digest(seed, "mismatch", row.observation_id)[:16], 16
        ) % len(candidates)
        assigned.append(candidates[position])
    result["mismatched_transform"] = assigned
    return result.reset_index(drop=True)


def evaluate_alignment(
    observations: pd.DataFrame,
    validation_space: np.ndarray,
    directions: dict[str, dict[str, Any]],
    *,
    space: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for transform, group in observations.groupby("transform", sort=True):
        transform = str(transform)
        direction = directions[transform]["unit"].astype(np.float64)
        lhs = group["lhs_index"].to_numpy(dtype=np.int64)
        rhs = group["rhs_index"].to_numpy(dtype=np.int64)
        delta = (
            validation_space[rhs].astype(np.float64)
            - validation_space[lhs].astype(np.float64)
        )
        norms = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(norms) & (norms > 1e-10)
        units = delta[valid] / norms[valid, None]
        valid_group = group.iloc[np.flatnonzero(valid)]
        alignments = units @ direction
        nulls = np.asarray(
            [
                float(
                    unit
                    @ directions[str(row.mismatched_transform)]["unit"].astype(
                        np.float64
                    )
                )
                for unit, row in zip(
                    units, valid_group.itertuples(index=False)
                )
            ],
            dtype=np.float64,
        )
        for row, alignment, null_alignment, norm in zip(
            valid_group.itertuples(index=False),
            alignments,
            nulls,
            norms[valid],
        ):
            rows.append(
                {
                    "space": space,
                    "observation_id": str(row.observation_id),
                    "transform": transform,
                    "core": str(row.core),
                    "train_cores": int(row.train_cores),
                    "support_tier": str(row.support_tier),
                    "mismatched_transform": str(row.mismatched_transform),
                    "alignment": float(alignment),
                    "null_alignment": float(null_alignment),
                    "alignment_gain": float(alignment - null_alignment),
                    "validation_delta_norm": float(norm),
                }
            )
    return pd.DataFrame(rows)


def _round_robin(
    grouped: dict[str, list[Any]],
    transforms: list[str],
    maximum: int,
    used_seeds: set[int],
) -> list[Any]:
    selected: list[Any] = []
    for round_index in range(max((len(grouped[value]) for value in transforms), default=0)):
        for transform in transforms:
            if len(selected) >= maximum:
                return selected
            records = grouped[transform]
            if round_index >= len(records):
                continue
            row = records[round_index]
            seed_index = int(row.lhs_index)
            if seed_index in used_seeds:
                continue
            used_seeds.add(seed_index)
            selected.append(row)
    return selected


def select_queries(
    observations: pd.DataFrame,
    common_transforms: set[str],
    *,
    maximum: int,
    per_transform: int,
    primary_support: int,
    seed: int,
) -> pd.DataFrame:
    eligible = observations.loc[
        observations["transform"].astype(str).isin(common_transforms)
    ].copy()
    grouped: dict[str, list[Any]] = {}
    support: dict[str, int] = {}
    for transform, group in eligible.groupby("transform", sort=True):
        transform = str(transform)
        records = sorted(
            group.itertuples(index=False),
            key=lambda row: stable_digest(seed, "query-order", row.observation_id),
        )[:per_transform]
        if records:
            grouped[transform] = records
            support[transform] = int(group["train_cores"].iloc[0])
    primary = sorted(
        [value for value in grouped if support[value] >= primary_support],
        key=lambda value: (-support[value], stable_digest(seed, value)),
    )
    secondary = sorted(
        [value for value in grouped if support[value] < primary_support],
        key=lambda value: (-support[value], stable_digest(seed, value)),
    )
    used_seeds: set[int] = set()
    selected = _round_robin(grouped, primary, maximum, used_seeds)
    if len(selected) < maximum:
        selected.extend(
            _round_robin(
                grouped,
                secondary,
                maximum - len(selected),
                used_seeds,
            )
        )
    if not selected:
        raise RuntimeError("No unseen-core retrieval queries satisfy the protocol")
    result = pd.DataFrame([row._asdict() for row in selected])
    result.insert(
        0,
        "query_id",
        [
            stable_digest(seed, "retrieval-query", value)
            for value in result["observation_id"]
        ],
    )
    if result["lhs_index"].duplicated().any():
        raise RuntimeError("Retrieval panel contains repeated seed molecules")
    return result.reset_index(drop=True)


def retrieval_experiment(
    *,
    space: str,
    train_space: np.ndarray,
    validation_space: np.ndarray,
    queries: pd.DataFrame,
    directions: dict[str, dict[str, Any]],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    fingerprints: Sequence[Any],
    hashes: Sequence[str],
    smiles: Sequence[str],
    scaffolds: Sequence[str],
    heavy_atoms: np.ndarray,
    validation_core_sets: list[set[str]],
    requested_targets: dict[tuple[str, str], set[int]],
    device: torch.device,
    local_neighbors: int,
    top_k: int,
    summary_top_k: int,
    random_replicates: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    seed_indices = queries["lhs_index"].to_numpy(dtype=np.int64)
    local_indices, _ = topk_l2(
        validation_space[seed_indices],
        train_space,
        k=local_neighbors,
        device=device,
        batch_size=batch_size,
    )
    targets: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    def add_target(
        row: Any,
        method: str,
        replicate: int,
        vector: np.ndarray,
        step: float,
        control_transform: str = "",
    ) -> None:
        targets.append(np.asarray(vector, dtype=np.float32))
        metadata.append(
            {
                "space": space,
                "query_id": str(row.query_id),
                "observation_id": str(row.observation_id),
                "transform": str(row.transform),
                "core": str(row.core),
                "lhs_substituent": str(row.lhs_substituent),
                "rhs_substituent": str(row.rhs_substituent),
                "train_cores": int(row.train_cores),
                "support_tier": str(row.support_tier),
                "seed_index": int(row.lhs_index),
                "true_target_index": int(row.rhs_index),
                "method": method,
                "replicate": int(replicate),
                "step_length": float(step),
                "control_transform": control_transform,
            }
        )

    for position, row in enumerate(queries.itertuples(index=False)):
        seed_vector = validation_space[int(row.lhs_index)].astype(np.float64)
        true_vector = validation_space[int(row.rhs_index)].astype(np.float64)
        true_delta = true_vector - seed_vector
        record = directions[str(row.transform)]
        step = float(record["median_norm"])
        add_target(row, "seed_nn", 0, seed_vector, 0.0)
        add_target(
            row,
            "mmp_direction",
            0,
            seed_vector + step * record["unit"].astype(np.float64),
            step,
        )
        mismatch = str(row.mismatched_transform)
        add_target(
            row,
            "mismatched_mmp_direction",
            0,
            seed_vector
            + step * directions[mismatch]["unit"].astype(np.float64),
            step,
            mismatch,
        )
        for alpha in (0.25, 0.50, 0.75, 1.00):
            add_target(
                row,
                f"interpolation_{alpha:.2f}",
                0,
                seed_vector + alpha * true_delta,
                float(alpha * np.linalg.norm(true_delta)),
            )
        local_displacements = (
            train_space[local_indices[position]].astype(np.float64)
            - seed_vector
        )
        for replicate in range(random_replicates):
            rng = np.random.default_rng(
                int(
                    stable_digest(
                        seed, "perturbation", row.query_id, replicate
                    )[:16],
                    16,
                )
            )
            samples = {
                "isotropic": rng.normal(size=train_space.shape[1]),
                "global_covariance": covariance_sample(
                    eigenvalues, eigenvectors, rng
                ),
                "local_covariance": local_covariance_sample(
                    local_displacements, rng
                ),
            }
            for method, sample in samples.items():
                try:
                    perturbation = step * unit_vector(sample)
                except ValueError:
                    perturbation = step * unit_vector(
                        rng.normal(size=train_space.shape[1])
                    )
                add_target(
                    row,
                    method,
                    replicate,
                    seed_vector + perturbation,
                    step,
                )

    target_matrix = np.stack(targets).astype(np.float32)
    excluded = np.asarray(
        [record["seed_index"] for record in metadata], dtype=np.int64
    )
    candidate_indices, candidate_distances = topk_l2(
        target_matrix,
        validation_space,
        k=top_k,
        device=device,
        batch_size=batch_size,
        exclude_indices=excluded,
    )
    rows: list[dict[str, Any]] = []
    for target_position, record in enumerate(metadata):
        candidates = candidate_indices[target_position]
        seed_index = int(record["seed_index"])
        true_index = int(record["true_target_index"])
        hits = np.flatnonzero(candidates == true_index)
        rank = int(hits[0] + 1) if len(hits) else 0
        top1 = int(candidates[0])
        requested = requested_targets.get(
            (record["core"], record["rhs_substituent"]), set()
        )
        top_summary = candidates[:summary_top_k]
        row = dict(record)
        row.update(
            {
                "target_rank_within_50": rank,
                "exact_derivative_recall_at_1": float(rank == 1),
                "exact_derivative_recall_at_10": float(0 < rank <= 10),
                "reciprocal_rank_within_50": float(1.0 / rank if rank else 0.0),
                "scaffold_retention": float(
                    bool(scaffolds[seed_index])
                    and scaffolds[seed_index] == scaffolds[top1]
                ),
                "mmp_consistency": float(
                    one_cut_related(validation_core_sets, seed_index, top1)
                ),
                "exact_requested_transform": float(top1 in requested),
                "seed_retrieved_tanimoto": tanimoto(
                    fingerprints[seed_index], fingerprints[top1]
                ),
                "top10_mmp_consistency": float(
                    np.mean(
                        [
                            one_cut_related(
                                validation_core_sets, seed_index, int(value)
                            )
                            for value in top_summary
                        ]
                    )
                ),
                "top10_scaffold_retention": float(
                    np.mean(
                        [
                            bool(scaffolds[seed_index])
                            and scaffolds[seed_index] == scaffolds[int(value)]
                            for value in top_summary
                        ]
                    )
                ),
                "top1_distance_to_constructed_target": float(
                    candidate_distances[target_position, 0]
                ),
                "seed_hash": str(hashes[seed_index]),
                "true_target_hash": str(hashes[true_index]),
                "top1_candidate_index": top1,
                "top1_candidate_hash": str(hashes[top1]),
                "seed_smiles": str(smiles[seed_index]),
                "true_target_smiles": str(smiles[true_index]),
                "top1_candidate_smiles": str(smiles[top1]),
                "seed_heavy_atoms": int(heavy_atoms[seed_index]),
                "top1_heavy_atoms": int(heavy_atoms[top1]),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    expected_methods = set(METHOD_ORDER)
    observed_methods = set(result["method"])
    if observed_methods != expected_methods:
        raise RuntimeError(
            f"Retrieval method mismatch for {space}: {sorted(observed_methods)}"
        )
    return result


def average_replicates(retrieval: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "space",
        "query_id",
        "observation_id",
        "transform",
        "core",
        "train_cores",
        "support_tier",
        "seed_index",
        "true_target_index",
        "method",
        "seed_hash",
        "true_target_hash",
    ]
    metrics = list(RETRIEVAL_METRICS) + [
        "top10_mmp_consistency",
        "top10_scaffold_retention",
    ]
    return (
        retrieval.groupby(identity, sort=False, dropna=False)[metrics]
        .mean()
        .reset_index()
    )


def by_transformation(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    analysis: str,
    method: str = "",
) -> pd.DataFrame:
    selected = frame if not method else frame.loc[frame["method"] == method]
    keys = ["space", "transform", "train_cores", "support_tier"]
    result = selected.groupby(keys, sort=True)[list(metrics)].agg(
        ["mean", "count"]
    )
    result.columns = [
        f"{metric}_{statistic}" for metric, statistic in result.columns
    ]
    result = result.reset_index()
    result.insert(0, "analysis", analysis)
    if method:
        result.insert(2, "method", method)
    return result


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    analysis: str,
    cohorts: Sequence[int],
    resamples: int,
    alpha: float,
    seed: int,
    reference: str = "released_hybrid_w3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["transform", "observation_id", "train_cores"]
    duplicates = frame.duplicated(keys + ["space"])
    if duplicates.any():
        raise RuntimeError(
            f"{analysis} contains duplicate space/observation rows before bootstrap"
        )
    values = frame.pivot(
        index=keys, columns="space", values=list(metrics)
    ).sort_index()
    for metric in metrics:
        if set(values[metric].columns) != set(SPACE_ORDER):
            raise RuntimeError(f"{analysis}/{metric} is not paired across spaces")
    values = values.reindex(
        columns=pd.MultiIndex.from_product([list(metrics), list(SPACE_ORDER)])
    )
    index_frame = values.index.to_frame(index=False)
    array = values.to_numpy(dtype=np.float64).reshape(
        len(values), len(metrics), len(SPACE_ORDER)
    )
    estimate_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for threshold in cohorts:
        mask = index_frame["train_cores"].to_numpy(dtype=np.int64) >= int(
            threshold
        )
        cohort_index = index_frame.loc[mask].reset_index(drop=True)
        cohort_values = array[mask]
        transformations = sorted(cohort_index["transform"].astype(str).unique())
        groups = [
            np.flatnonzero(
                cohort_index["transform"].astype(str).to_numpy() == transform
            )
            for transform in transformations
        ]
        if not transformations:
            continue
        per_transform = np.stack(
            [np.nanmean(cohort_values[group], axis=0) for group in groups]
        )
        point = np.nanmean(per_transform, axis=0)
        positive_fraction = np.mean(
            per_transform[:, list(metrics).index("alignment_gain"), :] > 0,
            axis=0,
        ) if "alignment_gain" in metrics else None
        rng = np.random.default_rng(
            seed + int(stable_digest(analysis, threshold)[:8], 16)
        )
        boot = np.empty(
            (resamples, len(metrics), len(SPACE_ORDER)), dtype=np.float64
        )
        for replicate in range(resamples):
            sampled_transformations = rng.integers(
                0, len(transformations), size=len(transformations)
            )
            sampled_means = []
            for transform_position in sampled_transformations:
                group = groups[int(transform_position)]
                sampled_rows = group[
                    rng.integers(0, len(group), size=len(group))
                ]
                sampled_means.append(
                    np.nanmean(cohort_values[sampled_rows], axis=0)
                )
            boot[replicate] = np.nanmean(
                np.stack(sampled_means), axis=0
            )
        low = np.nanquantile(boot, alpha / 2.0, axis=0)
        high = np.nanquantile(boot, 1.0 - alpha / 2.0, axis=0)
        reference_position = SPACE_ORDER.index(reference)
        for metric_position, metric in enumerate(metrics):
            for space_position, space in enumerate(SPACE_ORDER):
                record = {
                    "analysis": analysis,
                    "minimum_train_cores": int(threshold),
                    "transformations": len(transformations),
                    "observations": len(cohort_index),
                    "metric": metric,
                    "space": space,
                    "macro_estimate": float(point[metric_position, space_position]),
                    "ci_low": float(low[metric_position, space_position]),
                    "ci_high": float(high[metric_position, space_position]),
                }
                if positive_fraction is not None and metric == "alignment_gain":
                    record["positive_transformation_fraction"] = float(
                        positive_fraction[space_position]
                    )
                estimate_rows.append(record)
                if space == reference:
                    continue
                differences = (
                    boot[:, metric_position, space_position]
                    - boot[:, metric_position, reference_position]
                )
                difference_rows.append(
                    {
                        "analysis": analysis,
                        "minimum_train_cores": int(threshold),
                        "transformations": len(transformations),
                        "observations": len(cohort_index),
                        "metric": metric,
                        "space": space,
                        "reference_space": reference,
                        "paired_macro_difference": float(
                            point[metric_position, space_position]
                            - point[metric_position, reference_position]
                        ),
                        "ci_low": float(
                            np.nanquantile(differences, alpha / 2.0)
                        ),
                        "ci_high": float(
                            np.nanquantile(differences, 1.0 - alpha / 2.0)
                        ),
                    }
                )
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows)


def retrieval_summary(retrieval_average: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (space, method), group in retrieval_average.groupby(
        ["space", "method"], sort=True
    ):
        transform_means = group.groupby("transform", sort=True)[
            list(RETRIEVAL_METRICS)
        ].mean()
        row: dict[str, Any] = {
            "space": str(space),
            "method": str(method),
            "queries": int(group["query_id"].nunique()),
            "transformations": int(group["transform"].nunique()),
        }
        for metric in RETRIEVAL_METRICS:
            row[f"{metric}_pair_mean"] = float(group[metric].mean())
            row[f"{metric}_macro_transform_mean"] = float(
                transform_means[metric].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)
