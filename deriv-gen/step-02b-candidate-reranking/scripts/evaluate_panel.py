#!/usr/bin/env python3
"""Generate, filter, re-encode, latent-rerank, and score one Step-2b panel."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs

from candidate_model import generate_beam_pool, generate_sample_pool
from common import (
    STEP_ROOT,
    atomic_numpy_savez,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_csv,
    configure_determinism,
    cosine_rows,
    decode_tokens,
    fingerprint,
    fingerprint_generator,
    load_decoder,
    load_json,
    load_released_inference,
    policy_canonicalize,
    protocol,
    released_train_rows,
    require_one_gpu,
    sha256_file,
    stable_digest,
    utc_now,
    validate_manifest,
    validation_embeddings,
)
from gmolai_retrain.fast_inference import (
    build_smiles_encoder,
    implementation_metadata,
)


CONTROL_SOURCE_COLUMNS = {
    "correct_embedding": "correct_source_index",
    "shuffled_embedding": "shuffled_source_index",
    "nearest_wrong_embedding": "nearest_wrong_source_index",
}


def decode_and_filter(
    tokens: np.ndarray,
    resolved: dict[str, Any],
    *,
    cumulative_score: float,
    length: int,
    generator_kind: str,
) -> dict[str, Any]:
    raw, token_error = decode_tokens(tokens)
    canonical, rejection, rdkit_valid = policy_canonicalize(raw, resolved)
    record: dict[str, Any] = {
        "raw_smiles": raw,
        "token_error": token_error,
        "rdkit_valid": bool(rdkit_valid),
        "policy_accepted": canonical is not None,
        "policy_rejection": rejection,
        "cumulative_decoder_log_probability": float(cumulative_score),
        "generated_length": int(length),
        "generator_kind": generator_kind,
        "canonical_smiles": "",
        "molecule_hash": "",
        "scaffold": "",
        "atom_count": 0,
    }
    if canonical is not None:
        record.update(
            {
                "canonical_smiles": str(canonical.smiles),
                "molecule_hash": str(canonical.molecule_hash),
                "scaffold": str(canonical.scaffold),
                "atom_count": int(canonical.atom_count),
            }
        )
    return record


def greedy_records(
    model: Any,
    conditions: np.ndarray,
    resolved: dict[str, Any],
    *,
    batch_size: int,
    maximum_steps: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for offset in range(0, len(conditions), batch_size):
        stop = min(offset + batch_size, len(conditions))
        tensor = torch.as_tensor(
            conditions[offset:stop], dtype=torch.float32, device=device
        )
        generated = model.generate(tensor, maximum_steps=maximum_steps).cpu().numpy()
        for row in generated:
            result.append(
                decode_and_filter(
                    row,
                    resolved,
                    cumulative_score=math.nan,
                    length=maximum_steps,
                    generator_kind="greedy",
                )
            )
        print(f"    greedy {stop:,}/{len(conditions):,}", flush=True)
    return result


def beam_records(
    model: Any,
    conditions: np.ndarray,
    resolved: dict[str, Any],
    *,
    batch_size: int,
    maximum_steps: int,
    beam_width: int,
    device: torch.device,
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for offset in range(0, len(conditions), batch_size):
        stop = min(offset + batch_size, len(conditions))
        tensor = torch.as_tensor(
            conditions[offset:stop], dtype=torch.float32, device=device
        )
        generated, scores, lengths = generate_beam_pool(
            model,
            tensor,
            maximum_steps=maximum_steps,
            beam_width=beam_width,
        )
        generated_np = generated.cpu().numpy()
        scores_np = scores.cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        for local in range(stop - offset):
            current: list[dict[str, Any]] = []
            for beam in range(beam_width):
                current.append(
                    decode_and_filter(
                        generated_np[local, beam],
                        resolved,
                        cumulative_score=float(scores_np[local, beam]),
                        length=int(lengths_np[local, beam]),
                        generator_kind="beam",
                    )
                )
            result.append(current)
        print(f"    beam {stop:,}/{len(conditions):,}", flush=True)
    return result


def sample_records(
    model: Any,
    conditions: np.ndarray,
    resolved: dict[str, Any],
    *,
    policy: dict[str, Any],
    phase: str,
    control: str,
    global_seed: int,
    batch_size: int,
    maximum_steps: int,
    draws: int,
    device: torch.device,
) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for offset in range(0, len(conditions), batch_size):
        stop = min(offset + batch_size, len(conditions))
        tensor = torch.as_tensor(
            conditions[offset:stop], dtype=torch.float32, device=device
        )
        batch_seed = int(
            stable_digest(
                global_seed, phase, control, policy["name"], offset, stop
            )[:16],
            16,
        )
        generated, scores, lengths = generate_sample_pool(
            model,
            tensor,
            maximum_steps=maximum_steps,
            draws=draws,
            temperature=float(policy["temperature"]),
            top_p=float(policy["top_p"]),
            seed=batch_seed,
        )
        generated_np = generated.cpu().numpy()
        scores_np = scores.cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        for local in range(stop - offset):
            current: list[dict[str, Any]] = []
            for draw in range(draws):
                current.append(
                    decode_and_filter(
                        generated_np[local, draw],
                        resolved,
                        cumulative_score=float(scores_np[local, draw]),
                        length=int(lengths_np[local, draw]),
                        generator_kind="sample",
                    )
                )
            result.append(current)
        print(
            f"    {policy['name']} {stop:,}/{len(conditions):,}", flush=True
        )
    return result


def ordered_non_greedy(
    records: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    accepted = [record for record in records if record["policy_accepted"]]
    if policy["kind"] == "beam":
        penalty = float(policy["length_penalty"])
        for record in accepted:
            normalizer = ((5.0 + float(record["generated_length"])) / 6.0) ** penalty
            record["policy_order_score"] = (
                float(record["cumulative_decoder_log_probability"]) / normalizer
            )
    else:
        for record in accepted:
            record["policy_order_score"] = float(
                record["cumulative_decoder_log_probability"]
            ) / max(int(record["generated_length"]), 1)
    return sorted(
        accepted,
        key=lambda record: (
            -float(record["policy_order_score"]),
            str(record["canonical_smiles"]),
            str(record["raw_smiles"]),
        ),
    )


def assemble_candidates(
    *,
    phase: str,
    policy: dict[str, Any],
    control: str,
    query_position: int,
    target_index: int,
    source_index: int,
    greedy: dict[str, Any],
    raw_non_greedy: list[dict[str, Any]],
    maximum_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(record: dict[str, Any], rank: int) -> None:
        rows.append(
            {
                "phase": phase,
                "policy": policy["name"],
                "control": control,
                "query_position": int(query_position),
                "target_index": int(target_index),
                "condition_source_index": int(source_index),
                "proposal_rank": int(rank),
                "generator_kind": record["generator_kind"],
                "decoder_order_score": float(record.get("policy_order_score", math.nan)),
                "cumulative_decoder_log_probability": float(
                    record["cumulative_decoder_log_probability"]
                ),
                "generated_length": int(record["generated_length"]),
                "raw_smiles": str(record["raw_smiles"]),
                "canonical_smiles": str(record["canonical_smiles"]),
                "candidate_hash": str(record["molecule_hash"]),
                "candidate_scaffold": str(record["scaffold"]),
                "candidate_atom_count": int(record["atom_count"]),
            }
        )

    if greedy["policy_accepted"]:
        seen.add(str(greedy["molecule_hash"]))
        append(greedy, 1)
    next_rank = 2
    for record in ordered_non_greedy(raw_non_greedy, policy):
        identity = str(record["molecule_hash"])
        if identity in seen:
            continue
        seen.add(identity)
        append(record, next_rank)
        next_rank += 1
        if next_rank > maximum_candidates:
            break
    raw_count = len(raw_non_greedy)
    accepted_non_greedy_identities = {
        str(record["molecule_hash"])
        for record in raw_non_greedy
        if record["policy_accepted"]
    }
    if greedy["policy_accepted"]:
        accepted_non_greedy_identities.discard(str(greedy["molecule_hash"]))
    stats = {
        "phase": phase,
        "policy": policy["name"],
        "control": control,
        "query_position": int(query_position),
        "target_index": int(target_index),
        "condition_source_index": int(source_index),
        "greedy_token_decoded": float(not bool(greedy["token_error"])),
        "greedy_rdkit_valid": float(greedy["rdkit_valid"]),
        "greedy_policy_accepted": float(greedy["policy_accepted"]),
        "raw_non_greedy_draws": raw_count,
        "raw_token_decoded": sum(not bool(record["token_error"]) for record in raw_non_greedy),
        "raw_rdkit_valid": sum(bool(record["rdkit_valid"]) for record in raw_non_greedy),
        "raw_policy_accepted": sum(bool(record["policy_accepted"]) for record in raw_non_greedy),
        "unique_policy_accepted_non_greedy": len(accepted_non_greedy_identities),
        "retained_unique_candidate_count": len(rows),
    }
    return rows, stats


def load_phase(
    phase: str,
    root: Path,
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    if phase == "development":
        panel = pd.read_csv(root / "prepared" / "development_panel.csv")
        molecules = pd.read_parquet(paths["train_molecules"])
        all_source_indices = np.unique(
            np.concatenate(
                [panel[column].to_numpy(dtype=np.int64) for column in CONTROL_SOURCE_COLUMNS.values()]
            )
        )
        source_conditions = released_train_rows(
            paths["train_raw_embeddings"],
            paths["gmolai_calibrator"],
            all_source_indices,
        )
        lookup = {int(index): position for position, index in enumerate(all_source_indices)}
        controls = {
            control: source_conditions[
                np.asarray([lookup[int(value)] for value in panel[column]], dtype=np.int64)
            ]
            for control, column in CONTROL_SOURCE_COLUMNS.items()
        }
    elif phase == "final":
        panel = pd.read_csv(root / "prepared" / "fresh_validation_panel.csv")
        molecules = pd.read_parquet(paths["validation_molecules"])
        payload = torch.load(
            paths["validation_embeddings"], map_location="cpu", weights_only=False
        )
        matrix = validation_embeddings(payload)
        hashes = molecules["molecule_hash"].astype(str).tolist()
        if hashes != [str(value) for value in payload["molecule_hashes"]]:
            raise RuntimeError("Validation conditions and molecules are misaligned")
        controls = {
            control: matrix[panel[column].to_numpy(dtype=np.int64)]
            for control, column in CONTROL_SOURCE_COLUMNS.items()
        }
    else:
        raise ValueError(phase)
    if panel["query_position"].tolist() != list(range(len(panel))):
        raise RuntimeError(f"{phase} panel query positions are not contiguous")
    return panel, molecules, controls


def reencode_candidates(
    frame: pd.DataFrame,
    *,
    repo_root: Path,
    root: Path,
    cfg: dict[str, Any],
    phase: str,
    device: torch.device,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    grouped = (
        frame[["canonical_smiles", "candidate_atom_count"]]
        .drop_duplicates("canonical_smiles")
        .sort_values("canonical_smiles")
    )
    smiles = grouped["canonical_smiles"].astype(str).tolist()
    atom_counts = grouped["candidate_atom_count"].astype(int).tolist()
    inference = load_released_inference(repo_root)
    bundle = inference.load_model_bundle(repo_root / "inference" / "model", device)
    encoder = build_smiles_encoder(
        "optimized",
        bundle.model,
        bundle.coordinate_mean,
        bundle.coordinate_scale,
        device=device,
        batch_size=int(cfg["reranking"]["reencode_batch_size"]),
        node_budget=int(cfg["reranking"]["reencode_node_budget"]),
        workers=cfg["reranking"]["reencode_workers"],
        mean_node_weight=bundle.mean_node_weight,
    )
    try:
        embeddings = encoder.encode(smiles, atom_counts=atom_counts).astype(np.float32)
        metadata = implementation_metadata(encoder)
    finally:
        encoder.close()
    del encoder, bundle
    if embeddings.shape != (len(smiles), 384) or not np.isfinite(embeddings).all():
        raise RuntimeError("Frozen re-encoder returned an invalid candidate matrix")
    path = root / "outputs" / "raw" / f"{phase}_reencoded_unique_molecules.npz"
    atomic_numpy_savez(
        path,
        root,
        canonical_smiles=np.asarray(smiles, dtype=np.str_),
        embeddings=embeddings,
    )
    return embeddings, smiles, metadata


def annotate_candidates(
    frame: pd.DataFrame,
    *,
    embeddings: np.ndarray,
    unique_smiles: list[str],
    panel: pd.DataFrame,
    molecules: pd.DataFrame,
    controls: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    embedding_lookup = {smiles: index for index, smiles in enumerate(unique_smiles)}
    frame["candidate_embedding_index"] = frame["canonical_smiles"].map(
        embedding_lookup
    ).astype(np.int64)
    l2 = np.empty(len(frame), dtype=np.float32)
    relative_l2 = np.empty(len(frame), dtype=np.float32)
    cosine = np.empty(len(frame), dtype=np.float32)
    for control, supplied in controls.items():
        rows = np.flatnonzero(frame["control"].to_numpy() == control)
        candidate_values = embeddings[
            frame.iloc[rows]["candidate_embedding_index"].to_numpy(dtype=np.int64)
        ]
        query_positions = frame.iloc[rows]["query_position"].to_numpy(dtype=np.int64)
        supplied_values = supplied[query_positions]
        delta = candidate_values - supplied_values
        current_l2 = np.linalg.norm(delta.astype(np.float64), axis=1)
        l2[rows] = current_l2.astype(np.float32)
        relative_l2[rows] = (
            current_l2
            / np.maximum(np.linalg.norm(supplied_values.astype(np.float64), axis=1), 1e-12)
        ).astype(np.float32)
        cosine[rows] = cosine_rows(candidate_values, supplied_values).astype(np.float32)
    frame["latent_cosine_to_supplied_condition"] = cosine
    frame["latent_l2_to_supplied_condition"] = l2
    frame["latent_relative_l2_to_supplied_condition"] = relative_l2

    target_indices = panel["target_index"].to_numpy(dtype=np.int64)
    target_hashes = molecules.iloc[target_indices]["molecule_hash"].astype(str).to_numpy()
    target_scaffolds = (
        molecules.iloc[target_indices]["scaffold"].fillna("").astype(str).to_numpy()
    )
    target_smiles = molecules.iloc[target_indices]["canonical_smiles"].astype(str).to_numpy()
    frame_queries = frame["query_position"].to_numpy(dtype=np.int64)
    frame_targets = target_hashes[frame_queries]
    frame_target_scaffolds = target_scaffolds[frame_queries]
    frame["exact_target_identity"] = (
        frame["candidate_hash"].astype(str).to_numpy() == frame_targets
    ).astype(np.float32)
    frame["target_scaffold_recovery"] = (
        frame["candidate_scaffold"].astype(str).to_numpy() == frame_target_scaffolds
    ).astype(np.float32)

    source_hash_by_control: dict[str, np.ndarray] = {}
    source_scaffold_by_control: dict[str, np.ndarray] = {}
    source_smiles_by_control: dict[str, np.ndarray] = {}
    for control, source_column in CONTROL_SOURCE_COLUMNS.items():
        source_indices = panel[source_column].to_numpy(dtype=np.int64)
        source_hash_by_control[control] = (
            molecules.iloc[source_indices]["molecule_hash"].astype(str).to_numpy()
        )
        source_scaffold_by_control[control] = (
            molecules.iloc[source_indices]["scaffold"].fillna("").astype(str).to_numpy()
        )
        source_smiles_by_control[control] = (
            molecules.iloc[source_indices]["canonical_smiles"].astype(str).to_numpy()
        )
    exact_source = np.zeros(len(frame), dtype=np.float32)
    scaffold_source = np.zeros(len(frame), dtype=np.float32)
    for control in controls:
        rows = np.flatnonzero(frame["control"].to_numpy() == control)
        queries = frame.iloc[rows]["query_position"].to_numpy(dtype=np.int64)
        exact_source[rows] = (
            frame.iloc[rows]["candidate_hash"].astype(str).to_numpy()
            == source_hash_by_control[control][queries]
        ).astype(np.float32)
        scaffold_source[rows] = (
            frame.iloc[rows]["candidate_scaffold"].astype(str).to_numpy()
            == source_scaffold_by_control[control][queries]
        ).astype(np.float32)
    frame["exact_condition_source_identity"] = exact_source
    frame["condition_source_scaffold_recovery"] = scaffold_source

    generator = fingerprint_generator(
        int(cfg["evaluation"]["morgan_radius"]),
        int(cfg["evaluation"]["morgan_bits"]),
    )
    candidate_fingerprints = [fingerprint(smiles, generator) for smiles in unique_smiles]
    target_fingerprints = [fingerprint(smiles, generator) for smiles in target_smiles]
    source_fingerprints = {
        control: [fingerprint(smiles, generator) for smiles in values]
        for control, values in source_smiles_by_control.items()
    }
    morgan_target = np.empty(len(frame), dtype=np.float32)
    morgan_source = np.empty(len(frame), dtype=np.float32)
    for control in controls:
        rows = np.flatnonzero(frame["control"].to_numpy() == control)
        queries = frame.iloc[rows]["query_position"].to_numpy(dtype=np.int64)
        embedding_indices = frame.iloc[rows]["candidate_embedding_index"].to_numpy(
            dtype=np.int64
        )
        for local, (row, query, embedding_index) in enumerate(
            zip(rows, queries, embedding_indices)
        ):
            candidate_fp = candidate_fingerprints[int(embedding_index)]
            morgan_target[row] = float(
                DataStructs.TanimotoSimilarity(target_fingerprints[int(query)], candidate_fp)
            )
            morgan_source[row] = float(
                DataStructs.TanimotoSimilarity(
                    source_fingerprints[control][int(query)], candidate_fp
                )
            )
    frame["morgan_similarity_to_target"] = morgan_target
    frame["morgan_similarity_to_condition_source"] = morgan_source
    return frame


def summarize(
    candidates: pd.DataFrame,
    generation_stats: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    policies: list[dict[str, Any]],
    controls: dict[str, np.ndarray],
    candidate_sizes: list[int],
    historical_identity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_template = pd.DataFrame(
        {"query_position": np.arange(len(panel), dtype=np.int64)}
    )
    selection_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for policy in policies:
        for control in controls:
            base = candidates.loc[
                (candidates["policy"] == policy["name"])
                & (candidates["control"] == control)
            ].copy()
            stat = generation_stats.loc[
                (generation_stats["policy"] == policy["name"])
                & (generation_stats["control"] == control)
            ]
            raw_draws = float(stat["raw_non_greedy_draws"].sum())
            raw_valid_rate = float(stat["raw_rdkit_valid"].sum() / max(raw_draws, 1.0))
            raw_accept_rate = float(
                stat["raw_policy_accepted"].sum() / max(raw_draws, 1.0)
            )
            unique_rate = float(
                stat["unique_policy_accepted_non_greedy"].sum()
                / max(raw_draws, 1.0)
            )
            for size in candidate_sizes:
                subset = base.loc[base["proposal_rank"] <= int(size)].copy()
                aggregates = subset.groupby("query_position", sort=False).agg(
                    candidate_count=("candidate_hash", "size"),
                    oracle_target_recall=("exact_target_identity", "max"),
                    oracle_condition_source_recall=(
                        "exact_condition_source_identity",
                        "max",
                    ),
                )
                ordered = subset.sort_values(
                    [
                        "query_position",
                        "latent_relative_l2_to_supplied_condition",
                        "latent_cosine_to_supplied_condition",
                        "proposal_rank",
                        "canonical_smiles",
                    ],
                    ascending=[True, True, False, True, True],
                    kind="mergesort",
                )
                selected = ordered.drop_duplicates("query_position", keep="first")
                selected = selected[
                    [
                        "query_position",
                        "proposal_rank",
                        "generator_kind",
                        "raw_smiles",
                        "canonical_smiles",
                        "candidate_hash",
                        "candidate_scaffold",
                        "exact_target_identity",
                        "exact_condition_source_identity",
                        "target_scaffold_recovery",
                        "condition_source_scaffold_recovery",
                        "morgan_similarity_to_target",
                        "morgan_similarity_to_condition_source",
                        "latent_cosine_to_supplied_condition",
                        "latent_l2_to_supplied_condition",
                        "latent_relative_l2_to_supplied_condition",
                    ]
                ].rename(columns=lambda name: f"reranked_{name}" if name != "query_position" else name)
                query = query_template.merge(
                    aggregates, how="left", left_on="query_position", right_index=True
                ).merge(selected, how="left", on="query_position")
                query["candidate_count"] = query["candidate_count"].fillna(0).astype(int)
                query["oracle_target_recall"] = query["oracle_target_recall"].fillna(0.0)
                query["oracle_condition_source_recall"] = query[
                    "oracle_condition_source_recall"
                ].fillna(0.0)
                query["reranked_valid_top1"] = query[
                    "reranked_candidate_hash"
                ].notna().astype(np.float32)
                zero_fill = [
                    "reranked_exact_target_identity",
                    "reranked_exact_condition_source_identity",
                    "reranked_target_scaffold_recovery",
                    "reranked_condition_source_scaffold_recovery",
                    "reranked_morgan_similarity_to_target",
                    "reranked_morgan_similarity_to_condition_source",
                ]
                for column in zero_fill:
                    query[column] = query[column].fillna(0.0)
                query["phase"] = str(base["phase"].iloc[0]) if len(base) else ""
                query["policy"] = policy["name"]
                query["control"] = control
                query["candidate_set_size"] = int(size)
                selection_frames.append(query)

                oracle = float(query["oracle_target_recall"].mean())
                reranked_exact = float(query["reranked_exact_target_identity"].mean())
                record = {
                    "phase": query["phase"].iloc[0],
                    "policy": policy["name"],
                    "control": control,
                    "candidate_set_size": int(size),
                    "rows": len(query),
                    "candidate_set_nonempty_rate": float(
                        (query["candidate_count"] > 0).mean()
                    ),
                    "candidate_set_full_rate": float(
                        (query["candidate_count"] >= int(size)).mean()
                    ),
                    "mean_unique_valid_candidate_count": float(
                        query["candidate_count"].mean()
                    ),
                    "raw_non_greedy_rdkit_valid_rate": raw_valid_rate,
                    "raw_non_greedy_policy_accept_rate": raw_accept_rate,
                    "unique_candidate_rate_per_raw_proposal": unique_rate,
                    "oracle_target_recall_at_k": oracle,
                    "oracle_condition_source_recall_at_k": float(
                        query["oracle_condition_source_recall"].mean()
                    ),
                    "latent_reranked_valid_top1_rate": float(
                        query["reranked_valid_top1"].mean()
                    ),
                    "latent_reranked_exact_target_identity_at_1": reranked_exact,
                    "latent_reranked_exact_condition_source_identity_at_1": float(
                        query["reranked_exact_condition_source_identity"].mean()
                    ),
                    "latent_reranked_target_scaffold_recovery": float(
                        query["reranked_target_scaffold_recovery"].mean()
                    ),
                    "latent_reranked_condition_source_scaffold_recovery": float(
                        query["reranked_condition_source_scaffold_recovery"].mean()
                    ),
                    "latent_reranked_mean_morgan_to_target": float(
                        query["reranked_morgan_similarity_to_target"].mean()
                    ),
                    "latent_reranked_mean_morgan_to_condition_source": float(
                        query["reranked_morgan_similarity_to_condition_source"].mean()
                    ),
                    "latent_reranked_mean_cosine_to_supplied_condition": float(
                        query["reranked_latent_cosine_to_supplied_condition"].mean()
                    ),
                    "latent_reranked_median_cosine_to_supplied_condition": float(
                        query["reranked_latent_cosine_to_supplied_condition"].median()
                    ),
                    "latent_reranked_mean_l2_to_supplied_condition": float(
                        query["reranked_latent_l2_to_supplied_condition"].mean()
                    ),
                    "latent_reranked_median_l2_to_supplied_condition": float(
                        query["reranked_latent_l2_to_supplied_condition"].median()
                    ),
                    "latent_reranked_mean_relative_l2_to_supplied_condition": float(
                        query[
                            "reranked_latent_relative_l2_to_supplied_condition"
                        ].mean()
                    ),
                    "latent_reranked_median_relative_l2_to_supplied_condition": float(
                        query[
                            "reranked_latent_relative_l2_to_supplied_condition"
                        ].median()
                    ),
                    "rerank_target_selection_efficiency_given_oracle_presence": (
                        reranked_exact / oracle if oracle > 0.0 else math.nan
                    ),
                    "historical_step2_exact_identity": float(historical_identity),
                    "exact_target_identity_gain_over_historical_step2": (
                        reranked_exact - float(historical_identity)
                    ),
                }
                summary_rows.append(record)
    selections = pd.concat(selection_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    greedy = summary.loc[summary["candidate_set_size"] == 1, [
        "policy",
        "control",
        "latent_reranked_exact_target_identity_at_1",
        "latent_reranked_exact_condition_source_identity_at_1",
    ]].drop_duplicates(["policy", "control"])
    greedy = greedy.rename(
        columns={
            "latent_reranked_exact_target_identity_at_1": "same_panel_greedy_target_identity",
            "latent_reranked_exact_condition_source_identity_at_1": "same_panel_greedy_condition_source_identity",
        }
    )
    summary = summary.merge(greedy, how="left", on=["policy", "control"])
    summary["exact_target_identity_gain_over_same_panel_greedy"] = (
        summary["latent_reranked_exact_target_identity_at_1"]
        - summary["same_panel_greedy_target_identity"]
    )
    summary["condition_source_identity_gain_over_same_panel_greedy"] = (
        summary["latent_reranked_exact_condition_source_identity_at_1"]
        - summary["same_panel_greedy_condition_source_identity"]
    )
    return selections, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    phase = args.phase
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    state_path = root / "state" / (
        "DEVELOPMENT_COMPLETE.json" if phase == "development" else "FINAL_COMPLETE.json"
    )
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    if not (root / "state" / "PANELS_PREPARED.json").is_file():
        raise RuntimeError("Panels are not frozen")
    if phase == "final":
        policy_seal_path = root / "state" / "POLICY_FROZEN.json"
        if not policy_seal_path.is_file():
            raise RuntimeError("Freeze a development-selected policy before final generation")
        policy_seal = load_json(policy_seal_path)
        if not policy_seal.get("selected_before_final_generation"):
            raise RuntimeError("Candidate policy was not frozen prospectively")
    cfg = protocol(root)
    paths, input_hashes = validate_manifest(repo_root, root)
    device = require_one_gpu()
    seed = int(cfg["seed"])
    configure_determinism(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    started_at = utc_now()
    panel, molecules, controls = load_phase(phase, root, paths)
    expected_rows = int(
        cfg["panels"]["development_rows" if phase == "development" else "final_rows"]
    )
    if len(panel) != expected_rows:
        raise RuntimeError(f"Unexpected {phase} panel size")
    all_policies = list(cfg["generation"]["policies"])
    if phase == "development":
        policies = all_policies
    else:
        selected_name = str(policy_seal["selected_policy"])
        policies = [policy for policy in all_policies if policy["name"] == selected_name]
        if len(policies) != 1:
            raise RuntimeError("Frozen policy is absent from preregistration")
    resolved = load_json(paths["gmolai_resolved_config"])
    model, checkpoint = load_decoder(paths["decoder_checkpoint"], device)
    maximum_steps = int(cfg["generation"]["maximum_smiles_bytes"])
    maximum_candidates = int(cfg["generation"]["maximum_unique_candidates"])
    candidate_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for control, condition_matrix in controls.items():
        print(f"{phase}: generating {control}", flush=True)
        greedy = greedy_records(
            model,
            condition_matrix,
            resolved,
            batch_size=int(cfg["generation"]["greedy_query_batch_size"]),
            maximum_steps=maximum_steps,
            device=device,
        )
        beam_pool: list[list[dict[str, Any]]] | None = None
        if any(policy["kind"] == "beam" for policy in policies):
            beam_pool = beam_records(
                model,
                condition_matrix,
                resolved,
                batch_size=int(cfg["generation"]["beam_query_batch_size"]),
                maximum_steps=maximum_steps,
                beam_width=int(cfg["generation"]["beam_width"]),
                device=device,
            )
        sample_pools: dict[str, list[list[dict[str, Any]]]] = {}
        for policy in policies:
            if policy["kind"] != "sample":
                continue
            sample_pools[policy["name"]] = sample_records(
                model,
                condition_matrix,
                resolved,
                policy=policy,
                phase=phase,
                control=control,
                global_seed=seed,
                batch_size=int(cfg["generation"]["sample_query_batch_size"]),
                maximum_steps=maximum_steps,
                draws=int(cfg["generation"]["sample_draws"]),
                device=device,
            )
        for policy in policies:
            pool = beam_pool if policy["kind"] == "beam" else sample_pools[policy["name"]]
            if pool is None:
                raise RuntimeError("Candidate pool was not generated")
            for query_position in range(len(panel)):
                rows, stats = assemble_candidates(
                    phase=phase,
                    policy=policy,
                    control=control,
                    query_position=query_position,
                    target_index=int(panel.iloc[query_position]["target_index"]),
                    source_index=int(
                        panel.iloc[query_position][CONTROL_SOURCE_COLUMNS[control]]
                    ),
                    greedy=greedy[query_position],
                    raw_non_greedy=pool[query_position],
                    maximum_candidates=maximum_candidates,
                )
                candidate_rows.extend(rows)
                stat_rows.append(stats)
        del greedy, beam_pool, sample_pools
        gc.collect()
    del model, checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    candidates = pd.DataFrame(candidate_rows)
    generation_stats = pd.DataFrame(stat_rows)
    if candidates.empty:
        raise RuntimeError("No policy-accepted candidates were generated")
    print(
        f"{phase}: re-encoding {candidates['canonical_smiles'].nunique():,} unique candidates",
        flush=True,
    )
    embeddings, unique_smiles, reencoder_metadata = reencode_candidates(
        candidates,
        repo_root=repo_root,
        root=root,
        cfg=cfg,
        phase=phase,
        device=device,
    )
    candidates = annotate_candidates(
        candidates,
        embeddings=embeddings,
        unique_smiles=unique_smiles,
        panel=panel,
        molecules=molecules,
        controls=controls,
        cfg=cfg,
    )
    selections, summary = summarize(
        candidates,
        generation_stats,
        panel=panel,
        policies=policies,
        controls=controls,
        candidate_sizes=[int(value) for value in cfg["panels"]["candidate_set_sizes"]],
        historical_identity=float(cfg["evaluation"]["historical_step2_exact_identity"]),
    )
    candidates_path = root / "outputs" / "raw" / f"{phase}_candidates.parquet"
    selections_path = root / "outputs" / "raw" / f"{phase}_query_results.parquet"
    stats_path = root / "outputs" / "tables" / f"{phase}_generation_stats.csv"
    summary_path = root / "outputs" / "tables" / f"{phase}_metrics_by_policy_control_k.csv"
    atomic_write_parquet(candidates_path, candidates, root)
    atomic_write_parquet(selections_path, selections, root)
    atomic_write_csv(stats_path, generation_stats, root)
    atomic_write_csv(summary_path, summary, root)
    finished_at = utc_now()
    summary_json_path = root / "outputs" / f"{phase}_evaluation_summary.json"
    evaluation_summary = {
        "schema_version": 1,
        "phase": phase,
        "status": "complete",
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device),
        "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "rows_per_control": len(panel),
        "controls": list(controls),
        "policies": [policy["name"] for policy in policies],
        "candidate_sizes": cfg["panels"]["candidate_set_sizes"],
        "accepted_candidate_rows": len(candidates),
        "unique_accepted_candidate_molecules": len(unique_smiles),
        "ranking_inputs": [
            "released_gmolai_relative_l2_to_supplied_condition",
            "released_gmolai_cosine_to_supplied_condition_tie_break",
            "target_blind_generator_order_tie_break",
        ],
        "target_structural_information_used_for_ranking": False,
        "reencoder": reencoder_metadata,
        "decoder_checkpoint_sha256": input_hashes["decoder_checkpoint"],
        "gmolai_checkpoint_sha256": input_hashes["gmolai_checkpoint"],
        "calibrator_sha256": input_hashes["gmolai_calibrator"],
        "output_sha256": {
            "candidates": sha256_file(candidates_path),
            "query_results": sha256_file(selections_path),
            "generation_stats": sha256_file(stats_path),
            "metrics": sha256_file(summary_path),
            "reencoded_unique": sha256_file(
                root / "outputs" / "raw" / f"{phase}_reencoded_unique_molecules.npz"
            ),
        },
        "test_rows": 0,
        "endpoint_labels_used": False,
        "decoder_training": False,
        "encoder_training": False,
        "latent_perturbation": False,
        "derivative_generation": False,
        "input_sha256": input_hashes,
    }
    atomic_write_json(summary_json_path, evaluation_summary, root)
    state = {
        "schema_version": 1,
        "phase": phase,
        "status": "complete",
        "sealed_at": finished_at,
        "evaluation_summary_sha256": sha256_file(summary_json_path),
        "metrics_sha256": sha256_file(summary_path),
        "candidates_sha256": sha256_file(candidates_path),
        "query_results_sha256": sha256_file(selections_path),
        "decoder_checkpoint_sha256": input_hashes["decoder_checkpoint"],
        "fresh_validation_panel": phase == "final",
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    _, final_hashes = validate_manifest(repo_root, root)
    if final_hashes != input_hashes:
        raise RuntimeError("A frozen input changed during candidate evaluation")
    print(json.dumps(evaluation_summary, sort_keys=True))


if __name__ == "__main__":
    main()
