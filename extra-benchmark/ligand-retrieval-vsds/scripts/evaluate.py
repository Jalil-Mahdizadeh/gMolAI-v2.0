#!/usr/bin/env python3
"""Run frozen 5-shot, 1-shot, and scaffold-excluded ligand retrieval."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_csv,
    read_panel_tsv,
    read_tsv,
    sha256_file,
    write_csv,
)
from metrics import candidate_mask, compute_metrics, deterministic_random_scores


def anchor_digest(identities: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()


def load_anchor_schedule(path: Path):
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["target_id"], int(row["shots"]), int(row["draw_id"]))].append(row)
    schedule = {}
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["anchor_rank"]))
        shots = key[1]
        if len(ordered) != shots or [int(row["anchor_rank"]) for row in ordered] != list(range(shots)):
            raise RuntimeError(f"Invalid anchor ranks for {key}")
        identities = tuple(row["anchor_molecule_hash"] for row in ordered)
        if len(set(identities)) != shots:
            raise RuntimeError(f"Duplicate anchor identity for {key}")
        seeds = {int(row["draw_seed"]) for row in ordered}
        if len(seeds) != 1:
            raise RuntimeError(f"Inconsistent draw seed for {key}")
        schedule[key] = (next(iter(seeds)), identities)
    return schedule


def similarity_to_unique_anchors(
    model: str,
    target_vectors: np.ndarray,
    anchor_positions: list[int],
) -> np.ndarray:
    values = np.asarray(target_vectors, dtype=np.float64)
    if model == "morgan":
        if np.count_nonzero((values != 0.0) & (values != 1.0)):
            raise RuntimeError("Morgan representation is not binary")
        anchor_values = values[np.asarray(anchor_positions, dtype=np.int64)]
        intersections = values @ anchor_values.T
        value_counts = values.sum(axis=1)[:, None]
        anchor_counts = anchor_values.sum(axis=1)[None, :]
        union = value_counts + anchor_counts - intersections
        if np.any(union <= 0.0):
            raise RuntimeError("Morgan Tanimoto encountered an empty union")
        return np.clip(intersections / union, 0.0, 1.0)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-12) or not np.isfinite(norms).all():
        raise RuntimeError(f"Invalid learned vectors for {model}")
    normalized = values / norms[:, None]
    anchors = normalized[np.asarray(anchor_positions, dtype=np.int64)]
    return np.clip(normalized @ anchors.T, -1.0, 1.0)


def metric_row(
    *,
    model: str,
    display_name: str,
    is_control: bool,
    target_id: str,
    target_class: str,
    shots: int,
    condition: str,
    draw_id: int,
    draw_seed: int,
    anchors: tuple[str, ...],
    scores: np.ndarray,
    labels: np.ndarray,
    protocol: dict,
) -> dict[str, object]:
    result = compute_metrics(
        scores,
        labels,
        ef_fraction=0.01,
        bedroc_alpha=float(protocol["metrics"]["bedroc_alpha"]),
    )
    return {
        "model": model,
        "display_name": display_name,
        "is_random_control": is_control,
        "target_id": target_id,
        "target_class": target_class,
        "shots": shots,
        "condition": condition,
        "draw_id": draw_id,
        "draw_seed": draw_seed,
        "anchor_molecule_hashes": ";".join(anchors),
        "anchor_identity_sha256": anchor_digest(anchors),
        "inactive_or_lower_affinity_count": int(labels.size - labels.sum()),
        **result,
    }


def main() -> None:
    started = time.perf_counter()
    protocol = load_protocol()
    models = tuple(protocol["models"]["primary_order"])
    population_path = BENCHMARK_DIR / "state/POPULATION_FROZEN.json"
    anchors_state_path = BENCHMARK_DIR / "state/ANCHORS_FROZEN.json"
    population = load_json(population_path)
    anchors_state = load_json(anchors_state_path)
    if population.get("status") != "frozen" or anchors_state.get("status") != "frozen":
        raise RuntimeError("Population and anchors must be frozen before evaluation")
    anchor_path = BENCHMARK_DIR / "results/tables/anchor_draws.csv"
    if anchors_state.get("anchor_draws_sha256") != sha256_file(anchor_path):
        raise RuntimeError("Frozen anchor table changed")
    schedule = load_anchor_schedule(anchor_path)
    common_panel_path = BENCHMARK_DIR / "inputs/prepared/common_panel.tsv"
    membership_path = BENCHMARK_DIR / "inputs/prepared/common_memberships.tsv"
    if population.get("common_panel_sha256") != sha256_file(common_panel_path):
        raise RuntimeError("Frozen common panel changed")
    if population.get("common_memberships_sha256") != sha256_file(membership_path):
        raise RuntimeError("Frozen target memberships changed")
    panel = read_panel_tsv(common_panel_path)
    global_index = {row["molecule_hash"]: int(row["panel_index"]) for row in panel}
    target_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(membership_path):
        target_rows[row["target_id"]].append(row)
    target_metadata = {
        row["target_id"]: row for row in read_tsv(BENCHMARK_DIR / "inputs/prepared/targets.tsv")
    }
    draws = int(protocol["retrieval"]["draws_per_target"])
    primary_shots = int(protocol["retrieval"]["primary_shots"])
    secondary_shots = int(protocol["retrieval"]["secondary_shots"])
    scaffold_min_active = int(
        protocol["coverage_and_eligibility"]["scaffold_draw_minimum_remaining_actives"]
    )
    scaffold_min_inactive = int(
        protocol["coverage_and_eligibility"]["scaffold_draw_minimum_remaining_inactives"]
    )
    maps_path = BENCHMARK_DIR / "state/model_index_maps.npz"
    if population.get("model_index_maps_sha256") != sha256_file(maps_path):
        raise RuntimeError("Frozen model index maps changed")
    index_maps_file = np.load(maps_path, allow_pickle=False)
    output_rows: list[dict[str, object]] = []
    try:
        for model in models:
            embedding_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{model}.npy"
            if population["models"][model]["embedding_sha256"] != sha256_file(embedding_path):
                raise RuntimeError(f"Frozen representation changed for {model}")
            embedding = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            common_to_model = np.asarray(index_maps_file[model], dtype=np.int64)
            if common_to_model.shape != (len(panel),):
                raise RuntimeError(f"Invalid common index map for {model}")
            for target_id in sorted(target_rows):
                rows = sorted(target_rows[target_id], key=lambda row: row["molecule_hash"])
                identities = tuple(row["molecule_hash"] for row in rows)
                labels = np.asarray(
                    [row["label"] == "active" for row in rows], dtype=np.int8
                )
                scaffolds = np.asarray([row["scaffold"] for row in rows], dtype=object)
                target_global = np.asarray(
                    [global_index[identity] for identity in identities], dtype=np.int64
                )
                target_model = common_to_model[target_global]
                target_vectors = np.asarray(embedding[target_model])
                index_by_identity = {
                    identity: index for index, identity in enumerate(identities)
                }
                unique_anchor_identities = sorted(
                    {
                        identity
                        for shots in (secondary_shots, primary_shots)
                        for draw_id in range(draws)
                        for identity in schedule[(target_id, shots, draw_id)][1]
                    }
                )
                unique_anchor_positions = [
                    index_by_identity[identity] for identity in unique_anchor_identities
                ]
                similarity = similarity_to_unique_anchors(
                    model, target_vectors, unique_anchor_positions
                )
                anchor_column = {
                    identity: index for index, identity in enumerate(unique_anchor_identities)
                }
                for shots in (secondary_shots, primary_shots):
                    for draw_id in range(draws):
                        draw_seed, anchors = schedule[(target_id, shots, draw_id)]
                        anchor_positions = [index_by_identity[value] for value in anchors]
                        columns = [anchor_column[value] for value in anchors]
                        scores_all = np.max(similarity[:, columns], axis=1)
                        conditions = ("standard",)
                        if shots == primary_shots:
                            conditions = ("standard", "scaffold_excluded")
                        for condition in conditions:
                            mask = candidate_mask(
                                labels,
                                scaffolds,
                                anchor_positions,
                                scaffold_excluded=condition == "scaffold_excluded",
                            )
                            candidate_labels = labels[mask]
                            if condition == "scaffold_excluded" and (
                                int(candidate_labels.sum()) < scaffold_min_active
                                or int(candidate_labels.size - candidate_labels.sum())
                                < scaffold_min_inactive
                            ):
                                continue
                            output_rows.append(
                                metric_row(
                                    model=model,
                                    display_name=protocol["models"][model]["display_name"],
                                    is_control=False,
                                    target_id=target_id,
                                    target_class=target_metadata[target_id]["target_class"],
                                    shots=shots,
                                    condition=condition,
                                    draw_id=draw_id,
                                    draw_seed=draw_seed,
                                    anchors=anchors,
                                    scores=scores_all[mask],
                                    labels=candidate_labels,
                                    protocol=protocol,
                                )
                            )
            del embedding
    finally:
        index_maps_file.close()

    master_seed = int(protocol["retrieval"]["anchor_master_seed"])
    for target_id in sorted(target_rows):
        rows = sorted(target_rows[target_id], key=lambda row: row["molecule_hash"])
        identities = tuple(row["molecule_hash"] for row in rows)
        labels = np.asarray([row["label"] == "active" for row in rows], dtype=np.int8)
        scaffolds = np.asarray([row["scaffold"] for row in rows], dtype=object)
        index_by_identity = {identity: index for index, identity in enumerate(identities)}
        for shots in (secondary_shots, primary_shots):
            for draw_id in range(draws):
                draw_seed, anchors = schedule[(target_id, shots, draw_id)]
                anchor_positions = [index_by_identity[value] for value in anchors]
                scores_all = deterministic_random_scores(
                    identities,
                    target_id=target_id,
                    shots=shots,
                    draw_id=draw_id,
                    master_seed=master_seed,
                )
                conditions = ("standard",)
                if shots == primary_shots:
                    conditions = ("standard", "scaffold_excluded")
                for condition in conditions:
                    mask = candidate_mask(
                        labels,
                        scaffolds,
                        anchor_positions,
                        scaffold_excluded=condition == "scaffold_excluded",
                    )
                    candidate_labels = labels[mask]
                    if condition == "scaffold_excluded" and (
                        int(candidate_labels.sum()) < scaffold_min_active
                        or int(candidate_labels.size - candidate_labels.sum())
                        < scaffold_min_inactive
                    ):
                        continue
                    output_rows.append(
                        metric_row(
                            model="random",
                            display_name="Random ranking",
                            is_control=True,
                            target_id=target_id,
                            target_class=target_metadata[target_id]["target_class"],
                            shots=shots,
                            condition=condition,
                            draw_id=draw_id,
                            draw_seed=draw_seed,
                            anchors=anchors,
                            scores=scores_all[mask],
                            labels=candidate_labels,
                            protocol=protocol,
                        )
                    )
    output_rows.sort(
        key=lambda row: (
            str(row["condition"]),
            int(row["shots"]),
            str(row["model"]),
            str(row["target_id"]),
            int(row["draw_id"]),
        )
    )
    output_path = BENCHMARK_DIR / "results/tables/retrieval_per_draw.csv"
    write_csv(output_path, output_rows, tuple(output_rows[0]))
    result = {
        "schema_version": 1,
        "status": "ok",
        "population_state_sha256": sha256_file(population_path),
        "anchors_state_sha256": sha256_file(anchors_state_path),
        "models": list(models),
        "random_control_included": True,
        "rows": len(output_rows),
        "retrieval_per_draw": str(output_path),
        "retrieval_per_draw_sha256": sha256_file(output_path),
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state/RETRIEVAL_COMPLETE.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

