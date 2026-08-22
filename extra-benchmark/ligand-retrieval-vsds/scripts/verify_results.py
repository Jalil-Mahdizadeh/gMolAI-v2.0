#!/usr/bin/env python3
"""Independently verify population, anchors, retrieval rows, and final artifacts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    protocol_digest,
    read_csv,
    read_tsv,
    sha256_file,
)
from metrics import candidate_mask, deterministic_anchor_sample


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def require_hash(manifest: dict, key: str, path: Path) -> None:
    expected = manifest.get(key)
    observed = sha256_file(path)
    if expected != observed:
        raise RuntimeError(
            f"Artifact hash mismatch for {path}: expected {expected}, observed {observed}"
        )


def verify_figure_tree(value: object) -> int:
    verified = 0
    if isinstance(value, list):
        for item in value:
            verified += verify_figure_tree(item)
    elif isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"])
            if not path.is_absolute():
                path = BENCHMARK_DIR.parents[1] / path
            if sha256_file(path) != value["sha256"]:
                raise RuntimeError(f"Figure or source-data artifact changed: {path}")
            verified += 1
        for item in value.values():
            verified += verify_figure_tree(item)
    return verified


def anchor_digest(identities: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    protocol = load_protocol()
    models = tuple(protocol["models"]["primary_order"])
    if len(models) != 7 or len(set(models)) != 7 or "random" in models:
        raise RuntimeError("The frozen primary model roster is not exactly seven models")
    all_models = (*models, "random")
    preflight = load_json(BENCHMARK_DIR / "audits/preflight.json")
    preparation = load_json(BENCHMARK_DIR / "audits/data_preparation.json")
    screening = load_json(BENCHMARK_DIR / "state/SCREENING_COMPLETE.json")
    population_path = BENCHMARK_DIR / "state/POPULATION_FROZEN.json"
    anchors_path = BENCHMARK_DIR / "state/ANCHORS_FROZEN.json"
    retrieval_path = BENCHMARK_DIR / "state/RETRIEVAL_COMPLETE.json"
    summary_path = BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json"
    population = load_json(population_path)
    anchors = load_json(anchors_path)
    retrieval = load_json(retrieval_path)
    summary = load_json(summary_path)
    exposure = load_json(BENCHMARK_DIR / "audits/pretraining_exposure.json")
    figures = load_json(BENCHMARK_DIR / "audits/figure_manifest.json")
    report = load_json(BENCHMARK_DIR / "state/REPORT_COMPLETE.json")
    states = (preflight, preparation, screening, population, anchors, retrieval, summary, exposure, figures, report)
    if any(state.get("status") not in {"ok", "frozen"} for state in states):
        raise RuntimeError("At least one required benchmark stage is incomplete")
    if preflight.get("protocol_sha256") != protocol_digest(protocol):
        raise RuntimeError("Canonical protocol digest changed since preflight")
    if population.get("protocol_sha256") != sha256_file(BENCHMARK_DIR / "protocol.json"):
        raise RuntimeError("Protocol bytes changed since population freeze")
    if preparation.get("targets") != int(protocol["data"]["gap_definition"]["expected_targets"]):
        raise RuntimeError("Prepared target count differs from the source definition")
    if preparation.get("label_contradictions_retained") != 0:
        raise RuntimeError("A contradictory target-label identity was retained")

    coverage_path = BENCHMARK_DIR / "results/tables/model_coverage.csv"
    require_hash(population, "model_coverage_sha256", coverage_path)
    coverage = read_csv(coverage_path)
    if {row["model"] for row in coverage} != set(models) or len(coverage) != len(models):
        raise RuntimeError("Model coverage does not contain exactly the frozen seven models")
    for model in models:
        audit_path = BENCHMARK_DIR / "audits" / f"embedding-{model}.json"
        audit = load_json(audit_path)
        if audit.get("status") != "ok" or audit.get("model") != model:
            raise RuntimeError(f"Invalid embedding audit for {model}")
        if population["models"][model]["validation_sha256"] != sha256_file(audit_path):
            raise RuntimeError(f"Embedding validation changed after population freeze: {model}")
        embedding_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{model}.npy"
        if audit.get("embedding_sha256") != sha256_file(embedding_path):
            raise RuntimeError(f"Embedding changed after validation: {model}")
        if int(audit.get("nonfinite_vectors", -1)) != 0 or int(audit.get("zero_norm_vectors", -1)) != 0:
            raise RuntimeError(f"Invalid vectors were accepted for {model}")

    membership_path = BENCHMARK_DIR / "inputs/prepared/common_memberships.tsv"
    require_hash(population, "common_memberships_sha256", membership_path)
    memberships = read_tsv(membership_path)
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in memberships:
        by_target[row["target_id"]].append(row)
    if set(by_target) != set(population["eligible_target_ids"]):
        raise RuntimeError("Frozen target identities disagree with memberships")
    if len(by_target) != int(population["eligible_targets"]):
        raise RuntimeError("Frozen eligible-target count is inconsistent")

    anchor_table_path = BENCHMARK_DIR / "results/tables/anchor_draws.csv"
    require_hash(anchors, "anchor_draws_sha256", anchor_table_path)
    grouped_anchors: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(anchor_table_path):
        grouped_anchors[(row["target_id"], int(row["shots"]), int(row["draw_id"]))].append(row)
    draws = int(protocol["retrieval"]["draws_per_target"])
    primary_shots = int(protocol["retrieval"]["primary_shots"])
    secondary_shots = int(protocol["retrieval"]["secondary_shots"])
    master_seed = int(protocol["retrieval"]["anchor_master_seed"])
    schedule: dict[tuple[str, int, int], tuple[int, tuple[str, ...]]] = {}
    for target_id, target_rows in sorted(by_target.items()):
        active_ids = sorted(
            row["molecule_hash"] for row in target_rows if row["label"] == "active"
        )
        for shots in (secondary_shots, primary_shots):
            for draw_id in range(draws):
                key = (target_id, shots, draw_id)
                rows = sorted(grouped_anchors.get(key, []), key=lambda row: int(row["anchor_rank"]))
                if len(rows) != shots or [int(row["anchor_rank"]) for row in rows] != list(range(shots)):
                    raise RuntimeError(f"Missing or malformed anchor draw: {key}")
                expected_seed, expected_ids = deterministic_anchor_sample(
                    active_ids,
                    target_id=target_id,
                    shots=shots,
                    draw_id=draw_id,
                    master_seed=master_seed,
                )
                observed_ids = tuple(row["anchor_molecule_hash"] for row in rows)
                if observed_ids != expected_ids or {int(row["draw_seed"]) for row in rows} != {expected_seed}:
                    raise RuntimeError(f"Anchor schedule is not reproducible for {key}")
                schedule[key] = (expected_seed, expected_ids)
    if len(grouped_anchors) != len(schedule):
        raise RuntimeError("Unexpected anchor draws exist outside the frozen population")

    scaffold_eligibility_path = BENCHMARK_DIR / "results/tables/scaffold_draw_eligibility.csv"
    require_hash(anchors, "scaffold_draw_eligibility_sha256", scaffold_eligibility_path)
    recorded_scaffold = {
        (row["target_id"], int(row["draw_id"])): row
        for row in read_csv(scaffold_eligibility_path)
    }
    retrieval_table_path = BENCHMARK_DIR / "results/tables/retrieval_per_draw.csv"
    require_hash(retrieval, "retrieval_per_draw_sha256", retrieval_table_path)
    retrieval_rows = read_csv(retrieval_table_path)
    grouped_retrieval: dict[tuple[str, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in retrieval_rows:
        grouped_retrieval[(row["condition"], int(row["shots"]), row["target_id"], int(row["draw_id"]))].append(row)

    expected_group_keys: set[tuple[str, int, str, int]] = set()
    invariant_fields = (
        "draw_seed",
        "anchor_molecule_hashes",
        "anchor_identity_sha256",
        "candidate_count",
        "active_count",
        "inactive_or_lower_affinity_count",
        "cutoff_k",
        "realized_screened_fraction",
    )
    scaffold_min_active = int(protocol["coverage_and_eligibility"]["scaffold_draw_minimum_remaining_actives"])
    scaffold_min_inactive = int(protocol["coverage_and_eligibility"]["scaffold_draw_minimum_remaining_inactives"])
    scaffold_eligible_draws = 0
    for target_id, unsorted_rows in sorted(by_target.items()):
        target_rows = sorted(unsorted_rows, key=lambda row: row["molecule_hash"])
        identities = tuple(row["molecule_hash"] for row in target_rows)
        labels = np.asarray([row["label"] == "active" for row in target_rows], dtype=np.int8)
        scaffolds = np.asarray([row["scaffold"] for row in target_rows], dtype=object)
        index_by_id = {identity: index for index, identity in enumerate(identities)}
        for shots in (secondary_shots, primary_shots):
            for draw_id in range(draws):
                seed, anchor_ids = schedule[(target_id, shots, draw_id)]
                anchor_indices = [index_by_id[identity] for identity in anchor_ids]
                conditions = ["standard"]
                standard_mask = candidate_mask(labels, scaffolds, anchor_indices, scaffold_excluded=False)
                if np.any(standard_mask[np.asarray(anchor_indices, dtype=np.int64)]):
                    raise RuntimeError("An anchor leaked into a candidate pool")
                if shots == primary_shots:
                    scaffold_mask = candidate_mask(labels, scaffolds, anchor_indices, scaffold_excluded=True)
                    scaffold_active = int(labels[scaffold_mask].sum())
                    scaffold_inactive = int(scaffold_mask.sum() - scaffold_active)
                    eligible = scaffold_active >= scaffold_min_active and scaffold_inactive >= scaffold_min_inactive
                    record = recorded_scaffold.get((target_id, draw_id))
                    if record is None or as_bool(record["eligible"]) != eligible:
                        raise RuntimeError(f"Scaffold eligibility differs for {target_id}/{draw_id}")
                    expected_record = {
                        "standard_candidates": int(standard_mask.sum()),
                        "standard_remaining_actives": int(labels[standard_mask].sum()),
                        "scaffold_candidates": int(scaffold_mask.sum()),
                        "scaffold_remaining_actives": scaffold_active,
                    }
                    if any(int(record[key]) != value for key, value in expected_record.items()):
                        raise RuntimeError(f"Scaffold candidate accounting differs for {target_id}/{draw_id}")
                    if eligible:
                        conditions.append("scaffold_excluded")
                        scaffold_eligible_draws += 1
                for condition in conditions:
                    key = (condition, shots, target_id, draw_id)
                    expected_group_keys.add(key)
                    group = grouped_retrieval.get(key, [])
                    if len(group) != len(all_models) or {row["model"] for row in group} != set(all_models):
                        raise RuntimeError(f"Cross-model retrieval rows are incomplete for {key}")
                    for field in invariant_fields:
                        if len({row[field] for row in group}) != 1:
                            raise RuntimeError(f"Cross-model {field} differs for {key}")
                    mask = standard_mask if condition == "standard" else scaffold_mask
                    candidate_count = int(mask.sum())
                    active_count = int(labels[mask].sum())
                    inactive_count = candidate_count - active_count
                    cutoff = max(1, math.ceil(0.01 * candidate_count))
                    expected_fraction = cutoff / candidate_count
                    reference = group[0]
                    if int(reference["draw_seed"]) != seed:
                        raise RuntimeError(f"Draw seed changed for {key}")
                    if reference["anchor_molecule_hashes"] != ";".join(anchor_ids):
                        raise RuntimeError(f"Anchor identities changed for {key}")
                    if reference["anchor_identity_sha256"] != anchor_digest(anchor_ids):
                        raise RuntimeError(f"Anchor identity digest changed for {key}")
                    if (
                        int(reference["candidate_count"]) != candidate_count
                        or int(reference["active_count"]) != active_count
                        or int(reference["inactive_or_lower_affinity_count"]) != inactive_count
                        or int(reference["cutoff_k"]) != cutoff
                        or not math.isclose(float(reference["realized_screened_fraction"]), expected_fraction, rel_tol=0.0, abs_tol=1e-15)
                    ):
                        raise RuntimeError(f"Candidate or EF cutoff accounting differs for {key}")
                    if active_count <= 0 or inactive_count <= 0:
                        raise RuntimeError(f"A retrieval pool lacks one label for {key}")
                    for row in group:
                        for metric in ("bedroc20", "roc_auc", "average_precision"):
                            if not 0.0 <= float(row[metric]) <= 1.0:
                                raise RuntimeError(f"Out-of-range {metric} for {key}/{row['model']}")
                        if float(row["ef1"]) < 0.0:
                            raise RuntimeError(f"Negative EF1% for {key}/{row['model']}")
    if set(grouped_retrieval) != expected_group_keys:
        raise RuntimeError("Retrieval table has missing or unexpected condition/draw groups")
    if len(retrieval_rows) != len(expected_group_keys) * len(all_models):
        raise RuntimeError("Retrieval table row count is inconsistent")
    if scaffold_eligible_draws != sum(as_bool(row["eligible"]) for row in recorded_scaffold.values()):
        raise RuntimeError("Scaffold-eligible draw count is inconsistent")

    per_target_path = BENCHMARK_DIR / "results/tables/retrieval_per_target.csv"
    model_summary_path = BENCHMARK_DIR / "results/tables/model_summary.csv"
    paired_path = BENCHMARK_DIR / "results/tables/paired_comparisons.csv"
    scaffold_summary_path = BENCHMARK_DIR / "results/tables/scaffold_excluded_summary.csv"
    require_hash(summary, "retrieval_per_target_sha256", per_target_path)
    require_hash(summary, "model_summary_sha256", model_summary_path)
    require_hash(summary, "paired_comparisons_sha256", paired_path)
    require_hash(summary, "scaffold_excluded_summary_sha256", scaffold_summary_path)
    random_primary = [
        float(row["ef1"])
        for row in retrieval_rows
        if row["model"] == "random"
        and row["condition"] == "standard"
        and int(row["shots"]) == primary_shots
    ]
    random_mean = float(np.mean(random_primary))
    if not math.isclose(random_mean, float(summary["random_primary_ef1_mean"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Random-ranking audit mean differs from the summary state")
    if not 0.7 <= random_mean <= 1.3:
        raise RuntimeError("Random-ranking EF1% sanity control is outside [0.7, 1.3]")
    primary_summary_models = {
        row["model"]
        for row in read_csv(model_summary_path)
        if int(row["shots"]) == primary_shots
        and row["condition"] == "standard"
        and row["metric"] == "ef1"
        and not as_bool(row["is_random_control"])
    }
    if primary_summary_models != set(models):
        raise RuntimeError("Primary summary ranking does not contain exactly seven models")

    if exposure.get("no_unseen_or_ood_claim") is not True or exposure.get("pretrained_model_executed") is not False:
        raise RuntimeError("Exposure audit interpretation or execution contract changed")
    require_hash(exposure, "summary_table_sha256", BENCHMARK_DIR / "results/tables/pretraining_exposure.csv")
    require_hash(exposure, "identity_ledger_sha256", BENCHMARK_DIR / "audits/gmolai_pretraining_exposure_ledger.csv")
    figure_artifacts = verify_figure_tree(figures)
    require_hash(report, "report_sha256", BENCHMARK_DIR / "RESULTS.md")

    result = {
        "schema_version": 1,
        "status": "ok",
        "protocol_sha256": sha256_file(BENCHMARK_DIR / "protocol.json"),
        "models_verified": list(models),
        "targets_verified": len(by_target),
        "anchor_schedules_verified": len(schedule),
        "retrieval_groups_verified": len(expected_group_keys),
        "retrieval_rows_verified": len(retrieval_rows),
        "scaffold_eligible_draws_verified": scaffold_eligible_draws,
        "figure_and_source_data_artifacts_verified": figure_artifacts,
        "random_primary_ef1_mean": random_mean,
        "same_anchors_candidates_and_realized_cutoffs_for_all_models": True,
        "anchors_absent_from_candidate_pools": True,
        "all_seven_common_support_enforced": True,
        "formal_p_values_performed": False,
        "verification_inputs": {
            "population_state_sha256": sha256_file(population_path),
            "anchors_state_sha256": sha256_file(anchors_path),
            "retrieval_state_sha256": sha256_file(retrieval_path),
            "summary_state_sha256": sha256_file(summary_path),
            "exposure_audit_sha256": sha256_file(BENCHMARK_DIR / "audits/pretraining_exposure.json"),
            "figure_manifest_sha256": sha256_file(BENCHMARK_DIR / "audits/figure_manifest.json"),
            "report_state_sha256": sha256_file(BENCHMARK_DIR / "state/REPORT_COMPLETE.json"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audits/verification.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
