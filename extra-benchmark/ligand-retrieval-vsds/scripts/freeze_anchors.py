#!/usr/bin/env python3
"""Freeze exact 1-shot and 5-shot anchors after population freeze."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_csv,
    read_tsv,
    sha256_file,
    sha256_lines,
    write_csv,
)
from metrics import candidate_mask, deterministic_anchor_sample


def main() -> None:
    protocol = load_protocol()
    population_path = BENCHMARK_DIR / "state" / "POPULATION_FROZEN.json"
    population = load_json(population_path)
    if population.get("status") != "frozen" or population.get("performance_inspected"):
        raise RuntimeError("Population is not safely frozen before anchor generation")
    membership_path = BENCHMARK_DIR / "inputs/prepared/common_memberships.tsv"
    if population.get("common_memberships_sha256") != sha256_file(membership_path):
        raise RuntimeError("Frozen membership population changed")
    memberships = read_tsv(membership_path)
    by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in memberships:
        by_target[row["target_id"]].append(row)
    settings = protocol["retrieval"]
    draws = int(settings["draws_per_target"])
    master_seed = int(settings["anchor_master_seed"])
    shot_counts = (int(settings["secondary_shots"]), int(settings["primary_shots"]))
    anchor_rows = []
    scaffold_draw_rows = []
    for target_id in sorted(by_target):
        rows = sorted(by_target[target_id], key=lambda row: row["molecule_hash"])
        active_ids = tuple(
            row["molecule_hash"] for row in rows if row["label"] == "active"
        )
        index_by_hash = {row["molecule_hash"]: index for index, row in enumerate(rows)}
        labels = np.asarray([row["label"] == "active" for row in rows], dtype=np.int8)
        scaffolds = np.asarray([row["scaffold"] for row in rows], dtype=object)
        for shots in shot_counts:
            for draw_id in range(draws):
                draw_seed, anchors = deterministic_anchor_sample(
                    active_ids,
                    target_id=target_id,
                    shots=shots,
                    draw_id=draw_id,
                    master_seed=master_seed,
                )
                for anchor_rank, identity in enumerate(anchors):
                    anchor_rows.append(
                        {
                            "target_id": target_id,
                            "shots": shots,
                            "draw_id": draw_id,
                            "draw_seed": draw_seed,
                            "anchor_rank": anchor_rank,
                            "anchor_molecule_hash": identity,
                            "anchor_scaffold": rows[index_by_hash[identity]]["scaffold"],
                        }
                    )
                if shots == int(settings["primary_shots"]):
                    anchor_indices = [index_by_hash[identity] for identity in anchors]
                    standard = candidate_mask(
                        labels, scaffolds, anchor_indices, scaffold_excluded=False
                    )
                    scaffold = candidate_mask(
                        labels, scaffolds, anchor_indices, scaffold_excluded=True
                    )
                    scaffold_draw_rows.append(
                        {
                            "target_id": target_id,
                            "draw_id": draw_id,
                            "standard_candidates": int(standard.sum()),
                            "standard_remaining_actives": int(labels[standard].sum()),
                            "scaffold_candidates": int(scaffold.sum()),
                            "scaffold_remaining_actives": int(labels[scaffold].sum()),
                            "scaffold_removed_candidates": int(standard.sum() - scaffold.sum()),
                            "eligible": (
                                int(labels[scaffold].sum())
                                >= int(
                                    protocol["coverage_and_eligibility"][
                                        "scaffold_draw_minimum_remaining_actives"
                                    ]
                                )
                                and int(scaffold.sum() - labels[scaffold].sum())
                                >= int(
                                    protocol["coverage_and_eligibility"][
                                        "scaffold_draw_minimum_remaining_inactives"
                                    ]
                                )
                            ),
                        }
                    )
    anchor_path = BENCHMARK_DIR / "results/tables/anchor_draws.csv"
    scaffold_draw_path = BENCHMARK_DIR / "results/tables/scaffold_draw_eligibility.csv"
    write_csv(anchor_path, anchor_rows, tuple(anchor_rows[0]))
    write_csv(scaffold_draw_path, scaffold_draw_rows, tuple(scaffold_draw_rows[0]))
    eligible_draws = CounterLike(row for row in scaffold_draw_rows if row["eligible"])
    target_population = read_csv(
        BENCHMARK_DIR / "results/tables/population_target_summary.csv"
    )
    minimum_draws = int(
        protocol["coverage_and_eligibility"][
            "scaffold_target_minimum_eligible_draws"
        ]
    )
    target_summary = []
    for row in target_population:
        target_id = row["target_id"]
        count = eligible_draws.get(target_id, 0)
        target_summary.append(
            {
                **row,
                "five_shot_draws": draws if row["primary_target_eligible"] == "True" else 0,
                "scaffold_eligible_draws": count,
                "scaffold_target_eligible": row["primary_target_eligible"] == "True"
                and count >= minimum_draws,
            }
        )
    target_summary_path = BENCHMARK_DIR / "results/tables/target_summary.csv"
    write_csv(target_summary_path, target_summary, tuple(target_summary[0]))
    result = {
        "schema_version": 1,
        "status": "frozen",
        "freeze_stage": "after_population_freeze_before_retrieval",
        "population_state_sha256": sha256_file(population_path),
        "targets": len(by_target),
        "draws_per_target": draws,
        "shot_counts": list(shot_counts),
        "anchor_rows": len(anchor_rows),
        "anchor_draws": str(anchor_path),
        "anchor_draws_sha256": sha256_file(anchor_path),
        "anchor_identity_schedule_sha256": sha256_lines(
            f"{row['target_id']}:{row['shots']}:{row['draw_id']}:"
            f"{row['anchor_rank']}:{row['anchor_molecule_hash']}"
            for row in anchor_rows
        ),
        "scaffold_draw_eligibility": str(scaffold_draw_path),
        "scaffold_draw_eligibility_sha256": sha256_file(scaffold_draw_path),
        "scaffold_eligible_targets": sum(
            str(row["scaffold_target_eligible"]) == "True" for row in target_summary
        ),
        "target_summary": str(target_summary_path),
        "target_summary_sha256": sha256_file(target_summary_path),
        "performance_inspected": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "ANCHORS_FROZEN.json", result)
    print(json.dumps(result, sort_keys=True))


def CounterLike(rows):
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["target_id"])] += 1
    return counts


if __name__ == "__main__":
    main()

