#!/usr/bin/env python3
"""Validate and seal all raw proposal shards for one Step-2d phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from common import (
    STEP_ROOT,
    atomic_write_json,
    load_json,
    protocol,
    sha256_file,
    utc_now,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    args = parser.parse_args()
    root = args.step_root.resolve()
    phase = args.phase
    state_path = root / "state" / f"{phase.upper()}_GENERATION_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    cfg = protocol(root)
    shards = int(cfg["execution"]["gpu_shards"])
    panel_name = "development_panel.csv" if phase == "development" else "fresh_validation_panel.csv"
    panel = pd.read_csv(root / "prepared" / panel_name)
    if phase == "development":
        strategies = list(cfg["generation"]["development_strategies"])
    else:
        frozen = load_json(root / "state" / "STRATEGY_FROZEN.json")
        strategies = [dict(frozen["selected_strategy"])]
    output_hashes: dict[str, str] = {}
    total_rows = 0
    shard_seed_total = 0
    for shard in range(shards):
        seal_path = root / "state" / f"{phase.upper()}_SHARD_{shard:02d}_COMPLETE.json"
        if not seal_path.is_file():
            raise RuntimeError(f"Missing generation shard seal: {seal_path}")
        seal = load_json(seal_path)
        expected_positions = list(range(shard, len(panel), shards))
        if int(seal["seed_rows"]) != len(expected_positions):
            raise RuntimeError(f"Shard {shard} seed count changed")
        shard_seed_total += int(seal["seed_rows"])
        for strategy in strategies:
            name = strategy["name"]
            path = root / "outputs" / "raw" / phase / f"proposals-{name}-shard-{shard:02d}-of-{shards:02d}.parquet"
            if not path.is_file() or sha256_file(path) != seal["output_sha256"][name]:
                raise RuntimeError(f"Raw proposal shard missing or changed: {path}")
            metadata = pq.ParquetFile(path).metadata
            expected_rows = len(expected_positions) * 1000
            if metadata.num_rows != expected_rows:
                raise RuntimeError(f"Unexpected rows in {path}: {metadata.num_rows}")
            check = pd.read_parquet(
                path, columns=["query_position", "proposal_rank", "strategy", "phase"]
            )
            if set(check["query_position"].unique()) != set(expected_positions):
                raise RuntimeError(f"Shard query identities changed: {path}")
            grouped = check.groupby("query_position")["proposal_rank"]
            if not ((grouped.count() == 1000) & (grouped.min() == 1) & (grouped.max() == 1000)).all():
                raise RuntimeError(f"Proposal ranks are not complete: {path}")
            if set(check["strategy"]) != {name} or set(check["phase"]) != {phase}:
                raise RuntimeError(f"Phase/strategy annotation changed: {path}")
            output_hashes[path.relative_to(root).as_posix()] = sha256_file(path)
            total_rows += int(metadata.num_rows)
    if shard_seed_total != len(panel):
        raise RuntimeError("GPU shards do not cover the panel exactly")
    state = {
        "schema_version": 1,
        "status": "complete",
        "phase": phase,
        "sealed_at": utc_now(),
        "seed_rows": len(panel),
        "strategy_count": len(strategies),
        "strategies": [value["name"] for value in strategies],
        "raw_rows": total_rows,
        "raw_proposals_per_strategy_seed": 1000,
        "output_sha256": output_hashes,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
