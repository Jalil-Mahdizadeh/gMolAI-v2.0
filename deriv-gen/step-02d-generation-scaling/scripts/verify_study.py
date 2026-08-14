#!/usr/bin/env python3
"""Fail-closed verification for the complete Step-2d study."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import (
    STEP_ROOT,
    atomic_write_json,
    load_json,
    protocol,
    resolve_manifest_inputs,
    sha256_file,
    utc_now,
    write_hash_ledger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    cfg = protocol(root)
    paths, input_hashes = resolve_manifest_inputs(repo_root, root)
    registered = load_json(root / "state" / "REGISTERED.json")
    if sha256_file(root / "inputs" / "manifest.json") != registered["manifest_sha256"]:
        raise RuntimeError("Registered manifest changed")
    for relative, expected in registered["registered_source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Registered source changed: {relative}")
    required_states = [
        "PANELS_PREPARED.json",
        "COMPONENT_TESTS.json",
        "DEVELOPMENT_GENERATION_COMPLETE.json",
        "DEVELOPMENT_ANALYSIS_COMPLETE.json",
        "STRATEGY_FROZEN.json",
        "FINAL_GENERATION_COMPLETE.json",
        "FINAL_ANALYSIS_COMPLETE.json",
        "REPORT_COMPLETE.json",
    ]
    for name in required_states:
        state = load_json(root / "state" / name)
        if state.get("status") not in {"complete", "frozen_before_final_generation"}:
            raise RuntimeError(f"Incomplete state: {name}")
        if state.get("test_rows", 0) != 0 or state.get("endpoint_labels_used", False) is not False:
            raise RuntimeError(f"Scientific boundary changed: {name}")
    frozen = load_json(root / "state" / "STRATEGY_FROZEN.json")
    final_generation = load_json(root / "state" / "FINAL_GENERATION_COMPLETE.json")
    if final_generation["strategies"] != [frozen["selected_strategy"]["name"]]:
        raise RuntimeError("Final generation does not use the frozen development strategy")
    frozen_time = datetime.fromisoformat(frozen["frozen_at"])
    for shard in range(int(cfg["execution"]["gpu_shards"])):
        state = load_json(root / "state" / f"FINAL_SHARD_{shard:02d}_COMPLETE.json")
        if datetime.fromisoformat(state["completed_at"]) <= frozen_time:
            raise RuntimeError("Final generation did not occur after strategy freeze")

    panels = load_json(root / "prepared" / "panel_metadata.json")
    development = pd.read_csv(root / "prepared" / "development_panel.csv")
    final = pd.read_csv(root / "prepared" / "fresh_validation_panel.csv")
    prior_dev = pd.read_csv(paths["step2b_development_panel"])
    prior_final = pd.read_csv(paths["step2b_final_panel"])
    original = pd.read_csv(paths["step2_original_final_panel"])
    if set(development["target_index"]).intersection(prior_dev["target_index"]):
        raise RuntimeError("Step-2d development overlaps Step-2b development")
    if set(final["target_index"]).intersection(prior_final["target_index"]):
        raise RuntimeError("Step-2d final overlaps Step-2b final")
    if set(final["target_index"]).intersection(original["validation_index"]):
        raise RuntimeError("Step-2d final overlaps original Step-2 final")
    if set(development["target_hash"]).intersection(final["target_hash"]):
        raise RuntimeError("Development/final identity overlap")
    novelty = pd.read_parquet(root / "prepared" / "decoder_training_identities.parquet")
    if len(novelty) != 980_000 or novelty["molecule_hash"].nunique() != 980_000:
        raise RuntimeError("Novelty reference changed")

    final_summary = pd.read_csv(root / "outputs" / "tables" / "final_budget_summary.csv")
    seed_metrics = pd.read_parquet(root / "outputs" / "tables" / "final_seed_budget_metrics.parquet")
    candidates = pd.read_parquet(root / "outputs" / "tables" / "final_candidate_characterization.parquet")
    if len(final_summary) != len(cfg["budgets"]):
        raise RuntimeError("Final budget-summary row count changed")
    expected_seed_metric_rows = len(final) * len(cfg["budgets"])
    if len(seed_metrics) != expected_seed_metric_rows:
        raise RuntimeError("Final seed-budget row count changed")
    if candidates.duplicated(["strategy", "query_position", "candidate_hash"]).any():
        raise RuntimeError("Candidate identity is duplicated within a seed/strategy")
    if int(candidates["first_proposal_rank"].max()) > 1000:
        raise RuntimeError("Candidate lies outside maximum raw budget")
    for field in (
        "unique_accepted_identity_count",
        "mmp_derivative_count",
        "novel_genuine_nonseed_count",
        "novel_useful_local_count",
        "distinct_scaffold_count",
    ):
        pivot = seed_metrics.pivot(index="query_position", columns="budget", values=field)
        if (np.diff(pivot[sorted(pivot.columns)].to_numpy(), axis=1) < 0).any():
            raise RuntimeError(f"Nested final metric decreased: {field}")
    final_raw_rows = 0
    for path in (root / "outputs" / "raw" / "final").glob("*.parquet"):
        final_raw_rows += int(pq.ParquetFile(path).metadata.num_rows)
    if final_raw_rows != len(final) * 1000:
        raise RuntimeError("Final raw proposal cardinality changed")
    if list(root.rglob("*.pt")) or list(root.rglob("*.pth")) or list(root.rglob("*.ckpt")):
        raise RuntimeError("Step 2d contains a model checkpoint")
    decision = load_json(root / "outputs" / "decision.json")
    if decision["selected_generation_strategy"]["name"] != frozen["selected_strategy"]["name"]:
        raise RuntimeError("Decision strategy differs from frozen strategy")

    verification = {
        "schema_version": 1,
        "status": "passed",
        "verified_at": utc_now(),
        "study_id": cfg["study_id"],
        "registered_sources_unchanged": True,
        "frozen_inputs_unchanged": True,
        "input_sha256": input_hashes,
        "frozen_decoder_checkpoint_sha256": input_hashes["decoder_checkpoint"],
        "frozen_gmolai_checkpoint_sha256": input_hashes["gmolai_checkpoint"],
        "frozen_calibrator_sha256": input_hashes["gmolai_calibrator"],
        "embedding_space": "released_hybrid_w3",
        "development_rows": len(development),
        "final_rows": len(final),
        "decoder_training_novelty_reference_rows": len(novelty),
        "final_raw_proposal_rows": final_raw_rows,
        "final_candidate_rows": len(candidates),
        "selected_strategy": frozen["selected_strategy"]["name"],
        "recommended_budget": decision["recommended_raw_proposal_budget"],
        "final_generation_after_strategy_freeze": True,
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "property_optimization": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    verification_path = root / "outputs" / "verification.json"
    atomic_write_json(verification_path, verification, root)
    verified_state = {
        "schema_version": 1,
        "status": "complete",
        "verified_at": utc_now(),
        "verification_sha256": sha256_file(verification_path),
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(root / "state" / "VERIFIED.json", verified_state, root)
    complete = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "study_id": cfg["study_id"],
        "verification_sha256": sha256_file(verification_path),
        "decision_sha256": sha256_file(root / "outputs" / "decision.json"),
        "selected_strategy": frozen["selected_strategy"]["name"],
        "recommended_budget": decision["recommended_raw_proposal_budget"],
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "property_optimization": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(root / "state" / "COMPLETE.json", complete, root)
    write_hash_ledger(root, root / "outputs" / "SHA256SUMS")
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
