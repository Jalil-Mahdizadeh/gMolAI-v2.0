#!/usr/bin/env python3
"""Select and seal one candidate policy using train-development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    atomic_write_csv,
    atomic_write_json,
    load_json,
    protocol,
    sha256_file,
    utc_now,
    validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    output_path = root / "state" / "POLICY_FROZEN.json"
    if output_path.exists():
        print(output_path.read_text(encoding="utf-8"))
        return
    development_state = root / "state" / "DEVELOPMENT_COMPLETE.json"
    if not development_state.is_file():
        raise RuntimeError("Development candidate evaluation is incomplete")
    final_forbidden = [
        root / "state" / "FINAL_COMPLETE.json",
        root / "outputs" / "raw" / "final_candidates.parquet",
        root / "outputs" / "tables" / "final_metrics_by_policy_control_k.csv",
    ]
    if any(path.exists() for path in final_forbidden):
        raise RuntimeError("Refusing policy selection after final generation")
    cfg = protocol(root)
    paths, hashes = validate_manifest(repo_root, root)
    metrics_path = (
        root
        / "outputs"
        / "tables"
        / "development_metrics_by_policy_control_k.csv"
    )
    state = load_json(development_state)
    if sha256_file(metrics_path) != state["metrics_sha256"]:
        raise RuntimeError("Development metric table changed")
    metrics = pd.read_csv(metrics_path)
    maximum_k = max(int(value) for value in cfg["panels"]["candidate_set_sizes"])
    correct = metrics.loc[
        (metrics["control"] == "correct_embedding")
        & (metrics["candidate_set_size"] == maximum_k)
    ].copy()
    if len(correct) != len(cfg["generation"]["policies"]):
        raise RuntimeError("Development policy metric coverage is incomplete")
    order = {
        str(policy["name"]): int(policy["registered_order"])
        for policy in cfg["generation"]["policies"]
    }
    correct["registered_order"] = correct["policy"].map(order).astype(int)
    correct["eligible"] = (
        correct["latent_reranked_exact_target_identity_at_1"] + 1e-12
        >= correct["same_panel_greedy_target_identity"]
    )
    if not bool(correct["eligible"].any()):
        raise RuntimeError(
            "No preregistered policy preserved greedy exact identity on development"
        )
    eligible = correct.loc[correct["eligible"]].sort_values(
        [
            "latent_reranked_exact_target_identity_at_1",
            "oracle_target_recall_at_k",
            "candidate_set_full_rate",
            "registered_order",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = str(eligible.iloc[0]["policy"])
    selected_metrics = metrics.loc[
        (metrics["policy"] == selected)
        & (metrics["candidate_set_size"] == maximum_k)
    ].copy()
    selected_metrics = selected_metrics.sort_values("control")
    selection_table = correct.sort_values("registered_order")
    selection_path = root / "outputs" / "tables" / "development_policy_selection.csv"
    atomic_write_csv(selection_path, selection_table, root)
    selected_policy = next(
        policy for policy in cfg["generation"]["policies"] if policy["name"] == selected
    )
    final_panel_path = root / "prepared" / "fresh_validation_panel.csv"
    result = {
        "schema_version": 1,
        "status": "complete",
        "selected_policy": selected,
        "selected_policy_definition": selected_policy,
        "selected_at": utc_now(),
        "selected_before_final_generation": True,
        "selection_information_partition": "pretraining_train_decoder_development_only",
        "development_rows": int(cfg["panels"]["development_rows"]),
        "selection_candidate_size": maximum_k,
        "selection_order": cfg["development_selection"]["selection_order"],
        "selected_metrics_by_control": selected_metrics.to_dict(orient="records"),
        "development_metrics_sha256": sha256_file(metrics_path),
        "development_state_sha256": sha256_file(development_state),
        "selection_table_sha256": sha256_file(selection_path),
        "fresh_validation_panel_sha256_before_generation": sha256_file(final_panel_path),
        "decoder_checkpoint_sha256": hashes["decoder_checkpoint"],
        "gmolai_checkpoint_sha256": hashes["gmolai_checkpoint"],
        "target_information_used_for_candidate_ranking": False,
        "validation_generation_observed_during_selection": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(output_path, result, root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
