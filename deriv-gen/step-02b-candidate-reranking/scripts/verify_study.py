#!/usr/bin/env python3
"""Fail-closed integrity and target-blind-reranking verification for Step 2b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    atomic_write_json,
    load_json,
    protocol,
    sha256_file,
    utc_now,
    validate_manifest,
    write_hash_ledger,
)


def check(condition: bool, message: str, checks: list[dict[str, object]]) -> None:
    checks.append({"check": message, "pass": bool(condition)})
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    cfg = protocol(root)
    paths, hashes = validate_manifest(repo_root, root)
    checks: list[dict[str, object]] = []
    registered = load_json(root / "state" / "REGISTERED.json")
    check(
        registered["manifest_sha256"] == sha256_file(root / "inputs" / "manifest.json"),
        "registered manifest hash is intact",
        checks,
    )
    for relative, expected in registered["registered_source_sha256"].items():
        check(
            sha256_file(root / relative) == expected,
            f"registered source unchanged: {relative}",
            checks,
        )
    check(not list(root.rglob("*.pt")), "Step 2b contains no trained checkpoint", checks)
    required_states = [
        "PANELS_PREPARED.json",
        "COMPONENT_TESTS.json",
        "DEVELOPMENT_COMPLETE.json",
        "POLICY_FROZEN.json",
        "FINAL_COMPLETE.json",
        "COMPLETE.json",
    ]
    for name in required_states:
        check((root / "state" / name).is_file(), f"required state exists: {name}", checks)
    panel = pd.read_csv(root / "prepared" / "fresh_validation_panel.csv")
    original = pd.read_csv(paths["step2_original_final_panel"])
    overlap = set(panel["target_index"].astype(int)).intersection(
        set(original["validation_index"].astype(int))
    )
    check(len(panel) == int(cfg["panels"]["final_rows"]), "fresh final panel has registered size", checks)
    check(not overlap, "fresh final panel is disjoint from original Step-2 panel", checks)
    check(panel["target_index"].nunique() == len(panel), "fresh final targets are unique", checks)
    policy_seal = load_json(root / "state" / "POLICY_FROZEN.json")
    check(
        policy_seal["selected_before_final_generation"] is True,
        "policy was selected before final generation",
        checks,
    )
    check(
        policy_seal["fresh_validation_panel_sha256_before_generation"]
        == sha256_file(root / "prepared" / "fresh_validation_panel.csv"),
        "prospectively frozen final panel is unchanged",
        checks,
    )
    candidates_path = root / "outputs" / "raw" / "final_candidates.parquet"
    query_path = root / "outputs" / "raw" / "final_query_results.parquet"
    candidates = pd.read_parquet(candidates_path)
    query = pd.read_parquet(query_path)
    selected_policy = str(policy_seal["selected_policy"])
    check(set(candidates["policy"]) == {selected_policy}, "final candidates use one frozen policy", checks)
    check(
        candidates["proposal_rank"].between(1, 50).all(),
        "candidate proposal ranks stay inside registered sizes",
        checks,
    )
    check(
        not candidates.duplicated(["policy", "control", "query_position", "candidate_hash"]).any(),
        "candidate molecular identities are unique within every query set",
        checks,
    )
    check(
        candidates["canonical_smiles"].astype(str).str.len().gt(0).all(),
        "every ranked candidate is policy-accepted canonical SMILES",
        checks,
    )
    expected_query_rows = (
        int(cfg["panels"]["final_rows"])
        * len(cfg["panels"]["controls"])
        * len(cfg["panels"]["candidate_set_sizes"])
    )
    check(len(query) == expected_query_rows, "query-level metric coverage is complete", checks)
    for control in cfg["panels"]["controls"]:
        base = candidates.loc[
            (candidates["policy"] == selected_policy)
            & (candidates["control"] == control)
        ]
        for size in cfg["panels"]["candidate_set_sizes"]:
            subset = base.loc[base["proposal_rank"] <= int(size)].sort_values(
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
            expected = subset.drop_duplicates("query_position").set_index(
                "query_position"
            )["candidate_hash"]
            observed_rows = query.loc[
                (query["policy"] == selected_policy)
                & (query["control"] == control)
                & (query["candidate_set_size"] == int(size))
            ].set_index("query_position")
            observed = observed_rows["reranked_candidate_hash"].dropna().astype(str)
            expected = expected.loc[observed.index].astype(str)
            check(
                observed.equals(expected),
                f"target-blind relative-L2 reranking reproduced: {control}/k={size}",
                checks,
            )
    decision = load_json(root / "outputs" / "decision.json")
    check(decision["ranking_used_target_structure"] is False, "decision records target-blind ranking", checks)
    check(decision["latent_perturbation"] is False, "no latent perturbation was performed", checks)
    check(decision["derivative_generation"] is False, "no derivative generation was performed", checks)
    check(decision["test_rows"] == 0, "locked test rows used equals zero", checks)
    check(decision["endpoint_labels_used"] is False, "endpoint labels were not used", checks)
    check(
        decision["decoder_checkpoint_sha256"] == hashes["decoder_checkpoint"],
        "decision binds the frozen decoder",
        checks,
    )
    check(
        decision["gmolai_checkpoint_sha256"] == hashes["gmolai_checkpoint"],
        "decision binds frozen gMolAI",
        checks,
    )
    result = {
        "schema_version": 1,
        "status": "pass",
        "verified_at": utc_now(),
        "checks": checks,
        "checks_passed": len(checks),
        "decision": decision["decision"],
        "selected_policy": selected_policy,
        "fresh_panel_overlap_with_original_step2": 0,
        "decoder_checkpoint_sha256": hashes["decoder_checkpoint"],
        "gmolai_checkpoint_sha256": hashes["gmolai_checkpoint"],
        "test_rows": 0,
        "endpoint_labels_used": False,
        "latent_perturbation": False,
        "derivative_generation": False,
    }
    verification_path = root / "outputs" / "verification.json"
    atomic_write_json(verification_path, result, root)
    atomic_write_json(
        root / "state" / "VERIFIED.json",
        {
            "schema_version": 1,
            "status": "pass",
            "verified_at": result["verified_at"],
            "verification_sha256": sha256_file(verification_path),
            "checks_passed": len(checks),
        },
        root,
    )
    write_hash_ledger(root, root / "outputs" / "SHA256SUMS")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
