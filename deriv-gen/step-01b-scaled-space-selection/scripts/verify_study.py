#!/usr/bin/env python3
"""Read-only integrity and completeness verifier for the scaled study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analysis_core import METHOD_ORDER, RETRIEVAL_METRICS, SPACE_ORDER
from scaled_common import load_validate_manifest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01b-scaled-space-selection"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    expected = (
        repo_root / "deriv-gen" / "step-01b-scaled-space-selection"
    ).resolve()
    if step_root != expected:
        raise RuntimeError(f"Unexpected study root: {step_root}")

    required = [
        step_root / "README.md",
        step_root / "PROTOCOL.md",
        step_root / "RESULTS.md",
        step_root / "DECISION.md",
        step_root / "config" / "protocol.json",
        step_root / "inputs" / "manifest.json",
        step_root / "state" / "EXPORT_COMPLETE.json",
        step_root / "state" / "FRAGMENTATION_COMPLETE.json",
        step_root / "state" / "MMP_MINING_COMPLETE.json",
        step_root / "state" / "run_metadata.json",
        step_root / "state" / "COMPLETE.json",
        step_root / "outputs" / "SHA256SUMS",
        step_root / "outputs" / "study_summary.json",
        step_root / "outputs" / "space_decision.json",
        step_root / "outputs" / "raw" / "mmp_alignment_per_observation.parquet",
        step_root / "outputs" / "raw" / "retrieval_per_query_and_replicate.parquet",
        step_root / "outputs" / "raw" / "retrieval_per_query.parquet",
        step_root / "outputs" / "tables" / "mmp_support_thresholds.csv",
        step_root / "outputs" / "tables" / "mmp_support_by_transformation.csv",
        step_root / "outputs" / "tables" / "alignment_by_transformation.csv",
        step_root / "outputs" / "tables" / "retrieval_by_transformation.csv",
        step_root / "outputs" / "tables" / "retrieval_summary.csv",
        step_root / "outputs" / "tables" / "hierarchical_bootstrap_summary.csv",
        step_root / "outputs" / "tables" / "paired_differences_vs_released_w3.csv",
        step_root / "outputs" / "tables" / "space_selection.csv",
        step_root / "outputs" / "figures" / "primary_space_comparison.png",
        step_root / "outputs" / "figures" / "primary_space_comparison.svg",
        step_root / "outputs" / "figures" / "alignment_by_support.png",
        step_root / "outputs" / "figures" / "alignment_by_support.svg",
        step_root / "outputs" / "figures" / "retrieval_accuracy_similarity.png",
        step_root / "outputs" / "figures" / "retrieval_accuracy_similarity.svg",
        step_root / "intermediate" / "train_molecules.parquet",
        step_root / "intermediate" / "validation_molecules.parquet",
        step_root / "intermediate" / "retrieval_queries.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required study artifacts: {missing}")

    config = json.loads(
        (step_root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (step_root / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    _, input_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )

    for seal_name in (
        "EXPORT_COMPLETE.json",
        "FRAGMENTATION_COMPLETE.json",
        "MMP_MINING_COMPLETE.json",
    ):
        seal = json.loads(
            (step_root / "state" / seal_name).read_text(encoding="utf-8")
        )
        if seal.get("status") != "complete":
            raise RuntimeError(f"Stage seal is incomplete: {seal_name}")
        for record in seal.get("outputs", {}).values():
            artifact = step_root / record["path"]
            if (
                not artifact.is_file()
                or sha256_file(artifact) != record["sha256"]
            ):
                raise RuntimeError(
                    f"Stage artifact hash mismatch: {artifact}"
                )

    outputs = step_root / "outputs"
    ledger = outputs / "SHA256SUMS"
    ledger_entries = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        artifact = (outputs / relative).resolve()
        if outputs.resolve() not in artifact.parents:
            raise RuntimeError(f"Output ledger path escapes root: {relative}")
        if (
            not artifact.is_file()
            or sha256_file(artifact) != digest
        ):
            raise RuntimeError(f"Output ledger mismatch: {relative}")
        ledger_entries += 1
    if ledger_entries < 20:
        raise RuntimeError(
            f"Suspiciously short output ledger: {ledger_entries}"
        )

    complete = json.loads(
        (step_root / "state" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    if complete.get("status") != "complete":
        raise RuntimeError("Final completion seal is not complete")
    if not complete.get("single_gpu"):
        raise RuntimeError("Final seal does not record single-GPU execution")
    if int(complete.get("train_rows", -1)) != int(config["train_rows"]):
        raise RuntimeError("Final train population differs from protocol")
    if int(complete.get("validation_rows", -1)) != int(
        config["validation_rows"]
    ):
        raise RuntimeError("Final validation population differs from protocol")
    if int(complete.get("test_rows", -1)) != 0:
        raise RuntimeError("Locked-test molecules entered the study")
    if (
        complete.get("decoder_conditioning_representation")
        != "released_hybrid_w3"
    ):
        raise RuntimeError("Released decoder conditioning identity changed")
    if (
        sha256_file(step_root / "RESULTS.md")
        != complete["results_sha256"]
        or sha256_file(step_root / "DECISION.md")
        != complete["decision_sha256"]
        or sha256_file(ledger)
        != complete["output_ledger_sha256"]
    ):
        raise RuntimeError("Final report or output ledger identity mismatch")

    support = pd.read_csv(
        outputs / "tables" / "mmp_support_thresholds.csv"
    )
    if set(support["minimum_train_cores"].astype(int)) != {2, 5, 10, 20}:
        raise RuntimeError("Required MMP support thresholds are incomplete")
    if not (
        support.sort_values("minimum_train_cores")["transformations"]
        .to_numpy()[::-1]
        >= 0
    ).all():
        raise RuntimeError("Invalid support counts")
    support_sorted = support.sort_values("minimum_train_cores")
    if not support_sorted["transformations"].is_monotonic_decreasing:
        raise RuntimeError("Transformation support counts are not nested")

    spaces = set(SPACE_ORDER)
    methods = set(METHOD_ORDER)
    retrieval = pd.read_parquet(
        outputs / "raw" / "retrieval_per_query_and_replicate.parquet"
    )
    if set(retrieval["space"]) != spaces:
        raise RuntimeError("One or more candidate spaces are absent")
    for space in SPACE_ORDER:
        subset = retrieval.loc[retrieval["space"] == space]
        if set(subset["method"]) != methods:
            raise RuntimeError(f"Retrieval method set differs in {space}")
        for method in METHOD_ORDER:
            expected_replicates = (
                {0, 1, 2}
                if method
                in {"isotropic", "global_covariance", "local_covariance"}
                else {0}
            )
            observed = set(
                subset.loc[
                    subset["method"] == method, "replicate"
                ].astype(int)
            )
            if observed != expected_replicates:
                raise RuntimeError(
                    f"Replicate set differs for {space}/{method}: {observed}"
                )

    query_sets = {
        (
            space,
            method,
        ): set(
            retrieval.loc[
                (retrieval["space"] == space)
                & (retrieval["method"] == method),
                "query_id",
            ]
        )
        for space in SPACE_ORDER
        for method in METHOD_ORDER
    }
    canonical_queries = query_sets[(SPACE_ORDER[0], METHOD_ORDER[0])]
    if not canonical_queries or any(
        values != canonical_queries for values in query_sets.values()
    ):
        raise RuntimeError(
            "Molecule identities or query panel differ across spaces/methods"
        )
    if len(canonical_queries) != int(complete["retrieval_queries"]):
        raise RuntimeError("Final retrieval query count is inconsistent")

    hash_columns = (
        "seed_hash",
        "true_target_hash",
        "top1_candidate_hash",
    )
    for column in hash_columns:
        if not retrieval[column].astype(str).str.fullmatch(
            r"[0-9a-f]{64}"
        ).all():
            raise RuntimeError(f"Invalid retrieval identity: {column}")
    validation_molecules = pd.read_parquet(
        step_root / "intermediate" / "validation_molecules.parquet"
    )
    if validation_molecules["molecule_index"].duplicated().any():
        raise RuntimeError("Validation molecule indices are duplicated")
    hash_by_index = validation_molecules.set_index("molecule_index")[
        "molecule_hash"
    ].astype(str)
    for index_column, hash_column in (
        ("seed_index", "seed_hash"),
        ("true_target_index", "true_target_hash"),
        ("top1_candidate_index", "top1_candidate_hash"),
    ):
        expected_hashes = retrieval[index_column].map(hash_by_index)
        if expected_hashes.isna().any() or not expected_hashes.reset_index(
            drop=True
        ).equals(
            retrieval[hash_column].astype(str).reset_index(drop=True)
        ):
            raise RuntimeError(
                f"Retrieval index/hash mismatch: {hash_column}"
            )

    averaged = pd.read_parquet(
        outputs / "raw" / "retrieval_per_query.parquet"
    )
    for metric in RETRIEVAL_METRICS:
        if metric not in averaged or averaged[metric].isna().any():
            raise RuntimeError(f"Missing retrieval metric: {metric}")
    if (
        averaged[list(RETRIEVAL_METRICS)].select_dtypes("number") < 0
    ).any(axis=None):
        raise RuntimeError("A bounded retrieval metric is negative")

    alignment = pd.read_parquet(
        outputs / "raw" / "mmp_alignment_per_observation.parquet"
    )
    if set(alignment["space"]) != spaces:
        raise RuntimeError("Alignment spaces are incomplete")
    alignment_sets = {
        space: set(
            alignment.loc[
                alignment["space"] == space, "observation_id"
            ]
        )
        for space in SPACE_ORDER
    }
    canonical_alignment = alignment_sets[SPACE_ORDER[0]]
    if not canonical_alignment or any(
        values != canonical_alignment
        for values in alignment_sets.values()
    ):
        raise RuntimeError("Alignment observations are not paired")
    mismatch_counts = alignment.groupby("observation_id")[
        "mismatched_transform"
    ].nunique()
    if not (mismatch_counts == 1).all():
        raise RuntimeError(
            "Mismatched-direction assignments differ across spaces"
        )

    bootstrap = pd.read_csv(
        outputs / "tables" / "hierarchical_bootstrap_summary.csv"
    )
    for analysis in ("alignment_all", "retrieval_mmp_direction"):
        for threshold in (2, 5, 10, 20):
            subset = bootstrap.loc[
                (bootstrap["analysis"] == analysis)
                & (bootstrap["minimum_train_cores"] == threshold)
            ]
            if not subset.empty and set(subset["space"]) != spaces:
                raise RuntimeError(
                    f"Bootstrap spaces incomplete for {analysis}/{threshold}"
                )

    selection = pd.read_csv(
        outputs / "tables" / "space_selection.csv"
    )
    if set(selection["space"]) != spaces:
        raise RuntimeError("Selection table spaces are incomplete")
    selected_rows = selection.loc[
        selection["selected_edit_control_space"].astype(bool)
    ]
    selected = complete.get("selected_edit_control_space")
    if selected is None:
        if len(selected_rows):
            raise RuntimeError("Selection table and final seal disagree")
    elif (
        len(selected_rows) != 1
        or str(selected_rows.iloc[0]["space"]) != selected
    ):
        raise RuntimeError("Selected edit-control space is inconsistent")

    decision = json.loads(
        (outputs / "space_decision.json").read_text(encoding="utf-8")
    )
    if (
        decision["selected_edit_control_space"] != selected
        or decision["decoder_conditioning_representation"]
        != "released_hybrid_w3"
    ):
        raise RuntimeError("Machine-readable decision is inconsistent")
    summary = json.loads(
        (outputs / "study_summary.json").read_text(encoding="utf-8")
    )
    if (
        int(summary["data"]["train_rows"]) != 1_000_000
        or int(summary["data"]["validation_rows"]) != 50_000
        or int(summary["data"]["test_rows"]) != 0
        or int(summary["data"]["identity_overlap"]) != 0
    ):
        raise RuntimeError("Study population summary is invalid")
    if (
        summary["data"]["checkpoint_sha256"] != input_hashes["checkpoint"]
        or summary["data"]["calibrator_sha256"]
        != input_hashes["calibrator"]
    ):
        raise RuntimeError("Frozen model identities changed")

    print(
        json.dumps(
            {
                "status": "verified",
                "ledger_entries": ledger_entries,
                "spaces": list(SPACE_ORDER),
                "methods_per_space": len(METHOD_ORDER),
                "alignment_observations": len(canonical_alignment),
                "retrieval_queries": len(canonical_queries),
                "selected_edit_control_space": selected,
                "decoder_conditioning_representation": (
                    "released_hybrid_w3"
                ),
                "test_rows": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
