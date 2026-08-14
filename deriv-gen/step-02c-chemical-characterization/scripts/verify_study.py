#!/usr/bin/env python3
"""Fail-closed verification of the completed Step 2c chemistry audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=STEP_ROOT.parents[2])
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    cfg = protocol(step_root)
    paths, input_hashes = resolve_manifest_inputs(repo_root, step_root)
    checks: list[dict[str, Any]] = []

    def check(condition: bool, description: str, detail: Any = None) -> None:
        record = {"check": description, "passed": bool(condition)}
        if detail is not None:
            record["detail"] = detail
        checks.append(record)
        if not condition:
            raise RuntimeError(f"Verification failed: {description}: {detail}")

    for seal_name, expected_status in (
        ("REGISTERED.json", "registered_before_analysis"),
        ("COMPONENT_TESTS.json", "passed"),
        ("ANALYSIS_COMPLETE.json", "analysis_complete"),
        ("REPORT_COMPLETE.json", "report_complete"),
    ):
        path = step_root / "state" / seal_name
        check(path.is_file(), f"state seal exists: {seal_name}")
        check(
            load_json(path).get("status") == expected_status,
            f"state seal status: {seal_name}",
            load_json(path).get("status"),
        )

    summary = load_json(step_root / "outputs" / "study_summary.json")
    decision = load_json(step_root / "outputs" / "decision.json")
    candidates = pd.read_parquet(
        step_root / "outputs" / "tables" / "candidate_characterization.parquet"
    )
    seeds = pd.read_parquet(
        step_root / "outputs" / "tables" / "seed_characterization.parquet"
    )
    explanations = pd.read_parquet(
        step_root / "outputs" / "tables" / "mmp_explanations.parquet"
    )
    category_counts = pd.read_csv(
        step_root / "outputs" / "tables" / "candidate_category_counts.csv"
    )
    coverage = pd.read_csv(
        step_root / "outputs" / "tables" / "mmp_seed_coverage.csv"
    )

    expected_seeds = int(cfg["candidate_population"]["expected_seed_rows"])
    check(len(seeds) == expected_seeds, "all frozen seeds represented", len(seeds))
    check(
        seeds["query_position"].tolist() == list(range(expected_seeds)),
        "seed query positions are complete and ordered",
    )
    check(
        len(candidates) == int(summary["population"]["retained_candidate_rows"]),
        "candidate row count matches summary",
        len(candidates),
    )
    check(
        candidates["candidate_row_id"].tolist() == list(range(len(candidates))),
        "candidate identifiers are complete and ordered",
    )
    check(
        not candidates.duplicated(["query_position", "candidate_hash"]).any(),
        "molecular identities are unique within seed sets",
    )
    check(
        not candidates.duplicated(["query_position", "raw_smiles"]).any(),
        "retained raw SMILES are unique within seed sets",
    )
    check(
        int(candidates.groupby("query_position").size().max()) <= 50,
        "candidate-set size does not exceed 50",
    )
    check(candidates["audit_rdkit_valid"].astype(bool).all(), "all retained strings reparse")
    check(
        candidates["audit_policy_accepted"].astype(bool).all(),
        "all retained strings pass the unchanged policy",
    )
    check(
        set(candidates["chemical_category"].astype(str))
        == set(cfg["classification_precedence"]),
        "all frozen categories occur and no unknown category exists",
    )
    check(
        int(category_counts["candidate_count"].sum()) == len(candidates),
        "category counts partition candidate rows",
    )

    seed_identity = candidates["candidate_hash"].astype(str) == candidates["seed_hash"].astype(str)
    reported_seed_identity = candidates["is_seed_identity"].astype(bool)
    check(
        np.array_equal(seed_identity.to_numpy(), reported_seed_identity.to_numpy()),
        "seed molecular identity is hash-defined",
    )
    exact_spelling = seed_identity & (
        candidates["raw_smiles"].astype(str)
        == candidates["seed_canonical_smiles"].astype(str)
    )
    check(
        (
            candidates.loc[exact_spelling, "chemical_category"].astype(str)
            == "exact_seed_identity"
        ).all(),
        "canonical seed spellings are not counted as derivatives",
    )
    check(
        (
            candidates.loc[seed_identity & ~exact_spelling, "chemical_category"].astype(str)
            == "same_identity_alternative_smiles"
        ).all(),
        "alternate seed spellings are not counted as derivatives",
    )

    mmp_ids = set(candidates.loc[candidates["is_one_cut_mmp"], "candidate_row_id"].astype(int))
    explanation_ids = set(explanations["candidate_row_id"].astype(int)) if len(explanations) else set()
    check(mmp_ids == explanation_ids, "MMP flags equal the explanation-table support")
    if len(explanations):
        primary_counts = (
            explanations.loc[explanations["is_primary_explanation"].astype(bool)]
            .groupby("candidate_row_id")
            .size()
        )
        check(
            len(primary_counts) == len(mmp_ids) and (primary_counts == 1).all(),
            "every MMP candidate has exactly one deterministic primary explanation",
        )
        check(
            explanations["candidate_row_id"].isin(mmp_ids).all(),
            "explanation rows refer only to MMP candidates",
        )
    check(
        (~candidates.loc[candidates["is_one_cut_mmp"], "is_seed_identity"].astype(bool)).all(),
        "seed identities are never counted as MMP derivatives",
    )

    nonempty_seed = candidates["seed_has_bemis_murcko_scaffold"].astype(bool)
    retained = candidates["retains_nonempty_seed_scaffold"].astype(bool)
    same_key = candidates["same_scaffold_key"].astype(bool)
    check(
        np.array_equal(retained.to_numpy(), (nonempty_seed & same_key).to_numpy()),
        "non-empty scaffold retention has the registered denominator",
    )
    both_acyclic = candidates["both_seed_and_candidate_acyclic"].astype(bool)
    check(
        not (both_acyclic & nonempty_seed).any(),
        "acyclic status is separate from non-empty scaffold retention",
    )

    with np.load(
        step_root / "outputs" / "raw" / "within_set_pairwise_morgan.npz"
    ) as payload:
        all_values = payload["all_candidate_pairwise_morgan"]
        all_offsets = payload["all_candidate_query_offsets"]
        nonseed_values = payload["nonseed_candidate_pairwise_morgan"]
        nonseed_offsets = payload["nonseed_candidate_query_offsets"]
    check(len(all_offsets) == expected_seeds + 1, "all-candidate pairwise offsets cover every seed")
    check(
        len(nonseed_offsets) == expected_seeds + 1,
        "non-seed pairwise offsets cover every seed",
    )
    check(
        np.all(np.diff(all_offsets) >= 0) and int(all_offsets[-1]) == len(all_values),
        "all-candidate pairwise offsets are valid",
    )
    check(
        np.all(np.diff(nonseed_offsets) >= 0)
        and int(nonseed_offsets[-1]) == len(nonseed_values),
        "non-seed pairwise offsets are valid",
    )
    check(
        np.isfinite(all_values).all()
        and np.all((all_values >= 0) & (all_values <= 1)),
        "all pairwise Morgan values are finite and bounded",
    )
    check(
        np.isfinite(nonseed_values).all()
        and np.all((nonseed_values >= 0) & (nonseed_values <= 1)),
        "non-seed pairwise Morgan values are finite and bounded",
    )
    check(
        int(summary["within_set_diversity"]["all_candidate_pair_count"])
        == len(all_values),
        "pairwise count matches summary",
    )

    recomputed_thresholds = []
    for threshold in cfg["mmp"]["seed_coverage_thresholds"]:
        count = int((seeds["mmp_derivative_count"] >= int(threshold)).sum())
        recomputed_thresholds.append((int(threshold), count))
    reported_thresholds = [
        (int(row["minimum_mmp_derivatives"]), int(row["seed_count"]))
        for _, row in coverage.iterrows()
    ]
    check(
        recomputed_thresholds == reported_thresholds,
        "MMP seed-coverage thresholds reproduce from the seed table",
    )

    gates_cfg = cfg["bounded_conclusion"]
    values = decision["gate_values"]
    recomputed_gates = {
        "genuine_nonseed_yield": values["fraction_seeds_with_5_genuine_nonseed"]
        >= gates_cfg["minimum_fraction_seeds_with_5_genuine_nonseed"],
        "mmp_seed_coverage": values["fraction_seeds_with_1_mmp"]
        >= gates_cfg["minimum_fraction_seeds_with_1_mmp"],
        "scaffold_locality": values["nonempty_seed_scaffold_retention"]
        >= gates_cfg["minimum_nonempty_seed_scaffold_retention"],
        "seed_similarity_local_not_trivial": (
            gates_cfg["minimum_global_nonseed_median_morgan"]
            <= values["global_nonseed_median_morgan"]
            <= gates_cfg["maximum_global_nonseed_median_morgan"]
        ),
        "unique_nonseed_yield": values["median_unique_nonseed_per_seed"]
        >= gates_cfg["minimum_median_unique_nonseed_per_seed"],
        "within_set_nontrivial_diversity": values["global_pairwise_mean_morgan"]
        <= gates_cfg["maximum_global_pairwise_mean_morgan"],
    }
    check(recomputed_gates == decision["gates"], "bounded decision gates reproduce")
    check(
        summary["bounded_conclusion"]["classification"] == decision["classification"],
        "summary and decision classifications agree",
    )

    forbidden_model_files = [
        path
        for path in step_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
    ]
    check(not forbidden_model_files, "Step 2c contains no model/checkpoint artifact")
    check(summary["runtime"]["model_executed"] is False, "no model was executed")
    for key in (
        "encoder_training",
        "decoder_training",
        "candidate_regeneration",
        "latent_perturbation",
        "mmp_directed_generation",
    ):
        check(summary["scientific_boundary"][key] is False, f"scientific boundary: {key}=false")
    check(summary["scientific_boundary"]["locked_test_rows"] == 0, "locked test rows are zero")
    check(summary["scientific_boundary"]["endpoint_labels_used"] is False, "endpoint labels were not used")

    required_docs = ["README.md", "PROTOCOL.md", "DESIGN.md", "RESULTS.md", "DECISION.md"]
    for name in required_docs:
        check((step_root / name).is_file(), f"documentation exists: {name}")
    figure_files = sorted((step_root / "outputs" / "figures").glob("*"))
    check(
        len([path for path in figure_files if path.suffix == ".png"]) >= 4,
        "concise PNG figures exist",
    )
    check(
        len([path for path in figure_files if path.suffix == ".svg"]) >= 4,
        "matching SVG figures exist",
    )

    _, final_input_hashes = resolve_manifest_inputs(repo_root, step_root)
    check(input_hashes == final_input_hashes, "every frozen prior artifact remains unchanged")

    verification = {
        "schema_version": 1,
        "status": "verified",
        "verified_at": utc_now(),
        "check_count": len(checks),
        "passed_count": sum(int(record["passed"]) for record in checks),
        "checks": checks,
        "classification": decision["classification"],
        "input_sha256": input_hashes,
    }
    verification_path = step_root / "outputs" / "verification.json"
    atomic_write_json(verification_path, verification, step_root)
    complete = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "classification": decision["classification"],
        "seed_rows": len(seeds),
        "candidate_rows": len(candidates),
        "mmp_candidate_rows": len(mmp_ids),
        "model_executed": False,
        "candidate_regeneration": False,
        "latent_perturbation": False,
        "mmp_directed_generation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
        "study_summary_sha256": sha256_file(step_root / "outputs" / "study_summary.json"),
        "candidate_table_sha256": sha256_file(
            step_root / "outputs" / "tables" / "candidate_characterization.parquet"
        ),
        "seed_table_sha256": sha256_file(
            step_root / "outputs" / "tables" / "seed_characterization.parquet"
        ),
        "decision_sha256": sha256_file(step_root / "outputs" / "decision.json"),
        "verification_sha256": sha256_file(verification_path),
    }
    atomic_write_json(step_root / "state" / "COMPLETE.json", complete, step_root)
    write_hash_ledger(step_root)

    ledger_path = step_root / "outputs" / "SHA256SUMS"
    ledger_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    check(len(ledger_lines) > 20, "hash ledger covers study artifacts", len(ledger_lines))
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
