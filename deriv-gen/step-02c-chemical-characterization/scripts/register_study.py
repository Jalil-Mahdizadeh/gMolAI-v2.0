#!/usr/bin/env python3
"""Hash-bind the frozen Step-1b/Step-2b inputs before Step 2c analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    STEP_ROOT,
    atomic_write_json,
    load_json,
    protocol,
    sha256_file,
    utc_now,
)


EXTERNAL_INPUTS = {
    "external_step2b_candidates": "deriv-gen/step-02b-candidate-reranking/outputs/raw/final_candidates.parquet",
    "external_step2b_generation_stats": "deriv-gen/step-02b-candidate-reranking/outputs/tables/final_generation_stats.csv",
    "external_step2b_panel": "deriv-gen/step-02b-candidate-reranking/prepared/fresh_validation_panel.csv",
    "external_step2b_protocol": "deriv-gen/step-02b-candidate-reranking/config/protocol.json",
    "external_step2b_manifest": "deriv-gen/step-02b-candidate-reranking/inputs/manifest.json",
    "external_step2b_policy_seal": "deriv-gen/step-02b-candidate-reranking/state/POLICY_FROZEN.json",
    "external_step2b_final_seal": "deriv-gen/step-02b-candidate-reranking/state/FINAL_COMPLETE.json",
    "external_step2b_complete": "deriv-gen/step-02b-candidate-reranking/state/COMPLETE.json",
    "external_step2b_verification": "deriv-gen/step-02b-candidate-reranking/outputs/verification.json",
    "external_step2b_evaluation_source": "deriv-gen/step-02b-candidate-reranking/scripts/evaluate_panel.py",
    "external_step2b_common_source": "deriv-gen/step-02b-candidate-reranking/scripts/common.py",
    "external_validation_molecules": "deriv-gen/step-01b-scaled-space-selection/intermediate/validation_molecules.parquet",
    "external_step1b_protocol": "deriv-gen/step-01b-scaled-space-selection/config/protocol.json",
    "external_step1b_fragment_source": "deriv-gen/step-01b-scaled-space-selection/scripts/scaled_common.py",
    "external_chemistry_policy": "src/gmolai_retrain/chem.py",
    "external_resolved_config": "runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/resolved_config.json",
}

LOCAL_SOURCES = {
    "source_registration": "scripts/register_study.py",
    "source_protocol": "config/protocol.json",
    "source_common": "scripts/common.py",
    "source_audit_core": "scripts/audit_core.py",
    "source_audit": "scripts/audit_candidates.py",
    "source_report": "scripts/report_results.py",
    "source_verify": "scripts/verify_study.py",
    "source_component_tests": "scripts/test_components.py",
    "source_runner": "scripts/run_study.sh",
}


def file_record(path: Path, displayed_path: str) -> dict[str, object]:
    return {
        "path": displayed_path,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=STEP_ROOT.parents[2])
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--supersede-runtime-reason", default="")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    cfg = protocol(step_root)

    if (step_root / "state" / "ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("Refusing to register after Step 2c analysis")

    step2b_complete = load_json(repo_root / EXTERNAL_INPUTS["external_step2b_complete"])
    step2b_final = load_json(repo_root / EXTERNAL_INPUTS["external_step2b_final_seal"])
    policy_seal = load_json(repo_root / EXTERNAL_INPUTS["external_step2b_policy_seal"])
    step2b_protocol = load_json(repo_root / EXTERNAL_INPUTS["external_step2b_protocol"])
    if step2b_complete.get("status") != "complete" or step2b_complete.get("decision") != "GO":
        raise RuntimeError("Step 2b is not sealed complete with GO")
    if step2b_complete.get("derivative_generation") is not False:
        raise RuntimeError("Unexpected Step-2b scientific boundary")
    if step2b_final.get("fresh_validation_panel") is not True:
        raise RuntimeError("Step-2b final panel is not marked fresh")
    expected_policy = cfg["candidate_population"]["source_policy"]
    if policy_seal.get("selected_policy") != expected_policy:
        raise RuntimeError("Step-2b frozen policy differs from Step-2c protocol")
    if step2b_protocol["generation"]["maximum_unique_candidates"] != 50:
        raise RuntimeError("Step-2b candidate budget changed")

    files: dict[str, dict[str, object]] = {}
    for role, relative in EXTERNAL_INPUTS.items():
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[role] = file_record(path, relative)
    for role, relative in LOCAL_SOURCES.items():
        path = step_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[role] = file_record(path, str(path.relative_to(repo_root)))

    manifest = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "registered_at": utc_now(),
        "source_step2b_policy": expected_policy,
        "source_step2b_control": cfg["candidate_population"]["source_control"],
        "source_step2b_candidates_sha256": files["external_step2b_candidates"]["sha256"],
        "model_execution": False,
        "training": False,
        "candidate_regeneration": False,
        "latent_perturbation": False,
        "mmp_directed_generation": False,
        "locked_test_rows": 0,
        "endpoint_labels_used": False,
        "files": files,
    }
    manifest_path = step_root / "inputs" / "manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        comparable_existing = dict(existing)
        comparable_new = dict(manifest)
        comparable_existing.pop("registered_at", None)
        comparable_new.pop("registered_at", None)
        if comparable_existing != comparable_new:
            if not args.supersede_runtime_reason:
                raise RuntimeError(
                    "Existing Step-2c registration differs from current inputs; "
                    "a runtime-only supersession reason is required"
                )
            previous_hash = sha256_file(manifest_path)
            archive_path = (
                step_root
                / "inputs"
                / f"manifest.superseded-{previous_hash[:12]}.json"
            )
            if not archive_path.exists():
                atomic_write_json(archive_path, existing, step_root)
            manifest["supersedes_manifest_sha256"] = previous_hash
            manifest["supersession_reason"] = str(args.supersede_runtime_reason)
            atomic_write_json(manifest_path, manifest, step_root)
            atomic_write_json(
                step_root / "state" / "REGISTERED_SUPERSEDED_RUNTIME_MEMORY.json",
                {
                    "schema_version": 1,
                    "status": "superseded_before_completed_analysis",
                    "previous_manifest_sha256": previous_hash,
                    "previous_manifest_archive": str(
                        archive_path.relative_to(step_root)
                    ),
                    "replacement_manifest_sha256": sha256_file(manifest_path),
                    "reason": str(args.supersede_runtime_reason),
                    "scientific_protocol_changed": False,
                    "candidate_data_inspected_for_scientific_selection": False,
                },
                step_root,
            )
        else:
            manifest = existing
    else:
        atomic_write_json(manifest_path, manifest, step_root)

    seal = {
        "schema_version": 1,
        "status": "registered_before_analysis",
        "registered_at": manifest["registered_at"],
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(files),
        "candidate_regeneration": False,
        "model_execution": False,
    }
    atomic_write_json(step_root / "state" / "REGISTERED.json", seal, step_root)
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
