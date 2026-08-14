#!/usr/bin/env python3
"""Read-only integrity and completeness verifier for Step 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from decoder_model import ConditionalSmilesTransformer, decoder_parameter_count
from study_common import (
    load_validate_manifest,
    scaffold_group_keys,
    sha256_file,
)

CONTROLS = {
    "correct_embedding",
    "shuffled_embedding",
    "zero_embedding",
    "wrong_molecule_embedding",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-02-decoder-feasibility"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    expected = (repo_root / "deriv-gen" / "step-02-decoder-feasibility").resolve()
    if root != expected:
        raise RuntimeError(f"Unexpected Step 2 root: {root}")

    required = [
        root / "README.md",
        root / "DESIGN.md",
        root / "PROTOCOL.md",
        root / "RESULTS.md",
        root / "DECISION.md",
        root / "config" / "protocol.json",
        root / "config" / "development_extension.json",
        root / "config" / "decode_selection.json",
        root / "scripts" / "decoder_model.py",
        root / "scripts" / "study_common.py",
        root / "scripts" / "test_components.py",
        root / "scripts" / "prepare_data.py",
        root / "scripts" / "train_decoder.py",
        root / "scripts" / "activate_extension.py",
        root / "scripts" / "select_decode.py",
        root / "scripts" / "export_decoder.py",
        root / "scripts" / "evaluate_decoder.py",
        root / "scripts" / "report_results.py",
        root / "scripts" / "verify_study.py",
        root / "scripts" / "run_study.sh",
        root / "inputs" / "manifest.json",
        root / "prepared" / "tokens.npy",
        root / "prepared" / "split_indices.npz",
        root / "prepared" / "dataset_metadata.json",
        root / "checkpoints" / "best.pt",
        root / "checkpoints" / "last.pt",
        root / "checkpoints" / "decoder_inference.pt",
        root / "state" / "PREPARED.json",
        root / "state" / "TRAINING_COMPLETE.json",
        root / "state" / "DEVELOPMENT_EXTENSION_DECISION.json",
        root / "state" / "DECODE_SELECTION.json",
        root / "state" / "DECODER_EXPORT.json",
        root / "state" / "EVALUATION_COMPLETE.json",
        root / "state" / "COMPLETE.json",
        root / "state" / "SOURCE_SHA256SUMS",
        root / "outputs" / "SHA256SUMS",
        root / "outputs" / "study_summary.json",
        root / "outputs" / "evaluation_summary.json",
        root / "outputs" / "decoder_decision.json",
        root / "outputs" / "raw" / "validation_reconstructions.parquet",
        root / "outputs" / "raw" / "reencoded_unique_molecules.npz",
        root / "outputs" / "tables" / "training_curve.csv",
        root / "outputs" / "tables" / "metrics_by_control.csv",
        root / "outputs" / "tables" / "teacher_forced_by_control.csv",
        root / "outputs" / "tables" / "go_no_go_gates.csv",
        root / "outputs" / "examples" / "reconstruction_examples.csv",
        root / "outputs" / "figures" / "training_curves.png",
        root / "outputs" / "figures" / "training_curves.svg",
        root / "outputs" / "figures" / "condition_controls.png",
        root / "outputs" / "figures" / "condition_controls.svg",
        root / "outputs" / "figures" / "latent_consistency.png",
        root / "outputs" / "figures" / "latent_consistency.svg",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")

    config = json.loads(
        (root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (root / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    paths, input_hashes = load_validate_manifest(repo_root, root, manifest)
    if (
        not config["scientific_boundary"]["frozen_encoder"]
        or config["scientific_boundary"]["embedding_space"] != "released_hybrid_w3"
        or int(config["scientific_boundary"]["test_rows"]) != 0
        or bool(config["scientific_boundary"]["endpoint_labels_used"])
        or bool(config["scientific_boundary"]["perturbation_generation"])
    ):
        raise RuntimeError("Scientific boundary differs from frozen protocol")

    prepared = json.loads(
        (root / "state" / "PREPARED.json").read_text(encoding="utf-8")
    )
    if prepared.get("status") != "complete":
        raise RuntimeError("Preparation seal is incomplete")
    for record in prepared["outputs"].values():
        artifact = root / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"Prepared artifact hash mismatch: {artifact}")
    if (
        int(prepared["summary"]["rows"]) != 1_000_000
        or int(prepared["summary"]["train_rows"]) != 980_000
        or int(prepared["summary"]["dev_rows"]) != 20_000
        or int(prepared["summary"]["stereochemical_smiles"]) != 0
    ):
        raise RuntimeError("Prepared population differs from protocol")

    split = np.load(root / "prepared" / "split_indices.npz")
    train_index = split["train_indices"]
    dev_index = split["dev_indices"]
    if (
        len(train_index) != 980_000
        or len(dev_index) != 20_000
        or len(np.intersect1d(train_index, dev_index)) != 0
        or len(np.union1d(train_index, dev_index)) != 1_000_000
    ):
        raise RuntimeError("Decoder train/dev indices are invalid")

    training = json.loads(
        (root / "state" / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    if (
        training.get("status") != "complete"
        or training.get("mode") != "full"
        or int(training["active_train_rows"]) != 980_000
        or int(training["dev_rows"]) != 20_000
        or int(training["frozen_gmolai_parameters_in_optimizer"]) != 0
        or training["embedding_space"] != "released_hybrid_w3"
        or training["input_sha256"] != input_hashes
    ):
        raise RuntimeError("Training seal is inconsistent")
    extension = json.loads(
        (
            root / "state" / "DEVELOPMENT_EXTENSION_DECISION.json"
        ).read_text(encoding="utf-8")
    )
    if (
        extension.get("status") not in {"activated", "not_activated"}
        or bool(extension["final_validation_generation_started"])
    ):
        raise RuntimeError("Development-duration decision is invalid")
    if extension["status"] == "activated":
        if (
            int(config["training"]["maximum_epochs"]) != 24
            or int(config["training"]["cosine_schedule_epochs"]) != 12
            or not (12 < int(training["epochs_completed"]) <= 24)
        ):
            raise RuntimeError("Activated duration extension is inconsistent")
        for record_name in (
            "training_summary",
            "best_checkpoint",
            "training_curve",
        ):
            record_path = root / extension["archive"][record_name]
            digest_key = f"{record_name}_sha256"
            if (
                not record_path.is_file()
                or sha256_file(record_path)
                != extension["archive"][digest_key]
            ):
                raise RuntimeError(
                    f"Epoch-12 archive mismatch: {record_name}"
                )
    elif (
        int(
            config["training"].get("baseline_epochs", 12)
        )
        != 12
        or int(training["epochs_completed"]) != 12
    ):
        raise RuntimeError("Non-activated duration decision is inconsistent")

    checkpoint_path = root / training["best_checkpoint"]
    if sha256_file(checkpoint_path) != training["best_checkpoint_sha256"]:
        raise RuntimeError("Decoder checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    allowed_checkpoint_configs = {
        sha256_file(root / "config" / "protocol.json")
    }
    if extension["status"] == "activated":
        allowed_checkpoint_configs.add(extension["baseline_config_sha256"])
        if extension.get("extended_config_sha256"):
            allowed_checkpoint_configs.add(
                extension["extended_config_sha256"]
            )
    if (
        checkpoint.get("artifact_type") != "decoder_only"
        or checkpoint["frozen_input_sha256"] != input_hashes
        or checkpoint["config_sha256"] not in allowed_checkpoint_configs
        or int(checkpoint["model_config"]["condition_dimensions"]) != 384
    ):
        raise RuntimeError("Checkpoint is not a bound decoder-only artifact")
    forbidden_names = ("gmolai", "encoder", "calibrator", "checkpoint.")
    if any(
        any(token in str(name).lower() for token in forbidden_names)
        for name in checkpoint["model_state_dict"]
    ):
        raise RuntimeError("Frozen-model parameters entered decoder state")
    model = ConditionalSmilesTransformer(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if decoder_parameter_count(model) != int(training["decoder_parameters"]):
        raise RuntimeError("Decoder parameter count mismatch")
    export_seal = json.loads(
        (root / "state" / "DECODER_EXPORT.json").read_text(
            encoding="utf-8"
        )
    )
    export_path = root / export_seal["artifact"]
    if (
        export_seal.get("status") != "complete"
        or not export_path.is_file()
        or sha256_file(export_path) != export_seal["sha256"]
        or bool(export_seal["contains_optimizer_state"])
        or bool(export_seal["contains_gmolai_parameters"])
        or export_seal["source_training_checkpoint_sha256"]
        != training["best_checkpoint_sha256"]
    ):
        raise RuntimeError("Inference-only decoder export is inconsistent")
    exported = torch.load(
        export_path, map_location="cpu", weights_only=False
    )
    if (
        exported.get("artifact_type")
        != "conditional_smiles_decoder_inference"
        or "optimizer_state_dict" in exported
        or exported["frozen_input_sha256"] != input_hashes
        or exported["source_training_checkpoint_sha256"]
        != training["best_checkpoint_sha256"]
        or set(exported["model_state_dict"])
        != set(checkpoint["model_state_dict"])
        or any(
            not torch.equal(
                exported["model_state_dict"][name],
                checkpoint["model_state_dict"][name],
            )
            for name in checkpoint["model_state_dict"]
        )
    ):
        raise RuntimeError("Inference export differs from selected decoder")
    del model, checkpoint, exported

    decode_path = root / "state" / "DECODE_SELECTION.json"
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    if (
        decode.get("status") != "complete"
        or not bool(decode["selected_before_final_validation"])
        or decode["selected_decode_method"]
        not in {"greedy", "beam_w4_lp06"}
        or decode["decoder_checkpoint_sha256"]
        != training["best_checkpoint_sha256"]
        or int(decode["development_panel_rows"]) != 2048
        or int(decode["test_rows"]) != 0
        or bool(decode["endpoint_labels_used"])
    ):
        raise RuntimeError("Frozen decode selection is inconsistent")
    evaluation = json.loads(
        (root / "state" / "EVALUATION_COMPLETE.json").read_text(encoding="utf-8")
    )
    if (
        evaluation.get("status") != "complete"
        or int(evaluation["validation_rows"]) != 50_000
        or int(evaluation["generation_rows_per_control"]) != 10_000
        or int(evaluation["test_rows"]) != 0
    ):
        raise RuntimeError("Evaluation seal differs from protocol")
    raw_path = root / "outputs" / "raw" / "validation_reconstructions.parquet"
    if sha256_file(raw_path) != evaluation["raw_reconstructions_sha256"]:
        raise RuntimeError("Raw evaluation hash mismatch")
    raw = pd.read_parquet(raw_path)
    if set(raw["control"]) != CONTROLS or len(raw) != 40_000:
        raise RuntimeError("Raw condition-control table is incomplete")
    counts = raw.groupby("control").size()
    if not (counts == 10_000).all():
        raise RuntimeError("Control row counts are unequal")
    query_sets = {
        control: set(raw.loc[raw["control"] == control, "target_hash"].astype(str))
        for control in CONTROLS
    }
    reference = query_sets["correct_embedding"]
    if len(reference) != 10_000 or any(values != reference for values in query_sets.values()):
        raise RuntimeError("Target identities differ across controls")
    correct = raw.loc[raw["control"] == "correct_embedding"]
    shuffled = raw.loc[raw["control"] == "shuffled_embedding"]
    zero = raw.loc[raw["control"] == "zero_embedding"]
    wrong = raw.loc[raw["control"] == "wrong_molecule_embedding"]
    if not (
        correct["validation_index"].to_numpy()
        == correct["condition_source_index"].to_numpy()
    ).all():
        raise RuntimeError("Correct conditions are misassigned")
    if (
        (shuffled["validation_index"] == shuffled["condition_source_index"]).any()
        or (wrong["validation_index"] == wrong["condition_source_index"]).any()
        or not (zero["condition_source_index"] == -1).all()
    ):
        raise RuntimeError("Wrong-condition controls contain self conditions")
    binary_columns = [
        "rdkit_valid_smiles",
        "gmolai_policy_accepted",
        "exact_smiles_string_reconstruction",
        "exact_canonical_reconstruction",
        "molecular_identity_recovery",
        "scaffold_recovery",
    ]
    if any(not set(raw[column].dropna().unique()).issubset({0.0, 1.0}) for column in binary_columns):
        raise RuntimeError("A binary reconstruction metric is malformed")
    if (
        raw["molecular_identity_recovery"] > raw["gmolai_policy_accepted"]
    ).any():
        raise RuntimeError("Identity recovery was credited to rejected chemistry")
    if not (
        raw["exact_canonical_reconstruction"].to_numpy()
        == raw["molecular_identity_recovery"].to_numpy()
    ).all():
        raise RuntimeError("Canonical and hash identity recovery disagree")

    metrics = pd.read_csv(root / "outputs" / "tables" / "metrics_by_control.csv")
    teacher = pd.read_csv(
        root / "outputs" / "tables" / "teacher_forced_by_control.csv"
    )
    if (
        set(metrics["control"]) != CONTROLS
        or set(teacher["control"]) != CONTROLS
        or not (metrics["rows"] == 10_000).all()
        or not (teacher["rows"] == 10_000).all()
    ):
        raise RuntimeError("Machine-readable control summaries are incomplete")
    summarized_metrics = [
        "rdkit_valid_smiles",
        "gmolai_policy_accepted",
        "exact_smiles_string_reconstruction",
        "exact_canonical_reconstruction",
        "molecular_identity_recovery",
        "scaffold_recovery",
        "morgan_similarity_to_target",
        "condition_source_identity_recovery",
        "reencoded_cosine_to_target",
        "reencoded_cosine_to_supplied_condition",
        "reencoded_rmse_to_target",
        "reencoded_rmse_to_supplied_condition",
        "reencoded_relative_l2_to_target",
        "reencoded_relative_l2_to_supplied_condition",
    ]
    for control in CONTROLS:
        source = raw.loc[raw["control"] == control]
        reported = metrics.loc[metrics["control"] == control].iloc[0]
        for name in summarized_metrics:
            finite = source[name].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            observed = float(finite.mean()) if len(finite) else np.nan
            expected = float(reported[f"{name}_mean"])
            expected_n = int(reported[f"{name}_n"])
            if expected_n != len(finite) or not (
                (np.isnan(observed) and np.isnan(expected))
                or np.isclose(observed, expected, rtol=1e-12, atol=1e-12)
            ):
                raise RuntimeError(
                    f"Summary/raw mismatch for {control}/{name}"
                )

    gates = pd.read_csv(root / "outputs" / "tables" / "go_no_go_gates.csv")
    if len(gates) != 9:
        raise RuntimeError("Predeclared decision gate set is incomplete")
    recomputed = (
        np.isfinite(gates["observed"].to_numpy(dtype=float))
        & (
            gates["observed"].to_numpy(dtype=float)
            >= gates["threshold"].to_numpy(dtype=float)
        )
    )
    if not np.array_equal(recomputed, gates["passed"].astype(bool).to_numpy()):
        raise RuntimeError("Decision gate truth values are inconsistent")
    decision = json.loads(
        (root / "outputs" / "decoder_decision.json").read_text(encoding="utf-8")
    )
    expected_decision = "GO" if bool(recomputed.all()) else "NO-GO"
    gate_status = dict(
        zip(gates["gate"].astype(str), gates["passed"].astype(bool))
    )
    expected_condition_dependence = all(
        gate_status[name]
        for name in (
            "correct_identity_advantage",
            "hard_wrong_source_identity",
            "shuffled_source_identity",
        )
    )
    expected_fidelity = all(
        gate_status[name]
        for name in (
            "correct_valid_smiles",
            "correct_identity",
            "correct_scaffold",
            "correct_morgan",
            "correct_reencoded_cosine",
        )
    )
    if (
        decision["decision"] != expected_decision
        or bool(decision["all_predeclared_gates_pass"]) != bool(recomputed.all())
        or bool(decision["condition_dependence_demonstrated"])
        != expected_condition_dependence
        or bool(decision["fidelity_thresholds_pass"])
        != expected_fidelity
    ):
        raise RuntimeError("GO/NO-GO decision is inconsistent")

    train_molecules = pd.read_parquet(
        paths["train_molecules"], columns=["molecule_hash", "scaffold"]
    )
    validation_molecules = pd.read_parquet(
        paths["validation_molecules"], columns=["molecule_hash", "scaffold"]
    )
    group_keys = scaffold_group_keys(train_molecules)
    dev_groups = {group_keys[int(index)] for index in dev_index}
    if any(group_keys[int(index)] in dev_groups for index in train_index):
        raise RuntimeError("Decoder train/dev scaffold-group leakage")
    if set(train_molecules["molecule_hash"].astype(str)).intersection(
        validation_molecules["molecule_hash"].astype(str)
    ):
        raise RuntimeError("Train/validation molecule leakage")
    train_scaffold = set(
        train_molecules.loc[
            train_molecules["scaffold"].fillna("").astype(str) != "", "scaffold"
        ].astype(str)
    )
    validation_scaffold = set(
        validation_molecules.loc[
            validation_molecules["scaffold"].fillna("").astype(str) != "", "scaffold"
        ].astype(str)
    )
    if train_scaffold.intersection(validation_scaffold):
        raise RuntimeError("Train/validation scaffold leakage")

    outputs = root / "outputs"
    ledger = outputs / "SHA256SUMS"
    entries = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        artifact = (outputs / relative).resolve()
        if outputs.resolve() not in artifact.parents:
            raise RuntimeError(f"Ledger path escapes outputs: {relative}")
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise RuntimeError(f"Output ledger mismatch: {relative}")
        entries += 1
    if entries < 15:
        raise RuntimeError(f"Suspiciously short output ledger: {entries}")

    source_ledger = root / "state" / "SOURCE_SHA256SUMS"
    source_entries = 0
    for line in source_ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        artifact = (root / relative).resolve()
        if artifact != root and root not in artifact.parents:
            raise RuntimeError(f"Source ledger path escapes Step 2: {relative}")
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise RuntimeError(f"Source ledger mismatch: {relative}")
        source_entries += 1
    if source_entries < 15:
        raise RuntimeError(
            f"Suspiciously short source ledger: {source_entries}"
        )

    complete = json.loads(
        (root / "state" / "COMPLETE.json").read_text(encoding="utf-8")
    )
    if (
        complete.get("status") != "complete"
        or complete["decision"] != expected_decision
        or not bool(complete["single_gpu"])
        or int(complete["train_rows"]) != 980_000
        or int(complete["validation_rows"]) != 50_000
        or int(complete["generation_rows_per_control"]) != 10_000
        or set(complete["controls"]) != CONTROLS
        or int(complete["test_rows"]) != 0
        or bool(complete["endpoint_labels_used"])
        or bool(complete["latent_perturbation_performed"])
        or complete["development_extension_status"]
        != extension["status"]
        or complete["decode_method"]
        != decode["selected_decode_method"]
        or int(complete["frozen_gmolai_parameters_in_optimizer"]) != 0
        or complete["decoder_conditioning_representation"] != "released_hybrid_w3"
        or complete["decoder_inference_artifact_sha256"]
        != export_seal["sha256"]
        or complete["input_sha256"] != input_hashes
    ):
        raise RuntimeError("Final completion seal is inconsistent")
    if (
        sha256_file(root / "RESULTS.md") != complete["results_sha256"]
        or sha256_file(root / "DECISION.md") != complete["decision_sha256"]
        or sha256_file(ledger) != complete["output_ledger_sha256"]
        or sha256_file(source_ledger)
        != complete["source_ledger_sha256"]
    ):
        raise RuntimeError("Final report hashes differ from completion seal")

    print(
        json.dumps(
            {
                "status": "verified",
                "decision": expected_decision,
                "decoder_parameters": int(training["decoder_parameters"]),
                "train_rows": 980_000,
                "train_dev_rows": 20_000,
                "validation_rows": 50_000,
                "generation_rows_per_control": 10_000,
                "controls": sorted(CONTROLS),
                "test_rows": 0,
                "endpoint_labels_used": False,
                "latent_perturbation_performed": False,
                "ledger_entries": entries,
                "source_ledger_entries": source_entries,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
