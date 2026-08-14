#!/usr/bin/env python3
"""Final held-out validation and condition-use controls for the frozen decoder."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from decoder_model import ConditionalSmilesTransformer
from gmolai_retrain.chem import Rejection, canonicalize
from gmolai_retrain.fast_inference import (
    build_smiles_encoder,
    implementation_metadata,
)
from study_common import (
    PAD_TOKEN,
    atomic_numpy_savez,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    bootstrap_mean_ci,
    cosine_rows,
    decode_tokens,
    derangement,
    deterministic_panel_indices,
    load_validate_manifest,
    make_fingerprints,
    sha256_file,
    stable_digest,
    tanimoto,
    token_matrix,
    topk_l2,
    validation_embeddings,
)

RDLogger.DisableLog("rdApp.*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_packaged_inference(repo_root: Path) -> Any:
    path = repo_root / "inference" / "generate_embeddings.py"
    spec = importlib.util.spec_from_file_location(
        "gmolai_released_inference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load released inference entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_training(step_root: Path) -> tuple[dict[str, Any], Path]:
    seal_path = step_root / "state" / "TRAINING_COMPLETE.json"
    if not seal_path.is_file():
        raise RuntimeError("Decoder training is not complete")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    checkpoint = step_root / seal["best_checkpoint"]
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != seal["best_checkpoint_sha256"]
    ):
        raise RuntimeError("Selected decoder checkpoint changed")
    return seal, checkpoint


def teacher_forced_nll(
    model: ConditionalSmilesTransformer,
    tokens: torch.Tensor,
    conditions: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    loss_sum = 0.0
    correct_sum = 0.0
    token_count = 0
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(tokens), batch_size):
            stop = min(offset + batch_size, len(tokens))
            sequence = tokens[offset:stop].long()
            inputs, targets = sequence[:, :-1], sequence[:, 1:]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(inputs, conditions[offset:stop])
            losses = F.cross_entropy(
                logits.transpose(1, 2).float(),
                targets,
                reduction="none",
                ignore_index=PAD_TOKEN,
            )
            mask = targets.ne(PAD_TOKEN)
            loss_sum += float((losses * mask).sum())
            correct_sum += float(
                (logits.argmax(dim=-1).eq(targets) & mask).sum()
            )
            token_count += int(mask.sum())
    return loss_sum / token_count, correct_sum / token_count


def policy_canonicalize(
    raw: str, resolved_config: dict[str, Any]
) -> tuple[Any | None, str]:
    if not raw:
        return None, "decoder_token_error"
    data = resolved_config["data"]
    policy = data["canonicalization"]
    value = canonicalize(
        raw,
        isomeric_smiles=bool(policy["isomeric_smiles"]),
        fragment_policy=str(policy["fragment_policy"]),
        allowed_elements={str(item) for item in policy["allowed_elements"]},
        min_atoms=int(policy["min_atoms"]),
        max_atoms=int(policy["max_atoms"]),
        buckets=int(data["hash_buckets"]),
        split_cfg=data["split"],
    )
    if isinstance(value, Rejection):
        return None, value.reason
    return value, ""


def generate_control(
    model: ConditionalSmilesTransformer,
    source_matrix: np.ndarray,
    *,
    batch_size: int,
    maximum_steps: int,
    device: torch.device,
    decode_method: str,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    model.eval()
    for offset in range(0, len(source_matrix), batch_size):
        stop = min(offset + batch_size, len(source_matrix))
        condition = torch.as_tensor(
            source_matrix[offset:stop],
            dtype=torch.float32,
            device=device,
        )
        if decode_method == "greedy":
            generated = model.generate(
                condition, maximum_steps=maximum_steps
            ).cpu().numpy()
        elif decode_method == "beam_w4_lp06":
            generated = model.generate_beam(
                condition,
                maximum_steps=maximum_steps,
                beam_width=4,
                length_penalty=0.6,
            ).cpu().numpy()
        else:
            raise RuntimeError(
                f"Unknown frozen decode method: {decode_method}"
            )
        result.extend(decode_tokens(row) for row in generated)
        print(
            f"  generated {stop:,}/{len(source_matrix):,}",
            flush=True,
        )
    return result


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
    step_root = args.step_root.resolve()
    config = json.loads(
        (step_root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (step_root / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    input_paths, input_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )
    training, checkpoint_path = validate_training(step_root)
    decode_path = step_root / "state" / "DECODE_SELECTION.json"
    if not decode_path.is_file():
        raise RuntimeError("Decode method is not frozen")
    decode_selection = json.loads(
        decode_path.read_text(encoding="utf-8")
    )
    if (
        decode_selection.get("status") != "complete"
        or not decode_selection.get("selected_before_final_validation")
        or decode_selection.get("decoder_checkpoint_sha256")
        != sha256_file(checkpoint_path)
        or int(decode_selection.get("test_rows", -1)) != 0
    ):
        raise RuntimeError("Decode-selection seal is inconsistent")
    decode_method = str(
        decode_selection["selected_decode_method"]
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Final decoder evaluation requires one visible GPU")
    device = torch.device("cuda:0")
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.perf_counter()
    started_at = utc_now()

    validation_payload = torch.load(
        input_paths["validation_embeddings"],
        map_location="cpu",
        weights_only=False,
    )
    matrix = validation_embeddings(validation_payload)
    molecules = pd.read_parquet(input_paths["validation_molecules"])
    expected_columns = {
        "molecule_index",
        "molecule_hash",
        "canonical_smiles",
        "scaffold",
        "heavy_atoms",
    }
    if set(molecules.columns) != expected_columns:
        raise RuntimeError(
            "Validation chemistry table contains unexpected columns"
        )
    hashes = molecules["molecule_hash"].astype(str).tolist()
    if hashes != [str(value) for value in validation_payload["molecule_hashes"]]:
        raise RuntimeError("Validation chemistry and conditions are misaligned")
    if molecules["molecule_hash"].duplicated().any():
        raise RuntimeError("Validation molecule identities are not unique")
    train_molecules = pd.read_parquet(
        input_paths["train_molecules"],
        columns=["molecule_hash", "scaffold"],
    )
    if set(train_molecules["molecule_hash"].astype(str)).intersection(hashes):
        raise RuntimeError("Train/validation molecule identity overlap")
    train_scaffolds = set(
        train_molecules.loc[
            train_molecules["scaffold"].fillna("").astype(str) != "",
            "scaffold",
        ].astype(str)
    )
    validation_scaffolds = set(
        molecules.loc[
            molecules["scaffold"].fillna("").astype(str) != "", "scaffold"
        ].astype(str)
    )
    if train_scaffolds.intersection(validation_scaffolds):
        raise RuntimeError("Train/validation nonempty scaffold overlap")
    del train_molecules, train_scaffolds, validation_scaffolds

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("artifact_type") != "decoder_only":
        raise RuntimeError("Selected checkpoint is not decoder-only")
    if checkpoint["frozen_input_sha256"] != input_hashes:
        raise RuntimeError("Decoder checkpoint frozen-input binding changed")
    decoder_model_config = checkpoint["model_config"]
    model = ConditionalSmilesTransformer(
        decoder_model_config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    validation_tokens_np = token_matrix(
        molecules["canonical_smiles"].astype(str).tolist(),
        int(config["data"]["maximum_smiles_bytes"]),
    )
    validation_tokens = torch.from_numpy(validation_tokens_np).to(device)
    validation_conditions = torch.from_numpy(matrix).to(device)
    full_nll, full_token_accuracy = teacher_forced_nll(
        model,
        validation_tokens,
        validation_conditions,
        batch_size=int(config["training"]["dev_batch_size"]),
    )
    print(
        f"all-validation teacher NLL={full_nll:.4f}, "
        f"token accuracy={full_token_accuracy:.4f}",
        flush=True,
    )

    panel_count = int(config["data"]["final_generation_rows"])
    panel_indices = deterministic_panel_indices(
        hashes, panel_count, seed, "final-validation-generation"
    )
    panel = molecules.iloc[panel_indices].reset_index(drop=True)
    panel_conditions = matrix[panel_indices]
    shuffled_positions = derangement(
        panel_count, seed, "final-shuffled-control"
    )
    shuffled_sources = panel_indices[shuffled_positions]

    print("selecting nearest non-self hard-wrong conditions", flush=True)
    hard_indices, hard_distances = topk_l2(
        panel_conditions,
        matrix,
        k=1,
        device=device,
        batch_size=int(config["evaluation"]["nearest_neighbor_batch_size"]),
        exclude_indices=panel_indices,
    )
    hard_sources = hard_indices[:, 0]
    if np.any(hard_sources == panel_indices):
        raise RuntimeError("Hard-wrong control contains self conditions")

    controls: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "correct_embedding": (panel_conditions, panel_indices),
        "shuffled_embedding": (matrix[shuffled_sources], shuffled_sources),
        "zero_embedding": (
            np.zeros_like(panel_conditions),
            np.full(panel_count, -1, dtype=np.int64),
        ),
        "wrong_molecule_embedding": (
            matrix[hard_sources],
            hard_sources,
        ),
    }
    control_source_table = pd.DataFrame(
        {
            "query_position": np.arange(panel_count),
            "validation_index": panel_indices,
            "shuffled_source_index": shuffled_sources,
            "wrong_molecule_source_index": hard_sources,
            "wrong_molecule_distance": hard_distances[:, 0],
        }
    )
    atomic_write_csv(
        step_root / "prepared" / "final_control_sources.csv",
        control_source_table,
        step_root,
    )

    generated_by_control: dict[str, list[tuple[str, str]]] = {}
    for control, (source_matrix, _) in controls.items():
        print(f"generating control={control}", flush=True)
        generated_by_control[control] = generate_control(
            model,
            source_matrix,
            batch_size=(
                int(config["evaluation"]["generation_batch_size"])
                if decode_method == "greedy"
                else int(
                    config["evaluation"][
                        "beam_generation_batch_size"
                    ]
                )
            ),
            maximum_steps=int(config["data"]["maximum_smiles_bytes"]),
            device=device,
            decode_method=decode_method,
        )

    del model, checkpoint, validation_tokens, validation_conditions
    torch.cuda.empty_cache()
    gc.collect()

    target_smiles = panel["canonical_smiles"].astype(str).tolist()
    target_hashes = panel["molecule_hash"].astype(str).tolist()
    target_scaffolds = panel["scaffold"].fillna("").astype(str).tolist()
    all_validation_smiles = molecules["canonical_smiles"].astype(str).tolist()
    all_validation_hashes = molecules["molecule_hash"].astype(str).tolist()
    all_validation_scaffolds = (
        molecules["scaffold"].fillna("").astype(str).tolist()
    )
    target_fingerprints = make_fingerprints(target_smiles)
    validation_fingerprints = make_fingerprints(all_validation_smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(config["evaluation"]["morgan_radius"]),
        fpSize=int(config["evaluation"]["morgan_bits"]),
    )
    resolved_config = json.loads(
        input_paths["resolved_config"].read_text(encoding="utf-8")
    )

    rows: list[dict[str, Any]] = []
    unique_accepted: dict[str, int] = {}
    accepted_atom_counts: list[int] = []
    for control, generated in generated_by_control.items():
        source_indices = controls[control][1]
        for position, ((decoded_raw, token_error), source_index) in enumerate(
            zip(generated, source_indices)
        ):
            target_index = int(panel_indices[position])
            source_index = int(source_index)
            parsed = Chem.MolFromSmiles(decoded_raw) if decoded_raw else None
            rdkit_valid = parsed is not None
            canonical, policy_error = policy_canonicalize(
                decoded_raw, resolved_config
            )
            accepted = canonical is not None
            decoded_canonical = canonical.smiles if accepted else ""
            decoded_hash = canonical.molecule_hash if accepted else ""
            decoded_scaffold = canonical.scaffold if accepted else ""
            decoded_fingerprint = (
                generator.GetFingerprint(
                    Chem.MolFromSmiles(decoded_canonical)
                )
                if accepted
                else None
            )
            target_fp = target_fingerprints[position]
            target_similarity = (
                tanimoto(target_fp, decoded_fingerprint)
                if accepted
                else 0.0
            )
            if source_index >= 0:
                source_hash = all_validation_hashes[source_index]
                source_smiles = all_validation_smiles[source_index]
                source_scaffold = all_validation_scaffolds[source_index]
                source_similarity = (
                    tanimoto(
                        validation_fingerprints[source_index],
                        decoded_fingerprint,
                    )
                    if accepted
                    else 0.0
                )
            else:
                source_hash = ""
                source_smiles = ""
                source_scaffold = ""
                source_similarity = math.nan
            if accepted and decoded_canonical not in unique_accepted:
                unique_accepted[decoded_canonical] = len(unique_accepted)
                accepted_atom_counts.append(int(canonical.atom_count))
            stereochemical_target = (
                ("@" in target_smiles[position])
                or ("/" in target_smiles[position])
                or ("\\" in target_smiles[position])
            )
            rows.append(
                {
                    "control": control,
                    "query_position": position,
                    "validation_index": target_index,
                    "condition_source_index": source_index,
                    "target_hash": target_hashes[position],
                    "target_smiles": target_smiles[position],
                    "target_scaffold": target_scaffolds[position],
                    "condition_source_hash": source_hash,
                    "condition_source_smiles": source_smiles,
                    "condition_source_scaffold": source_scaffold,
                    "decoded_smiles_raw": decoded_raw,
                    "token_decode_error": token_error,
                    "rdkit_valid_smiles": float(rdkit_valid),
                    "gmolai_policy_accepted": float(accepted),
                    "policy_rejection_reason": policy_error,
                    "decoded_canonical_smiles": decoded_canonical,
                    "decoded_hash": decoded_hash,
                    "decoded_scaffold": decoded_scaffold,
                    "exact_smiles_string_reconstruction": float(
                        decoded_raw == target_smiles[position]
                    ),
                    "exact_canonical_reconstruction": float(
                        accepted
                        and decoded_canonical == target_smiles[position]
                    ),
                    "molecular_identity_recovery": float(
                        decoded_hash == target_hashes[position]
                    ),
                    "scaffold_recovery": float(
                        accepted
                        and decoded_scaffold
                        == target_scaffolds[position]
                    ),
                    "morgan_similarity_to_target": target_similarity,
                    "condition_source_identity_recovery": (
                        float(decoded_hash == source_hash)
                        if source_index >= 0
                        else math.nan
                    ),
                    "condition_source_scaffold_recovery": (
                        float(
                            accepted
                            and decoded_scaffold == source_scaffold
                        )
                        if source_index >= 0
                        else math.nan
                    ),
                    "morgan_similarity_to_condition_source": (
                        source_similarity
                    ),
                    "stereochemical_target": float(
                        stereochemical_target
                    ),
                    "stereochemical_identity_recovery": (
                        float(decoded_hash == target_hashes[position])
                        if stereochemical_target
                        else math.nan
                    ),
                    "accepted_embedding_index": (
                        unique_accepted.get(decoded_canonical, -1)
                        if accepted
                        else -1
                    ),
                }
            )
        print(
            f"chemistry metrics complete for {control}", flush=True
        )
    frame = pd.DataFrame(rows)

    unique_smiles = list(unique_accepted)
    print(
        f"re-encoding {len(unique_smiles):,} unique accepted decoded molecules "
        "with frozen released gMolAI",
        flush=True,
    )
    inference = load_packaged_inference(repo_root)
    bundle = inference.load_model_bundle(
        repo_root / "inference" / "model", device
    )
    encoder = build_smiles_encoder(
        str(config["evaluation"]["reencode_backend"]).split("_gine")[0],
        bundle.model,
        bundle.coordinate_mean,
        bundle.coordinate_scale,
        device=device,
        batch_size=int(config["evaluation"]["reencode_batch_size"]),
        node_budget=int(config["evaluation"]["reencode_node_budget"]),
        workers=config["evaluation"]["reencode_workers"],
        mean_node_weight=bundle.mean_node_weight,
    )
    try:
        decoded_embeddings = encoder.encode(
            unique_smiles, atom_counts=accepted_atom_counts
        )
        reencode_metadata = implementation_metadata(encoder)
    finally:
        encoder.close()
    del encoder, bundle
    torch.cuda.empty_cache()
    gc.collect()
    if decoded_embeddings.shape != (len(unique_smiles), 384):
        raise RuntimeError("Frozen re-encoder returned unexpected shape")
    atomic_numpy_savez(
        step_root / "outputs" / "raw" / "reencoded_unique_molecules.npz",
        step_root,
        canonical_smiles=np.asarray(unique_smiles, dtype=np.str_),
        embeddings=decoded_embeddings.astype(np.float32),
    )

    latent_to_target = np.full(len(frame), np.nan, dtype=np.float64)
    latent_to_supplied = np.full(len(frame), np.nan, dtype=np.float64)
    rmse_to_target = np.full(len(frame), np.nan, dtype=np.float64)
    rmse_to_supplied = np.full(len(frame), np.nan, dtype=np.float64)
    l2_to_target = np.full(len(frame), np.nan, dtype=np.float64)
    l2_to_supplied = np.full(len(frame), np.nan, dtype=np.float64)
    relative_l2_to_target = np.full(len(frame), np.nan, dtype=np.float64)
    relative_l2_to_supplied = np.full(len(frame), np.nan, dtype=np.float64)

    valid_rows = np.flatnonzero(
        frame["accepted_embedding_index"].to_numpy(dtype=np.int64) >= 0
    )
    decoded_index = frame.loc[
        valid_rows, "accepted_embedding_index"
    ].to_numpy(dtype=np.int64)
    decoded_values = decoded_embeddings[decoded_index].astype(np.float64)
    target_index = frame.loc[
        valid_rows, "validation_index"
    ].to_numpy(dtype=np.int64)
    target_values = matrix[target_index].astype(np.float64)
    delta_target = decoded_values - target_values
    latent_to_target[valid_rows] = cosine_rows(
        decoded_values, target_values
    )
    rmse_to_target[valid_rows] = np.sqrt(
        np.mean(np.square(delta_target), axis=1)
    )
    l2_to_target[valid_rows] = np.linalg.norm(delta_target, axis=1)
    relative_l2_to_target[valid_rows] = l2_to_target[valid_rows] / np.maximum(
        np.linalg.norm(target_values, axis=1), 1e-12
    )

    supplied_source = frame.loc[
        valid_rows, "condition_source_index"
    ].to_numpy(dtype=np.int64)
    has_source = supplied_source >= 0
    sourced_rows = valid_rows[has_source]
    supplied_values = matrix[supplied_source[has_source]].astype(np.float64)
    sourced_decoded = decoded_values[has_source]
    delta_source = sourced_decoded - supplied_values
    latent_to_supplied[sourced_rows] = cosine_rows(
        sourced_decoded, supplied_values
    )
    rmse_to_supplied[sourced_rows] = np.sqrt(
        np.mean(np.square(delta_source), axis=1)
    )
    l2_to_supplied[sourced_rows] = np.linalg.norm(delta_source, axis=1)
    relative_l2_to_supplied[sourced_rows] = l2_to_supplied[
        sourced_rows
    ] / np.maximum(np.linalg.norm(supplied_values, axis=1), 1e-12)

    frame["reencoded_cosine_to_target"] = latent_to_target
    frame["reencoded_cosine_to_supplied_condition"] = latent_to_supplied
    frame["reencoded_rmse_to_target"] = rmse_to_target
    frame["reencoded_rmse_to_supplied_condition"] = rmse_to_supplied
    frame["reencoded_l2_to_target"] = l2_to_target
    frame["reencoded_l2_to_supplied_condition"] = l2_to_supplied
    frame["reencoded_relative_l2_to_target"] = relative_l2_to_target
    frame[
        "reencoded_relative_l2_to_supplied_condition"
    ] = relative_l2_to_supplied

    raw_path = (
        step_root / "outputs" / "raw" / "validation_reconstructions.parquet"
    )
    atomic_write_parquet(raw_path, frame, step_root)

    metric_columns = [
        "rdkit_valid_smiles",
        "gmolai_policy_accepted",
        "exact_smiles_string_reconstruction",
        "exact_canonical_reconstruction",
        "molecular_identity_recovery",
        "scaffold_recovery",
        "morgan_similarity_to_target",
        "condition_source_identity_recovery",
        "condition_source_scaffold_recovery",
        "morgan_similarity_to_condition_source",
        "reencoded_cosine_to_target",
        "reencoded_cosine_to_supplied_condition",
        "reencoded_rmse_to_target",
        "reencoded_rmse_to_supplied_condition",
        "reencoded_relative_l2_to_target",
        "reencoded_relative_l2_to_supplied_condition",
    ]
    summary_rows: list[dict[str, Any]] = []
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    for control, group in frame.groupby("control", sort=False):
        record: dict[str, Any] = {
            "control": control,
            "rows": len(group),
            "stereochemical_targets": int(
                group["stereochemical_target"].sum()
            ),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            mean, low, high = bootstrap_mean_ci(
                finite,
                seed=seed
                + int(stable_digest(control, metric)[:8], 16),
                resamples=resamples,
            )
            record[f"{metric}_n"] = len(finite)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
            record[f"{metric}_median"] = (
                float(np.median(finite)) if len(finite) else math.nan
            )
        summary_rows.append(record)
    metrics = pd.DataFrame(summary_rows)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "metrics_by_control.csv",
        metrics,
        step_root,
    )

    panel_teacher: list[dict[str, Any]] = []
    loaded = ConditionalSmilesTransformer(
        decoder_model_config
    ).to(device)
    loaded.load_state_dict(
        torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )["model_state_dict"],
        strict=True,
    )
    panel_tokens = torch.from_numpy(
        token_matrix(
            target_smiles,
            int(config["data"]["maximum_smiles_bytes"]),
        )
    ).to(device)
    for control, (source, _) in controls.items():
        source_tensor = torch.from_numpy(source).to(device)
        nll, accuracy = teacher_forced_nll(
            loaded,
            panel_tokens,
            source_tensor,
            batch_size=int(config["training"]["dev_batch_size"]),
        )
        panel_teacher.append(
            {
                "control": control,
                "rows": panel_count,
                "teacher_forced_nll": nll,
                "teacher_forced_token_accuracy": accuracy,
            }
        )
    del loaded, panel_tokens
    torch.cuda.empty_cache()
    teacher_frame = pd.DataFrame(panel_teacher)
    atomic_write_csv(
        step_root / "outputs" / "tables" / "teacher_forced_by_control.csv",
        teacher_frame,
        step_root,
    )

    examples = frame.pivot(
        index=[
            "query_position",
            "target_hash",
            "target_smiles",
        ],
        columns="control",
        values=[
            "decoded_smiles_raw",
            "decoded_canonical_smiles",
            "exact_canonical_reconstruction",
            "molecular_identity_recovery",
            "morgan_similarity_to_target",
            "condition_source_smiles",
            "condition_source_identity_recovery",
        ],
    )
    examples.columns = [
        f"{metric}__{control}" for metric, control in examples.columns
    ]
    examples = examples.reset_index()
    priority = examples[
        "molecular_identity_recovery__correct_embedding"
    ].to_numpy(dtype=float)
    ranked_success = sorted(
        np.flatnonzero(priority == 1.0),
        key=lambda index: stable_digest(
            seed, "example-success", examples.iloc[index]["target_hash"]
        ),
    )
    ranked_failure = sorted(
        np.flatnonzero(priority != 1.0),
        key=lambda index: stable_digest(
            seed, "example-failure", examples.iloc[index]["target_hash"]
        ),
    )
    selected_examples = [
        *ranked_success[:125],
        *ranked_failure[:125],
    ]
    if len(selected_examples) < 250:
        selected_set = set(selected_examples)
        remainder = sorted(
            (
                index
                for index in range(len(examples))
                if index not in selected_set
            ),
            key=lambda index: stable_digest(
                seed, "example-remainder", examples.iloc[index]["target_hash"]
            ),
        )
        selected_examples.extend(remainder[: 250 - len(selected_examples)])
    atomic_write_csv(
        step_root / "outputs" / "examples" / "reconstruction_examples.csv",
        examples.iloc[selected_examples].reset_index(drop=True),
        step_root,
    )

    finished_at = utc_now()
    evaluation_summary = {
        "schema_version": 1,
        "status": "complete",
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": time.perf_counter() - started,
        "validation_rows_teacher_forced": len(molecules),
        "validation_rows_generation": panel_count,
        "controls": list(controls),
        "decode_method": decode_method,
        "decode_selection_sha256": sha256_file(decode_path),
        "teacher_forced_all_validation": {
            "correct_embedding_nll": full_nll,
            "correct_embedding_token_accuracy": full_token_accuracy,
        },
        "hard_wrong": {
            "definition": config["evaluation"]["wrong_molecule_control"],
            "mean_distance": float(hard_distances[:, 0].mean()),
            "median_distance": float(np.median(hard_distances[:, 0])),
        },
        "stereochemical_targets": int(
            frame.loc[
                frame["control"] == "correct_embedding",
                "stereochemical_target",
            ].sum()
        ),
        "stereochemical_evaluation": (
            "not_estimable_no_stereochemical_targets"
            if int(
                frame.loc[
                    frame["control"] == "correct_embedding",
                    "stereochemical_target",
                ].sum()
            )
            == 0
            else "reported"
        ),
        "unique_policy_accepted_decoded_molecules": len(unique_smiles),
        "reencoder": reencode_metadata,
        "frozen_encoder_checkpoint_sha256": input_hashes[
            "packaged_checkpoint"
        ],
        "frozen_calibrator_sha256": input_hashes[
            "packaged_calibrator"
        ],
        "decoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "output_sha256": {
            "raw_reconstructions": sha256_file(raw_path),
            "metrics_by_control": sha256_file(
                step_root / "outputs" / "tables" / "metrics_by_control.csv"
            ),
            "teacher_forced_by_control": sha256_file(
                step_root
                / "outputs"
                / "tables"
                / "teacher_forced_by_control.csv"
            ),
        },
        "rejection_reasons": dict(
            Counter(
                frame.loc[
                    frame["gmolai_policy_accepted"] == 0,
                    "policy_rejection_reason",
                ].astype(str)
            )
        ),
        "input_sha256": input_hashes,
    }
    atomic_write_json(
        step_root / "outputs" / "evaluation_summary.json",
        evaluation_summary,
        step_root,
    )
    atomic_write_json(
        step_root / "state" / "EVALUATION_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "sealed_at": finished_at,
            "decoder_checkpoint_sha256": sha256_file(checkpoint_path),
            "raw_reconstructions_sha256": sha256_file(raw_path),
            "metrics_sha256": sha256_file(
                step_root / "outputs" / "tables" / "metrics_by_control.csv"
            ),
            "summary_sha256": sha256_file(
                step_root / "outputs" / "evaluation_summary.json"
            ),
            "validation_rows": len(molecules),
            "generation_rows_per_control": panel_count,
            "test_rows": 0,
        },
        step_root,
    )
    _, final_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )
    if final_hashes != input_hashes:
        raise RuntimeError("A frozen input changed during final evaluation")
    print(json.dumps(evaluation_summary, sort_keys=True))


if __name__ == "__main__":
    main()
