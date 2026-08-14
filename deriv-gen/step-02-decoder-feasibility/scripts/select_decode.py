#!/usr/bin/env python3
"""Select greedy or beam decoding on the train-partition development panel."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from decoder_model import ConditionalSmilesTransformer
from gmolai_retrain.chem import Rejection, canonicalize
from study_common import (
    atomic_write_json,
    decode_tokens,
    derangement,
    load_validate_manifest,
    released_train_embeddings,
    sha256_file,
    stable_digest,
)

RDLogger.DisableLog("rdApp.*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_and_identity(
    decoded: list[str],
    targets: pd.DataFrame,
    resolved: dict[str, Any],
) -> tuple[float, float, float]:
    policy = resolved["data"]["canonicalization"]
    valid = 0
    accepted = 0
    identity = 0
    for value, target in zip(decoded, targets.itertuples(index=False)):
        if not value:
            continue
        valid += int(Chem.MolFromSmiles(value) is not None)
        result = canonicalize(
            value,
            isomeric_smiles=bool(policy["isomeric_smiles"]),
            fragment_policy=str(policy["fragment_policy"]),
            allowed_elements={
                str(item) for item in policy["allowed_elements"]
            },
            min_atoms=int(policy["min_atoms"]),
            max_atoms=int(policy["max_atoms"]),
            buckets=int(resolved["data"]["hash_buckets"]),
            split_cfg=resolved["data"]["split"],
        )
        if isinstance(result, Rejection):
            continue
        accepted += 1
        identity += int(result.molecule_hash == str(target.molecule_hash))
    count = max(len(decoded), 1)
    return valid / count, accepted / count, identity / count


def generate(
    model: ConditionalSmilesTransformer,
    conditions: torch.Tensor,
    *,
    method: str,
    maximum_steps: int,
    batch_size: int,
) -> list[str]:
    result: list[str] = []
    for offset in range(0, len(conditions), batch_size):
        current = conditions[offset : offset + batch_size]
        if method == "greedy":
            tokens = model.generate(
                current, maximum_steps=maximum_steps
            )
        elif method == "beam_w4_lp06":
            tokens = model.generate_beam(
                current,
                maximum_steps=maximum_steps,
                beam_width=4,
                length_penalty=0.6,
            )
        else:
            raise ValueError(method)
        for row in tokens.cpu().numpy():
            value, error = decode_tokens(row)
            result.append(value if not error else "")
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
    root = args.step_root.resolve()
    output_path = root / "state" / "DECODE_SELECTION.json"
    if output_path.exists():
        print(output_path.read_text(encoding="utf-8"))
        return
    if (root / "state" / "EVALUATION_COMPLETE.json").exists():
        raise RuntimeError("Refusing decode selection after final evaluation")
    policy_path = root / "config" / "decode_selection.json"
    config_path = root / "config" / "protocol.json"
    manifest_path = root / "inputs" / "manifest.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths, input_hashes = load_validate_manifest(
        repo_root, root, manifest
    )
    training = json.loads(
        (root / "state" / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    checkpoint_path = root / training["best_checkpoint"]
    if sha256_file(checkpoint_path) != training["best_checkpoint_sha256"]:
        raise RuntimeError("Frozen decoder checkpoint changed")
    prepared = json.loads(
        (root / "state" / "PREPARED.json").read_text(encoding="utf-8")
    )
    split_path = root / prepared["outputs"]["splits"]["path"]
    if sha256_file(split_path) != prepared["outputs"]["splits"]["sha256"]:
        raise RuntimeError("Prepared split changed")
    split = np.load(split_path)
    dev_indices = split["dev_indices"].astype(np.int64)
    population = pd.read_parquet(paths["train_molecules"])
    panel_count = int(policy["development_panel"]["rows"])
    ordered = sorted(
        range(len(dev_indices)),
        key=lambda position: stable_digest(
            int(config["seed"]),
            "dev-generation-panel",
            population.iloc[int(dev_indices[position])]["molecule_hash"],
        ),
    )
    panel_indices = dev_indices[
        np.asarray(ordered[:panel_count], dtype=np.int64)
    ]
    if panel_count != int(config["training"]["train_dev_generation_rows"]):
        raise RuntimeError("Decode-selection panel differs from training panel")

    raw = torch.load(
        paths["train_raw_embeddings"], map_location="cpu", weights_only=False
    )
    calibrator = torch.load(
        paths["calibrator"], map_location="cpu", weights_only=False
    )
    released = released_train_embeddings(raw, calibrator)
    del raw, calibrator
    device = torch.device("cuda:0")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = ConditionalSmilesTransformer(
        checkpoint["model_config"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    mapping = derangement(
        panel_count, 78_131, "train-dev-generation"
    )
    source_indices = {
        "correct_embedding": panel_indices,
        "shuffled_embedding": panel_indices[mapping],
        "zero_embedding": None,
    }
    resolved = json.loads(
        paths["resolved_config"].read_text(encoding="utf-8")
    )
    targets = population.iloc[panel_indices].reset_index(drop=True)
    source_targets = {
        "correct_embedding": targets,
        "shuffled_embedding": population.iloc[
            panel_indices[mapping]
        ].reset_index(drop=True),
        "zero_embedding": targets,
    }
    records: list[dict[str, Any]] = []
    for method in ("greedy", "beam_w4_lp06"):
        for control, indices in source_indices.items():
            if indices is None:
                conditions = torch.zeros(
                    (panel_count, 384),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                conditions = torch.from_numpy(
                    released[indices]
                ).to(device)
            decoded = generate(
                model,
                conditions,
                method=method,
                maximum_steps=int(
                    config["data"]["maximum_smiles_bytes"]
                ),
                batch_size=(
                    512 if method == "greedy" else 128
                ),
            )
            valid, accepted, target_identity = valid_and_identity(
                decoded, targets, resolved
            )
            _, _, source_identity = valid_and_identity(
                decoded, source_targets[control], resolved
            )
            records.append(
                {
                    "method": method,
                    "control": control,
                    "rows": panel_count,
                    "valid_smiles": valid,
                    "gmolai_policy_accepted": accepted,
                    "target_identity": target_identity,
                    "condition_source_identity": (
                        source_identity
                        if control != "zero_embedding"
                        else None
                    ),
                }
            )
            print(
                f"{method} {control}: valid={valid:.4f} "
                f"target={target_identity:.4f} source={source_identity:.4f}",
                flush=True,
            )
    frame = pd.DataFrame(records)
    limits = policy["select_beam_only_if_all"]
    greedy = frame.loc[frame["method"] == "greedy"].set_index("control")
    beam = frame.loc[frame["method"] == "beam_w4_lp06"].set_index("control")
    identity_gain = float(
        beam.loc["correct_embedding", "target_identity"]
        - greedy.loc["correct_embedding", "target_identity"]
    )
    valid_drop = float(
        greedy.loc["correct_embedding", "valid_smiles"]
        - beam.loc["correct_embedding", "valid_smiles"]
    )
    best_control = float(
        beam.loc[
            ["shuffled_embedding", "zero_embedding"],
            "target_identity",
        ].max()
    )
    source_drop = float(
        beam.loc["correct_embedding", "condition_source_identity"]
        - beam.loc["shuffled_embedding", "condition_source_identity"]
    )
    checks = {
        "correct_identity_gain": bool(
            identity_gain
            >= float(
                limits[
                    "correct_identity_gain_over_greedy_minimum"
                ]
            )
        ),
        "valid_smiles_drop": bool(
            valid_drop
            <= float(
                limits["valid_smiles_drop_vs_greedy_maximum"]
            )
        ),
        "beam_wrong_target_identity": bool(
            best_control
            <= float(
                limits[
                    "best_shuffled_or_zero_target_identity_maximum"
                ]
            )
        ),
        "beam_shuffled_source_identity": bool(
            source_drop
            <= float(
                limits[
                    "shuffled_condition_source_identity_within_correct_maximum_drop"
                ]
            )
        ),
    }
    selected = "beam_w4_lp06" if all(checks.values()) else "greedy"
    decision = {
        "schema_version": 1,
        "status": "complete",
        "selected_decode_method": selected,
        "selected_before_final_validation": True,
        "selected_at": utc_now(),
        "policy_sha256": sha256_file(policy_path),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "decoder_checkpoint_sha256": sha256_file(checkpoint_path),
        "development_panel_rows": panel_count,
        "metrics": records,
        "observed": {
            "beam_correct_identity_gain": identity_gain,
            "beam_correct_valid_smiles_drop": valid_drop,
            "beam_best_shuffled_or_zero_target_identity": best_control,
            "beam_correct_minus_shuffled_source_identity": source_drop,
        },
        "checks": checks,
        "frozen_input_sha256": input_hashes,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(output_path, decision, root)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
