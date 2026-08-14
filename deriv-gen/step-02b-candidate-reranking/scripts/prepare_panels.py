#!/usr/bin/env python3
"""Freeze train-development and fresh validation panels plus controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    STEP_ROOT,
    atomic_write_csv,
    atomic_write_json,
    derangement,
    deterministic_panel_indices,
    load_json,
    protocol,
    released_train_rows,
    require_one_gpu,
    sha256_file,
    stable_digest,
    topk_l2,
    utc_now,
    validate_manifest,
    validation_embeddings,
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
    output_state = root / "state" / "PANELS_PREPARED.json"
    if output_state.exists():
        print(output_state.read_text(encoding="utf-8"))
        return
    registered_path = root / "state" / "REGISTERED.json"
    if not registered_path.is_file():
        raise RuntimeError("Register Step 2b before preparing panels")
    registered = load_json(registered_path)
    cfg = protocol(root)
    paths, input_hashes = validate_manifest(repo_root, root)
    if registered["manifest_sha256"] != sha256_file(root / "inputs" / "manifest.json"):
        raise RuntimeError("Registration/manifest mismatch")
    device = require_one_gpu()
    seed = int(cfg["seed"])

    train_molecules = pd.read_parquet(paths["train_molecules"])
    split = np.load(paths["step2_split_indices"])
    dev_indices = split["dev_indices"].astype(np.int64)
    if len(dev_indices) != 20_000:
        raise RuntimeError(f"Unexpected decoder-development population: {len(dev_indices)}")
    step2_protocol = load_json(
        repo_root / "deriv-gen" / "step-02-decoder-feasibility" / "config" / "protocol.json"
    )
    step2_seed = int(step2_protocol["seed"])
    dev_count = int(cfg["panels"]["development_rows"])
    ordered = sorted(
        range(len(dev_indices)),
        key=lambda position: stable_digest(
            step2_seed,
            "dev-generation-panel",
            train_molecules.iloc[int(dev_indices[position])]["molecule_hash"],
        ),
    )
    dev_panel_indices = dev_indices[np.asarray(ordered[:dev_count], dtype=np.int64)]
    dev_bank = released_train_rows(
        paths["train_raw_embeddings"], paths["gmolai_calibrator"], dev_indices
    )
    dev_position_by_global = {int(value): pos for pos, value in enumerate(dev_indices)}
    dev_panel_positions = np.asarray(
        [dev_position_by_global[int(value)] for value in dev_panel_indices], dtype=np.int64
    )
    dev_conditions = dev_bank[dev_panel_positions]
    dev_shuffle_positions = derangement(dev_count, seed, "step2b-dev-shuffled")
    dev_shuffled_sources = dev_panel_indices[dev_shuffle_positions]
    dev_wrong_positions, dev_wrong_distances = topk_l2(
        dev_conditions,
        dev_bank,
        k=1,
        device=device,
        batch_size=512,
        exclude_indices=dev_panel_positions,
    )
    dev_wrong_sources = dev_indices[dev_wrong_positions[:, 0]]
    if np.any(dev_wrong_sources == dev_panel_indices):
        raise RuntimeError("Development nearest-wrong control contains self")
    dev_panel = pd.DataFrame(
        {
            "query_position": np.arange(dev_count, dtype=np.int64),
            "target_index": dev_panel_indices,
            "target_hash": train_molecules.iloc[dev_panel_indices]["molecule_hash"].astype(str).to_numpy(),
            "correct_source_index": dev_panel_indices,
            "shuffled_source_index": dev_shuffled_sources,
            "nearest_wrong_source_index": dev_wrong_sources,
            "nearest_wrong_distance": dev_wrong_distances[:, 0],
        }
    )

    validation_payload = torch.load(
        paths["validation_embeddings"], map_location="cpu", weights_only=False
    )
    validation_matrix = validation_embeddings(validation_payload)
    validation_molecules = pd.read_parquet(paths["validation_molecules"])
    hashes = validation_molecules["molecule_hash"].astype(str).tolist()
    if hashes != [str(value) for value in validation_payload["molecule_hashes"]]:
        raise RuntimeError("Validation molecule/embedding order mismatch")
    original = pd.read_csv(paths["step2_original_final_panel"])
    original_indices = set(original["validation_index"].astype(int).tolist())
    if len(original_indices) != 10_000:
        raise RuntimeError("Original Step-2 generation panel identity changed")
    remaining = np.asarray(
        [index for index in range(len(hashes)) if index not in original_indices],
        dtype=np.int64,
    )
    final_count = int(cfg["panels"]["final_rows"])
    chosen_positions = deterministic_panel_indices(
        [hashes[int(index)] for index in remaining],
        final_count,
        seed,
        "step2b-fresh-final-validation",
    )
    final_indices = remaining[chosen_positions]
    if set(final_indices.tolist()).intersection(original_indices):
        raise RuntimeError("Fresh Step-2b panel overlaps original Step-2 panel")
    final_conditions = validation_matrix[final_indices]
    final_shuffle_positions = derangement(final_count, seed, "step2b-final-shuffled")
    final_shuffled_sources = final_indices[final_shuffle_positions]
    final_wrong_indices, final_wrong_distances = topk_l2(
        final_conditions,
        validation_matrix,
        k=1,
        device=device,
        batch_size=512,
        exclude_indices=final_indices,
    )
    final_wrong_sources = final_wrong_indices[:, 0]
    if np.any(final_wrong_sources == final_indices):
        raise RuntimeError("Final nearest-wrong control contains self")
    final_panel = pd.DataFrame(
        {
            "query_position": np.arange(final_count, dtype=np.int64),
            "target_index": final_indices,
            "target_hash": validation_molecules.iloc[final_indices]["molecule_hash"].astype(str).to_numpy(),
            "correct_source_index": final_indices,
            "shuffled_source_index": final_shuffled_sources,
            "nearest_wrong_source_index": final_wrong_sources,
            "nearest_wrong_distance": final_wrong_distances[:, 0],
        }
    )

    dev_path = root / "prepared" / "development_panel.csv"
    final_path = root / "prepared" / "fresh_validation_panel.csv"
    atomic_write_csv(dev_path, dev_panel, root)
    atomic_write_csv(final_path, final_panel, root)
    metadata = {
        "schema_version": 1,
        "prepared_at": utc_now(),
        "development": {
            "partition": "train",
            "rows": len(dev_panel),
            "decoder_dev_bank_rows": len(dev_indices),
            "panel_sha256": sha256_file(dev_path),
            "target_unique": int(dev_panel["target_index"].nunique()),
            "nearest_wrong_mean_l2": float(dev_panel["nearest_wrong_distance"].mean()),
        },
        "final": {
            "partition": "validation",
            "rows": len(final_panel),
            "panel_sha256": sha256_file(final_path),
            "target_unique": int(final_panel["target_index"].nunique()),
            "original_step2_rows_excluded": len(original_indices),
            "overlap_with_original_step2_panel": 0,
            "post_hoc": False,
            "nearest_wrong_mean_l2": float(final_panel["nearest_wrong_distance"].mean()),
        },
        "test_rows": 0,
        "endpoint_labels_used": False,
        "input_sha256": input_hashes,
    }
    metadata_path = root / "prepared" / "panel_metadata.json"
    atomic_write_json(metadata_path, metadata, root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": utc_now(),
        "development_panel_sha256": sha256_file(dev_path),
        "fresh_validation_panel_sha256": sha256_file(final_path),
        "metadata_sha256": sha256_file(metadata_path),
        "fresh_validation_generation_started": False,
        "overlap_with_original_step2_panel": 0,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(output_state, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
