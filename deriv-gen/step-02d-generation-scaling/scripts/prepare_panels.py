#!/usr/bin/env python3
"""Prepare prospective development/final panels and decoder-fit novelty reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    STEP_ROOT,
    atomic_numpy_save,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    deterministic_subset,
    load_json,
    protocol,
    released_train_rows,
    resolve_manifest_inputs,
    sha256_file,
    utc_now,
    validation_embeddings,
)


def panel_frame(indices: np.ndarray, molecules: pd.DataFrame) -> pd.DataFrame:
    selected = molecules.iloc[indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "query_position": np.arange(len(indices), dtype=np.int64),
            "target_index": indices.astype(np.int64),
            "target_hash": selected["molecule_hash"].astype(str).to_numpy(),
            "seed_canonical_smiles": selected["canonical_smiles"].astype(str).to_numpy(),
            "seed_scaffold": selected["scaffold"].fillna("").astype(str).to_numpy(),
            "seed_heavy_atoms": selected["heavy_atoms"].astype(np.int16).to_numpy(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    state_path = root / "state" / "PANELS_PREPARED.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    registered_path = root / "state" / "REGISTERED.json"
    if not registered_path.is_file():
        raise RuntimeError("Register Step 2d before preparing panels")
    registered = load_json(registered_path)
    cfg = protocol(root)
    paths, hashes = resolve_manifest_inputs(repo_root, root)
    if registered["manifest_sha256"] != sha256_file(root / "inputs" / "manifest.json"):
        raise RuntimeError("Registration/manifest mismatch")

    train = pd.read_parquet(paths["train_molecules"])
    split = np.load(paths["step2_split_indices"])
    train_indices = split["train_indices"].astype(np.int64)
    dev_indices = split["dev_indices"].astype(np.int64)
    if len(train_indices) != 980_000 or len(dev_indices) != 20_000:
        raise RuntimeError("Frozen Step-2 split sizes changed")
    if set(train_indices.tolist()).intersection(dev_indices.tolist()):
        raise RuntimeError("Decoder fit and development rows overlap")

    prior_dev = pd.read_csv(paths["step2b_development_panel"])
    excluded_dev = set(prior_dev["target_index"].astype(int).tolist())
    eligible_dev = np.asarray(
        [value for value in dev_indices if int(value) not in excluded_dev], dtype=np.int64
    )
    dev_selected = deterministic_subset(
        eligible_dev,
        train.iloc[eligible_dev]["molecule_hash"].astype(str).tolist(),
        int(cfg["panels"]["development_rows"]),
        int(cfg["seed"]),
        "step2d-fresh-development",
    )
    development = panel_frame(dev_selected, train)
    development_conditions = released_train_rows(
        paths["train_raw_embeddings"], paths["gmolai_calibrator"], dev_selected
    )

    validation = pd.read_parquet(paths["validation_molecules"])
    validation_payload = torch.load(
        paths["validation_embeddings"], map_location="cpu", weights_only=False
    )
    validation_matrix = validation_embeddings(validation_payload)
    if validation["molecule_hash"].astype(str).tolist() != [
        str(value) for value in validation_payload["molecule_hashes"]
    ]:
        raise RuntimeError("Validation molecule and embedding rows differ")
    original = pd.read_csv(paths["step2_original_final_panel"])
    if "validation_index" not in original.columns:
        raise RuntimeError("Original Step-2 validation index column changed")
    step2b_final = pd.read_csv(paths["step2b_final_panel"])
    excluded_final = set(original["validation_index"].astype(int).tolist())
    excluded_final.update(step2b_final["target_index"].astype(int).tolist())
    if len(excluded_final) != 20_000:
        raise RuntimeError("Prior Step-2 and Step-2b final panels overlap or changed")
    eligible_final = np.asarray(
        [value for value in range(len(validation)) if value not in excluded_final],
        dtype=np.int64,
    )
    final_selected = deterministic_subset(
        eligible_final,
        validation.iloc[eligible_final]["molecule_hash"].astype(str).tolist(),
        int(cfg["panels"]["final_rows"]),
        int(cfg["seed"]),
        "step2d-fresh-final-validation",
    )
    final = panel_frame(final_selected, validation)
    final_conditions = np.ascontiguousarray(validation_matrix[final_selected], dtype=np.float32)

    if set(development["target_hash"]).intersection(final["target_hash"]):
        raise RuntimeError("Development and final molecular identities overlap")
    novelty = train.iloc[train_indices][["molecule_hash", "canonical_smiles"]].copy()
    novelty["molecule_hash"] = novelty["molecule_hash"].astype(str)
    if len(novelty) != 980_000 or novelty["molecule_hash"].nunique() != 980_000:
        raise RuntimeError("Decoder-fit novelty reference is not 980,000 unique identities")
    if set(development["target_hash"]).intersection(novelty["molecule_hash"]):
        raise RuntimeError("Decoder development identities appear in decoder-fit reference")

    dev_path = root / "prepared" / "development_panel.csv"
    final_path = root / "prepared" / "fresh_validation_panel.csv"
    dev_cond_path = root / "prepared" / "development_conditions.npy"
    final_cond_path = root / "prepared" / "final_conditions.npy"
    novelty_path = root / "prepared" / "decoder_training_identities.parquet"
    atomic_write_csv(dev_path, development, root)
    atomic_write_csv(final_path, final, root)
    atomic_numpy_save(dev_cond_path, development_conditions, root)
    atomic_numpy_save(final_cond_path, final_conditions, root)
    atomic_write_parquet(novelty_path, novelty.sort_values("molecule_hash"), root)
    metadata = {
        "schema_version": 1,
        "prepared_at": utc_now(),
        "development": {
            "partition": "decoder_scaffold_disjoint_development_holdout",
            "rows": len(development),
            "excluded_prior_step2b_rows": len(excluded_dev),
            "panel_sha256": sha256_file(dev_path),
            "conditions_sha256": sha256_file(dev_cond_path),
        },
        "final": {
            "partition": "pretraining_validation",
            "rows": len(final),
            "excluded_prior_final_rows": len(excluded_final),
            "eligible_rows_before_selection": len(eligible_final),
            "panel_sha256": sha256_file(final_path),
            "conditions_sha256": sha256_file(final_cond_path),
            "post_hoc": False,
        },
        "decoder_training_novelty_reference": {
            "rows": len(novelty),
            "unique_identities": int(novelty["molecule_hash"].nunique()),
            "sha256": sha256_file(novelty_path),
        },
        "input_sha256": hashes,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    metadata_path = root / "prepared" / "panel_metadata.json"
    atomic_write_json(metadata_path, metadata, root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": utc_now(),
        "development_panel_sha256": sha256_file(dev_path),
        "final_panel_sha256": sha256_file(final_path),
        "novelty_reference_sha256": sha256_file(novelty_path),
        "final_generation_started": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
