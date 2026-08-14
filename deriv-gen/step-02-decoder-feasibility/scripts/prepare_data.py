#!/usr/bin/env python3
"""Validate frozen training inputs and prepare lossless byte-SMILES targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from study_common import (
    atomic_numpy_save,
    atomic_numpy_savez,
    atomic_write_json,
    load_validate_manifest,
    scaffold_group_keys,
    sha256_file,
    stable_digest,
    token_matrix,
)


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
    paths, hashes = load_validate_manifest(repo_root, step_root, manifest)

    molecules = pd.read_parquet(paths["train_molecules"])
    expected_columns = {
        "molecule_index",
        "molecule_hash",
        "canonical_smiles",
        "scaffold",
        "heavy_atoms",
    }
    if set(molecules.columns) != expected_columns:
        raise RuntimeError(
            "Training chemistry table contains unexpected columns"
        )
    payload = torch.load(
        paths["train_raw_embeddings"], map_location="cpu", weights_only=False
    )
    if payload["metadata"].get("split") != "train":
        raise RuntimeError("Frozen embedding export is not training partition")
    if (
        payload["metadata"].get("embedding_definition")
        != "clean_graph_z_plus_mean_node_z_raw_blocks"
    ):
        raise RuntimeError("Frozen train export is not raw hybrid blocks")
    if payload["metadata"].get("checkpoint_sha256") != hashes["checkpoint"]:
        raise RuntimeError("Train embedding checkpoint identity changed")
    if int(payload["metadata"].get("sampling_seed")) != 1_618_033:
        raise RuntimeError("Train embedding sampling identity changed")
    hashes_in_payload = [str(value) for value in payload["molecule_hashes"]]
    hashes_in_table = molecules["molecule_hash"].astype(str).tolist()
    expected_rows = int(config["data"]["train_rows"])
    if (
        len(molecules) != expected_rows
        or len(hashes_in_payload) != expected_rows
        or hashes_in_payload != hashes_in_table
    ):
        raise RuntimeError("Training chemistry and embedding rows are misaligned")
    if molecules["molecule_hash"].duplicated().any():
        raise RuntimeError("Training molecule identities are not unique")

    smiles = molecules["canonical_smiles"].astype(str).tolist()
    encoded = token_matrix(
        smiles, int(config["data"]["maximum_smiles_bytes"])
    )
    round_trip = []
    for row in encoded[:1000]:
        values = []
        for token in row[1:]:
            token = int(token)
            if token == 2:
                break
            if token == 0:
                raise RuntimeError("Tokenized SMILES lost EOS")
            values.append(token - 3)
        round_trip.append(bytes(values).decode("ascii"))
    if round_trip != smiles[:1000]:
        raise RuntimeError("Lossless byte-token round trip failed")

    group_keys = scaffold_group_keys(molecules)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[key].append(index)
    seed = int(config["seed"])
    ordered_groups = sorted(
        grouped,
        key=lambda value: stable_digest(seed, "train-dev-scaffold", value),
    )
    target = int(config["data"]["train_partition_dev_rows_target"])
    selected_groups: set[str] = set()
    selected_rows = 0
    for key in ordered_groups:
        selected_groups.add(key)
        selected_rows += len(grouped[key])
        if selected_rows >= target:
            break
    dev = np.asarray(
        [index for index, key in enumerate(group_keys) if key in selected_groups],
        dtype=np.int64,
    )
    train = np.asarray(
        [index for index, key in enumerate(group_keys) if key not in selected_groups],
        dtype=np.int64,
    )
    if set(group_keys[index] for index in train).intersection(
        group_keys[index] for index in dev
    ):
        raise RuntimeError("Train/dev scaffold-group separation failed")
    if len(train) + len(dev) != expected_rows:
        raise RuntimeError("Prepared split lost training rows")

    prepared = step_root / "prepared"
    atomic_numpy_save(prepared / "tokens.npy", encoded, step_root)
    atomic_numpy_savez(
        prepared / "split_indices.npz",
        step_root,
        train_indices=train,
        dev_indices=dev,
    )
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "rows": expected_rows,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "scaffold_groups": len(grouped),
        "dev_scaffold_groups": len(selected_groups),
        "tokenizer": config["data"]["tokenizer"],
        "token_matrix_shape": list(encoded.shape),
        "maximum_observed_smiles_bytes": max(map(len, smiles)),
        "minimum_observed_smiles_bytes": min(map(len, smiles)),
        "ascii_characters": sorted(set("".join(smiles))),
        "stereochemical_smiles": int(
            sum(
                ("@" in value) or ("/" in value) or ("\\" in value)
                for value in smiles
            )
        ),
        "input_sha256": hashes,
    }
    atomic_write_json(
        prepared / "dataset_metadata.json", metadata, step_root
    )
    outputs = {
        "tokens": prepared / "tokens.npy",
        "splits": prepared / "split_indices.npz",
        "metadata": prepared / "dataset_metadata.json",
    }
    seal = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            name: {
                "path": str(path.relative_to(step_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
        "summary": metadata,
    }
    atomic_write_json(
        step_root / "state" / "PREPARED.json", seal, step_root
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
