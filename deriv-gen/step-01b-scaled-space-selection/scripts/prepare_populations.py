#!/usr/bin/env python3
"""Join molecule identities and fragment the scaled train/validation populations."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from scaled_common import (
    atomic_write_json,
    atomic_write_parquet,
    chemical_records,
    fragment_molecules,
    load_validate_manifest,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01b-scaled-space-selection"),
    )
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    config = json.loads(
        (step_root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (step_root / "inputs" / "manifest.json").read_text(encoding="utf-8")
    )
    paths, _ = load_validate_manifest(repo_root, step_root, manifest)
    export_seal_path = step_root / "state" / "EXPORT_COMPLETE.json"
    if not export_seal_path.is_file():
        raise RuntimeError("The 1M export has not been validated")
    export_seal = json.loads(export_seal_path.read_text(encoding="utf-8"))
    train_path = step_root / export_seal["export_path"]
    if sha256_file(train_path) != export_seal["export_sha256"]:
        raise RuntimeError("The sealed 1M train export changed")
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    validation = torch.load(
        paths["validation_embeddings"], map_location="cpu", weights_only=False
    )
    if train["metadata"]["split"] != "train":
        raise RuntimeError("Train export split identity failed")
    if validation["metadata"]["split"] != "validation":
        raise RuntimeError("Validation export split identity failed")
    if int(validation["metadata"]["embedding_parameters"]["mean_node_weight"]) != 3:
        raise RuntimeError("Validation payload is not the released weight-3 vector")
    if len(train["molecule_hashes"]) != int(config["train_rows"]):
        raise RuntimeError("Train row count changed")
    if len(validation["molecule_hashes"]) != int(config["validation_rows"]):
        raise RuntimeError("Validation row count changed")

    cache = json.loads(paths["chemical_record_cache"].read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        paths["dataset_manifest"].read_text(encoding="utf-8")
    )
    if cache["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Chemical cache and dataset manifest identities differ")

    print("joining 1M train molecule identities to immutable chemistry", flush=True)
    train_hashes, train_smiles, train_scaffolds = chemical_records(
        train, cache["records"], repo_root / "work"
    )
    print("joining 50k validation molecule identities to immutable chemistry", flush=True)
    validation_hashes, validation_smiles, validation_scaffolds = chemical_records(
        validation, cache["records"], repo_root / "work"
    )
    overlap = set(train_hashes).intersection(validation_hashes)
    if overlap:
        raise RuntimeError(f"Train/validation molecule identity overlap: {len(overlap)}")

    workers = args.workers or min(48, len(os.sched_getaffinity(0)))
    if workers <= 0:
        raise RuntimeError("No fragmentation workers are available")
    print(f"fragmenting 1M train molecules with {workers} workers", flush=True)
    train_fragments, train_heavy, train_stats = fragment_molecules(
        train_smiles, settings=config["mmp"], workers=workers
    )
    print(f"fragmenting 50k validation molecules with {workers} workers", flush=True)
    validation_fragments, validation_heavy, validation_stats = fragment_molecules(
        validation_smiles, settings=config["mmp"], workers=workers
    )
    if train_stats["parse_failures"] or validation_stats["parse_failures"]:
        raise RuntimeError("An immutable molecule failed RDKit parsing")

    intermediate = step_root / "intermediate"
    train_molecules = pd.DataFrame(
        {
            "molecule_index": range(len(train_hashes)),
            "molecule_hash": train_hashes,
            "canonical_smiles": train_smiles,
            "scaffold": train_scaffolds,
            "heavy_atoms": train_heavy,
        }
    )
    validation_molecules = pd.DataFrame(
        {
            "molecule_index": range(len(validation_hashes)),
            "molecule_hash": validation_hashes,
            "canonical_smiles": validation_smiles,
            "scaffold": validation_scaffolds,
            "heavy_atoms": validation_heavy,
        }
    )
    atomic_write_parquet(
        intermediate / "train_molecules.parquet", train_molecules, step_root
    )
    atomic_write_parquet(
        intermediate / "validation_molecules.parquet",
        validation_molecules,
        step_root,
    )
    atomic_write_parquet(
        intermediate / "train_fragments.parquet", train_fragments, step_root
    )
    atomic_write_parquet(
        intermediate / "validation_fragments.parquet",
        validation_fragments,
        step_root,
    )
    outputs = {
        "train_molecules": intermediate / "train_molecules.parquet",
        "validation_molecules": intermediate / "validation_molecules.parquet",
        "train_fragments": intermediate / "train_fragments.parquet",
        "validation_fragments": intermediate / "validation_fragments.parquet",
    }
    seal = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "identity_overlap": 0,
        "train": train_stats,
        "validation": validation_stats,
        "outputs": {
            name: {
                "path": str(path.relative_to(step_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    atomic_write_json(
        step_root / "state" / "FRAGMENTATION_COMPLETE.json", seal, step_root
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
