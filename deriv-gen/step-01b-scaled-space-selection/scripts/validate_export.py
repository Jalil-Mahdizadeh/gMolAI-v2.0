#!/usr/bin/env python3
"""Validate and seal the generated 1M train embedding export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scaled_common import (
    atomic_write_json,
    load_validate_manifest,
    payload_array,
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
    export_path = step_root / "exports" / "train_raw_hybrid_1m.pt"
    if not export_path.is_file():
        raise FileNotFoundError(export_path)
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    embeddings = payload_array(payload, "embeddings")
    expected_rows = int(config["train_rows"])
    if embeddings.shape != (expected_rows, 384):
        raise RuntimeError(f"Unexpected train export shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Train export contains non-finite coordinates")
    if metadata.get("split") != "train":
        raise RuntimeError(f"Generated export is not train split: {metadata.get('split')}")
    if metadata.get("embedding_definition") != "clean_graph_z_plus_mean_node_z_raw_blocks":
        raise RuntimeError(
            f"Generated export is not raw hybrid: {metadata.get('embedding_definition')}"
        )
    if metadata.get("checkpoint_sha256") != hashes["checkpoint"]:
        raise RuntimeError("Generated export checkpoint identity mismatch")
    sampling = metadata.get("sampling")
    observed_seed = metadata.get("sampling_seed")
    if isinstance(sampling, dict):
        observed_seed = sampling.get("seed", observed_seed)
    if int(observed_seed) != int(config["train_sampling_seed"]):
        raise RuntimeError(
            f"Generated export sampling seed mismatch: {observed_seed}"
        )
    molecule_hashes = [str(value) for value in payload.get("molecule_hashes", [])]
    if len(molecule_hashes) != expected_rows:
        raise RuntimeError("Generated export lost molecule hashes")
    if len(set(molecule_hashes)) != expected_rows:
        raise RuntimeError("Generated train export contains duplicate molecule hashes")
    if payload_array(payload, "source_buckets").shape[0] != expected_rows:
        raise RuntimeError("Generated export source-bucket alignment failed")
    calibrator = torch.load(paths["calibrator"], map_location="cpu", weights_only=False)
    if calibrator["metadata"]["source_embedding_sha256"] != (
        "7cc13b3dd1780eafdd59fd26e9a24a20adbae333f82649fbdfa917b0333e7b77"
    ):
        raise RuntimeError("Promoted calibrator source identity changed")
    seal = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "rows": expected_rows,
        "dimensions": 384,
        "split": "train",
        "sampling_seed": int(config["train_sampling_seed"]),
        "checkpoint_sha256": hashes["checkpoint"],
        "calibrator_sha256": hashes["calibrator"],
        "export_path": str(export_path.relative_to(step_root)),
        "export_sha256": sha256_file(export_path),
        "export_size_bytes": export_path.stat().st_size,
    }
    atomic_write_json(step_root / "state" / "EXPORT_COMPLETE.json", seal, step_root)
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
