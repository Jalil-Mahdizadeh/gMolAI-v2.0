#!/usr/bin/env python3
"""Export the selected decoder without optimizer state or any gMolAI weights."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from study_common import (
    atomic_torch_save,
    atomic_write_json,
    load_validate_manifest,
    sha256_file,
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
    root = args.step_root.resolve()
    config_path = root / "config" / "protocol.json"
    manifest_path = root / "inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, input_hashes = load_validate_manifest(repo_root, root, manifest)
    training = json.loads(
        (root / "state" / "TRAINING_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    decode_path = root / "state" / "DECODE_SELECTION.json"
    decode = json.loads(decode_path.read_text(encoding="utf-8"))
    checkpoint_path = root / training["best_checkpoint"]
    checkpoint_hash = sha256_file(checkpoint_path)
    if (
        checkpoint_hash != training["best_checkpoint_sha256"]
        or decode["decoder_checkpoint_sha256"] != checkpoint_hash
    ):
        raise RuntimeError("Selected decoder checkpoint is inconsistent")
    source = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if (
        source.get("artifact_type") != "decoder_only"
        or source["frozen_input_sha256"] != input_hashes
    ):
        raise RuntimeError("Source artifact is not the selected decoder")
    output = root / "checkpoints" / "decoder_inference.pt"
    payload = {
        "schema_version": 1,
        "artifact_type": "conditional_smiles_decoder_inference",
        "embedding_space": "released_hybrid_w3",
        "condition_dimensions": 384,
        "model_config": source["model_config"],
        "model_state_dict": source["model_state_dict"],
        "decode_method": decode["selected_decode_method"],
        "source_training_checkpoint_sha256": checkpoint_hash,
        "frozen_input_sha256": input_hashes,
        "protocol_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_torch_save(output, payload, root)
    seal = {
        "schema_version": 1,
        "status": "complete",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(output.relative_to(root)),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "contains_optimizer_state": False,
        "contains_gmolai_parameters": False,
        "decode_method": decode["selected_decode_method"],
        "source_training_checkpoint_sha256": checkpoint_hash,
        "frozen_input_sha256": input_hashes,
    }
    atomic_write_json(
        root / "state" / "DECODER_EXPORT.json", seal, root
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
