#!/usr/bin/env python3
"""Hash-bind every frozen Step-2b input before candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    STEP2_ROOT,
    STEP_ROOT,
    atomic_write_json,
    load_json,
    sha256_file,
    utc_now,
)


def relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    state_path = root / "state" / "REGISTERED.json"
    manifest_path = root / "inputs" / "manifest.json"
    if state_path.exists():
        state = load_json(state_path)
        if not manifest_path.is_file() or sha256_file(manifest_path) != state["manifest_sha256"]:
            raise RuntimeError("Registered Step-2b manifest changed")
        print(json.dumps(state, sort_keys=True))
        return
    forbidden_outputs = [
        root / "state" / "DEVELOPMENT_COMPLETE.json",
        root / "state" / "POLICY_FROZEN.json",
        root / "state" / "FINAL_COMPLETE.json",
    ]
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("Refusing to register after candidate evaluation")
    if list(root.rglob("*.pt")):
        raise RuntimeError("Step 2b must not contain model checkpoints")

    step2_manifest = load_json(STEP2_ROOT / "inputs" / "manifest.json")
    inherited = {
        "gmolai_checkpoint": "checkpoint",
        "gmolai_calibrator": "calibrator",
        "gmolai_resolved_config": "resolved_config",
        "train_raw_embeddings": "train_raw_embeddings",
        "train_molecules": "train_molecules",
        "validation_embeddings": "validation_embeddings",
        "validation_molecules": "validation_molecules",
        "packaged_checkpoint": "packaged_checkpoint",
        "packaged_calibrator": "packaged_calibrator",
        "packaged_resolved_config": "packaged_resolved_config",
        "representation_selection": "representation_selection",
        "inference_entrypoint": "inference_entrypoint",
        "chemistry_policy": "chemistry_policy",
        "optimized_inference": "optimized_inference",
        "gmolai_model_definition": "model_definition",
        "container": "container",
    }
    files: dict[str, dict[str, str]] = {}
    for role, inherited_role in inherited.items():
        record = step2_manifest["files"][inherited_role]
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Step-2 inherited input changed: {inherited_role}")
        files[role] = {
            "path": relative_or_absolute(path, repo_root),
            "sha256": observed,
        }

    explicit = {
        "decoder_checkpoint": STEP2_ROOT / "checkpoints" / "best.pt",
        "decoder_inference_export": STEP2_ROOT / "checkpoints" / "decoder_inference.pt",
        "step2_training_complete": STEP2_ROOT / "state" / "TRAINING_COMPLETE.json",
        "step2_complete": STEP2_ROOT / "state" / "COMPLETE.json",
        "step2_manifest": STEP2_ROOT / "inputs" / "manifest.json",
        "step2_split_indices": STEP2_ROOT / "prepared" / "split_indices.npz",
        "step2_original_final_panel": STEP2_ROOT / "prepared" / "final_control_sources.csv",
        "step2_decoder_model_source": STEP2_ROOT / "scripts" / "decoder_model.py",
        "step2_common_source": STEP2_ROOT / "scripts" / "study_common.py",
    }
    for role, path in explicit.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        files[role] = {
            "path": relative_or_absolute(path, repo_root),
            "sha256": sha256_file(path),
        }

    training = load_json(explicit["step2_training_complete"])
    complete = load_json(explicit["step2_complete"])
    if training.get("best_checkpoint_sha256") != files["decoder_checkpoint"]["sha256"]:
        raise RuntimeError("Training seal and selected decoder differ")
    if complete.get("decoder_checkpoint_sha256") != files["decoder_checkpoint"]["sha256"]:
        raise RuntimeError("Step-2 completion seal and selected decoder differ")
    if complete.get("test_rows") != 0 or complete.get("endpoint_labels_used") is not False:
        raise RuntimeError("Step-2 boundary changed")

    manifest = {
        "schema_version": 1,
        "study_id": "gmolai-decoder-candidate-reranking-step2b-v1",
        "embedding_space": "released_hybrid_w3",
        "policy": "all encoder, calibrator, decoder, chemistry, partition, and inference inputs are immutable",
        "files": files,
        "forbidden_inputs": [
            "locked internal test partition",
            "MoleculeNet or HIV endpoint labels",
            "target structural quantities as ranking features",
        ],
    }
    atomic_write_json(manifest_path, manifest, root)
    local_sources = sorted((root / "scripts").glob("*.py"))
    local_sources.extend(sorted((root / "scripts").glob("*.sh")))
    local_sources.extend(sorted((root / "config").glob("*.json")))
    source_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path) for path in local_sources
    }
    state = {
        "schema_version": 1,
        "status": "registered_before_candidate_generation",
        "registered_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(root / "config" / "protocol.json"),
        "registered_source_sha256": source_hashes,
        "decoder_checkpoint_sha256": files["decoder_checkpoint"]["sha256"],
        "gmolai_checkpoint_sha256": files["gmolai_checkpoint"]["sha256"],
        "calibrator_sha256": files["gmolai_calibrator"]["sha256"],
        "embedding_space": "released_hybrid_w3",
        "decoder_training": False,
        "encoder_training": False,
        "latent_perturbation": False,
        "derivative_generation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
