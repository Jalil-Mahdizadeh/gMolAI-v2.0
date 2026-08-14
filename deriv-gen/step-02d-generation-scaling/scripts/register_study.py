#!/usr/bin/env python3
"""Hash-bind every frozen Step-2d input before any candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    STEP1B_ROOT,
    STEP2B_ROOT,
    STEP2C_ROOT,
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
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    manifest_path = root / "inputs" / "manifest.json"
    state_path = root / "state" / "REGISTERED.json"
    if state_path.exists():
        state = load_json(state_path)
        if sha256_file(manifest_path) != state["manifest_sha256"]:
            raise RuntimeError("Registered manifest changed")
        print(json.dumps(state, sort_keys=True))
        return
    if (root / "outputs").exists() and any(
        path.is_file() for path in (root / "outputs").rglob("*")
    ):
        raise RuntimeError("Refusing registration after outputs exist")
    if list(root.rglob("*.pt")) or list(root.rglob("*.pth")):
        raise RuntimeError("Step 2d must contain no model checkpoints")

    step2b_manifest = load_json(STEP2B_ROOT / "inputs" / "manifest.json")
    inherited_roles = [
        "chemistry_policy",
        "container",
        "decoder_checkpoint",
        "decoder_inference_export",
        "gmolai_calibrator",
        "gmolai_checkpoint",
        "gmolai_model_definition",
        "gmolai_resolved_config",
        "inference_entrypoint",
        "optimized_inference",
        "packaged_calibrator",
        "packaged_checkpoint",
        "packaged_resolved_config",
        "representation_selection",
        "step2_common_source",
        "step2_complete",
        "step2_decoder_model_source",
        "step2_manifest",
        "step2_original_final_panel",
        "step2_split_indices",
        "step2_training_complete",
        "train_molecules",
        "train_raw_embeddings",
        "validation_embeddings",
        "validation_molecules",
    ]
    files: dict[str, dict[str, str]] = {}
    for role in inherited_roles:
        record = step2b_manifest["files"][role]
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        observed = sha256_file(path)
        if observed != str(record["sha256"]):
            raise RuntimeError(f"Step-2b inherited input changed: {role}")
        files[role] = {
            "path": relative_or_absolute(path, repo_root),
            "sha256": observed,
        }

    explicit = {
        "step1b_complete": STEP1B_ROOT / "state" / "COMPLETE.json",
        "step1b_protocol": STEP1B_ROOT / "config" / "protocol.json",
        "step1b_fragment_source": STEP1B_ROOT / "scripts" / "scaled_common.py",
        "step2b_complete": STEP2B_ROOT / "state" / "COMPLETE.json",
        "step2b_verification": STEP2B_ROOT / "outputs" / "verification.json",
        "step2b_protocol": STEP2B_ROOT / "config" / "protocol.json",
        "step2b_development_panel": STEP2B_ROOT / "prepared" / "development_panel.csv",
        "step2b_final_panel": STEP2B_ROOT / "prepared" / "fresh_validation_panel.csv",
        "step2b_candidate_source": STEP2B_ROOT / "scripts" / "candidate_model.py",
        "step2b_common_source_frozen": STEP2B_ROOT / "scripts" / "common.py",
        "step2c_complete": STEP2C_ROOT / "state" / "COMPLETE.json",
        "step2c_verification": STEP2C_ROOT / "outputs" / "verification.json",
        "step2c_protocol": STEP2C_ROOT / "config" / "protocol.json",
        "step2c_audit_core_source": STEP2C_ROOT / "scripts" / "audit_core.py",
    }
    for role, path in explicit.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        files[role] = {
            "path": relative_or_absolute(path, repo_root),
            "sha256": sha256_file(path),
        }

    training = load_json(Path(files["step2_training_complete"]["path"]) if Path(files["step2_training_complete"]["path"]).is_absolute() else repo_root / files["step2_training_complete"]["path"])
    step2_complete = load_json(STEP2_ROOT / "state" / "COMPLETE.json")
    if training.get("best_checkpoint_sha256") != files["decoder_checkpoint"]["sha256"]:
        raise RuntimeError("Training seal and frozen decoder differ")
    if step2_complete.get("decoder_checkpoint_sha256") != files["decoder_checkpoint"]["sha256"]:
        raise RuntimeError("Step-2 completion seal and decoder differ")
    for prior in (STEP2B_ROOT / "state" / "COMPLETE.json", STEP2C_ROOT / "state" / "COMPLETE.json"):
        value = load_json(prior)
        if value.get("test_rows") != 0 or value.get("endpoint_labels_used") is not False:
            raise RuntimeError(f"Prior boundary changed: {prior}")

    cfg = load_json(root / "config" / "protocol.json")
    step1_cfg = load_json(STEP1B_ROOT / "config" / "protocol.json")
    for key in (
        "min_core_heavy_atoms",
        "min_variable_heavy_atoms",
        "max_variable_heavy_atoms",
        "min_core_fraction",
        "max_variable_heavy_atom_delta",
        "max_parent_heavy_atom_delta",
    ):
        if cfg["mmp"][key] != step1_cfg["mmp"][key]:
            raise RuntimeError(f"Step-2d MMP rule differs from Step 1b: {key}")

    manifest = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "embedding_space": "released_hybrid_w3",
        "policy": "all encoder, calibrator, decoder, chemistry, split, panel, and prior-analysis inputs are immutable",
        "files": files,
        "forbidden_inputs": [
            "locked internal test partition",
            "MoleculeNet or HIV endpoint labels",
            "latent perturbations",
            "MMP direction vectors",
            "target structure for generation ordering",
        ],
    }
    atomic_write_json(manifest_path, manifest, root)
    registered_sources = sorted((root / "scripts").glob("*.py"))
    registered_sources.extend(sorted((root / "scripts").glob("*.sh")))
    registered_sources.extend(sorted((root / "scripts").glob("*.slurm")))
    registered_sources.extend(sorted((root / "config").glob("*.json")))
    registered_sources.extend([root / "PROTOCOL.md", root / "DESIGN.md"])
    source_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in registered_sources
        if path.is_file()
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
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
