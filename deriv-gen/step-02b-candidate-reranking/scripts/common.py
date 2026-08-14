"""Shared integrity, chemistry, and metric utilities for Step 2b."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

THIS_FILE = Path(__file__).resolve()
STEP_ROOT = THIS_FILE.parents[1]
DERIV_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
STEP2_ROOT = DERIV_ROOT / "step-02-decoder-feasibility"
STEP2_SCRIPTS = STEP2_ROOT / "scripts"
REPO_SOURCE = REPO_ROOT / "src"
if str(REPO_SOURCE) not in sys.path:
    sys.path.insert(0, str(REPO_SOURCE))
if str(STEP2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP2_SCRIPTS))

from decoder_model import ConditionalSmilesTransformer  # noqa: E402
from study_common import (  # noqa: E402
    atomic_numpy_savez,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    bootstrap_mean_ci,
    cosine_rows,
    decode_tokens,
    derangement,
    deterministic_panel_indices,
    make_fingerprints,
    sha256_file,
    stable_digest,
    tanimoto,
    topk_l2,
    validation_embeddings,
)
from gmolai_retrain.chem import Rejection, canonicalize  # noqa: E402


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def protocol(root: Path = STEP_ROOT) -> dict[str, Any]:
    return load_json(root / "config" / "protocol.json")


def ensure_inside(path: Path, root: Path = STEP_ROOT) -> Path:
    resolved = path.resolve()
    expected = root.resolve()
    if resolved != expected and expected not in resolved.parents:
        raise RuntimeError(f"Refusing Step-2b write outside {expected}: {resolved}")
    return resolved


def validate_manifest(
    repo_root: Path = REPO_ROOT, root: Path = STEP_ROOT
) -> tuple[dict[str, Path], dict[str, str]]:
    manifest_path = root / "inputs" / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("embedding_space") != "released_hybrid_w3":
        raise RuntimeError("Step-2b manifest changed embedding space")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    forbidden = ("test-partition", "moleculenet", "hiv")
    for role, record in manifest["files"].items():
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input {role}: {path}")
        observed = sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"Frozen input changed for {role}: {observed} != {expected}"
            )
        if role != "container" and any(
            token in str(path).lower() for token in forbidden
        ):
            raise RuntimeError(f"Forbidden evaluation input: {path}")
        if root.resolve() in path.resolve().parents:
            raise RuntimeError(f"Manifest input is not external to Step 2b: {path}")
        paths[role] = path
        hashes[role] = observed

    expected_equal = [
        ("gmolai_checkpoint", "packaged_checkpoint"),
        ("gmolai_calibrator", "packaged_calibrator"),
        ("gmolai_resolved_config", "packaged_resolved_config"),
    ]
    for first, second in expected_equal:
        if hashes[first] != hashes[second]:
            raise RuntimeError(f"Frozen identities differ: {first} vs {second}")
    training = load_json(paths["step2_training_complete"])
    if training.get("best_checkpoint_sha256") != hashes["decoder_checkpoint"]:
        raise RuntimeError("Step-2 training seal does not bind decoder checkpoint")
    complete = load_json(paths["step2_complete"])
    if complete.get("decoder_checkpoint_sha256") != hashes["decoder_checkpoint"]:
        raise RuntimeError("Step-2 completion seal does not bind decoder checkpoint")
    if complete.get("latent_perturbation_performed") is not False:
        raise RuntimeError("Unexpected Step-2 scientific boundary")
    return paths, hashes


def load_decoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[ConditionalSmilesTransformer, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("artifact_type") != "decoder_only":
        raise RuntimeError("Frozen Step-2 checkpoint is not decoder-only")
    model = ConditionalSmilesTransformer(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, checkpoint


def policy_canonicalize(
    raw_smiles: str, resolved_config: dict[str, Any]
) -> tuple[Any | None, str, bool]:
    if not raw_smiles:
        return None, "decoder_token_error", False
    rdkit_valid = Chem.MolFromSmiles(raw_smiles) is not None
    if not rdkit_valid:
        return None, "rdkit_parse_failure", False
    data = resolved_config["data"]
    policy = data["canonicalization"]
    result = canonicalize(
        raw_smiles,
        isomeric_smiles=bool(policy["isomeric_smiles"]),
        fragment_policy=str(policy["fragment_policy"]),
        allowed_elements={str(item) for item in policy["allowed_elements"]},
        min_atoms=int(policy["min_atoms"]),
        max_atoms=int(policy["max_atoms"]),
        buckets=int(data["hash_buckets"]),
        split_cfg=data["split"],
    )
    if isinstance(result, Rejection):
        return None, result.reason, True
    return result, "", True


def released_train_rows(
    raw_path: Path,
    calibrator_path: Path,
    indices: np.ndarray,
) -> np.ndarray:
    raw_payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw = raw_payload["embeddings"]
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] != 384:
        raise RuntimeError("Unexpected train raw embedding payload")
    selected = raw[torch.as_tensor(indices, dtype=torch.long)].float().numpy()
    del raw_payload, raw
    calibrator = torch.load(
        calibrator_path, map_location="cpu", weights_only=False
    )
    mean = calibrator["coordinate_mean"].detach().cpu().float().numpy()
    scale = calibrator["coordinate_scale"].detach().cpu().float().numpy()
    if mean.shape != (384,) or scale.shape != (384,):
        raise RuntimeError("Unexpected frozen calibrator dimensions")
    result = np.ascontiguousarray(
        (selected - mean[None, :]) / scale[None, :], dtype=np.float32
    )
    result[:, 256:] *= 3.0
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite released train conditions")
    return result


def load_released_inference(repo_root: Path = REPO_ROOT) -> Any:
    path = repo_root / "inference" / "generate_embeddings.py"
    spec = importlib.util.spec_from_file_location(
        "gmolai_step2b_released_inference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load released inference entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fingerprint_generator(radius: int = 2, bits: int = 2048) -> Any:
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(bits)
    )


def fingerprint(smiles: str, generator: Any) -> Any:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise RuntimeError(f"Cannot fingerprint canonical SMILES: {smiles}")
    return generator.GetFingerprint(molecule)


def cosine_one(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return math.nan
    return float(np.dot(first.astype(np.float64), second.astype(np.float64)) / denominator)


def dataframe_hash(path: Path) -> str:
    return sha256_file(path)


def require_one_gpu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Step 2b requires exactly one visible GPU")
    return torch.device("cuda:0")


def configure_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_hash_ledger(root: Path, output_path: Path) -> None:
    omitted = {output_path.resolve()}
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in omitted or "__pycache__" in path.parts:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    atomic_write_text(output_path, "\n".join(lines) + "\n", root)


__all__ = [
    "ConditionalSmilesTransformer",
    "DERIV_ROOT",
    "REPO_ROOT",
    "STEP2_ROOT",
    "STEP_ROOT",
    "atomic_numpy_savez",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
    "bootstrap_mean_ci",
    "configure_determinism",
    "cosine_one",
    "cosine_rows",
    "decode_tokens",
    "derangement",
    "deterministic_panel_indices",
    "fingerprint",
    "fingerprint_generator",
    "load_decoder",
    "load_json",
    "load_released_inference",
    "make_fingerprints",
    "policy_canonicalize",
    "protocol",
    "released_train_rows",
    "require_one_gpu",
    "sha256_file",
    "stable_digest",
    "tanimoto",
    "topk_l2",
    "utc_now",
    "validate_manifest",
    "validation_embeddings",
    "write_hash_ledger",
]
