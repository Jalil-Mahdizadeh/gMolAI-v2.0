"""Shared integrity and I/O utilities for the frozen Step 2d study."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


THIS_FILE = Path(__file__).resolve()
STEP_ROOT = THIS_FILE.parents[1]
DERIV_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]
STEP1B_ROOT = DERIV_ROOT / "step-01b-scaled-space-selection"
STEP2_ROOT = DERIV_ROOT / "step-02-decoder-feasibility"
STEP2B_ROOT = DERIV_ROOT / "step-02b-candidate-reranking"
STEP2C_ROOT = DERIV_ROOT / "step-02c-chemical-characterization"

for source in (
    REPO_ROOT / "src",
    STEP2_ROOT / "scripts",
    STEP2B_ROOT / "scripts",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from decoder_model import ConditionalSmilesTransformer  # noqa: E402
from study_common import (  # noqa: E402
    atomic_numpy_save,
    atomic_numpy_savez,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    decode_tokens,
    released_train_embeddings,
    sha256_file,
    stable_digest,
    validation_embeddings,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def protocol(root: Path = STEP_ROOT) -> dict[str, Any]:
    return load_json(root / "config" / "protocol.json")


def ensure_inside(path: Path, root: Path = STEP_ROOT) -> Path:
    resolved = path.resolve()
    expected = root.resolve()
    if resolved != expected and expected not in resolved.parents:
        raise RuntimeError(f"Refusing Step-2d write outside {expected}: {resolved}")
    return resolved


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": math.nan,
            "standard_deviation": math.nan,
            "minimum": math.nan,
            "q10": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q90": math.nan,
            "maximum": math.nan,
        }
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "q10": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "maximum": float(array.max()),
    }


def resolve_manifest_inputs(
    repo_root: Path = REPO_ROOT, root: Path = STEP_ROOT
) -> tuple[dict[str, Path], dict[str, str]]:
    manifest_path = root / "inputs" / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("embedding_space") != "released_hybrid_w3":
        raise RuntimeError("Step-2d embedding-space identity changed")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    forbidden = ("test-partition", "test-standardized", "moleculenet", "hiv")
    for role, record in manifest["files"].items():
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input {role}: {path}")
        observed = sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            raise RuntimeError(f"Frozen input changed for {role}: {observed} != {expected}")
        if role != "container" and any(token in str(path).lower() for token in forbidden):
            raise RuntimeError(f"Forbidden evaluation input: {path}")
        if root.resolve() in path.resolve().parents:
            raise RuntimeError(f"Manifest input must be external to Step 2d: {path}")
        paths[role] = path
        hashes[role] = observed

    for first, second in (
        ("gmolai_checkpoint", "packaged_checkpoint"),
        ("gmolai_calibrator", "packaged_calibrator"),
        ("gmolai_resolved_config", "packaged_resolved_config"),
    ):
        if hashes[first] != hashes[second]:
            raise RuntimeError(f"Frozen identities differ: {first} vs {second}")
    training = load_json(paths["step2_training_complete"])
    complete = load_json(paths["step2_complete"])
    if training.get("best_checkpoint_sha256") != hashes["decoder_checkpoint"]:
        raise RuntimeError("Step-2 training seal does not bind decoder checkpoint")
    if complete.get("decoder_checkpoint_sha256") != hashes["decoder_checkpoint"]:
        raise RuntimeError("Step-2 completion seal does not bind decoder checkpoint")
    if complete.get("test_rows") != 0 or complete.get("endpoint_labels_used") is not False:
        raise RuntimeError("Step-2 scientific boundary changed")
    return paths, hashes


def load_decoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[ConditionalSmilesTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("artifact_type") != "decoder_only":
        raise RuntimeError("Frozen checkpoint is not decoder-only")
    model = ConditionalSmilesTransformer(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def require_one_gpu() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each Step-2d generation shard requires exactly one visible GPU")
    return torch.device("cuda:0")


def configure_determinism(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)


def released_train_rows(
    raw_path: Path, calibrator_path: Path, indices: np.ndarray
) -> np.ndarray:
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw = payload["embeddings"]
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] != 384:
        raise RuntimeError("Unexpected frozen train embedding payload")
    selected = raw[torch.as_tensor(indices, dtype=torch.long)].float().numpy()
    calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
    mean = calibrator["coordinate_mean"].detach().cpu().float().numpy()
    scale = calibrator["coordinate_scale"].detach().cpu().float().numpy()
    result = np.ascontiguousarray((selected - mean[None, :]) / scale[None, :])
    result[:, 256:] *= 3.0
    if result.shape != (len(indices), 384) or not np.isfinite(result).all():
        raise RuntimeError("Invalid released-hybrid-w3 train conditions")
    return result.astype(np.float32, copy=False)


def deterministic_subset(
    indices: np.ndarray,
    hashes: list[str],
    count: int,
    seed: int,
    label: str,
) -> np.ndarray:
    if len(indices) != len(hashes) or not 0 < count <= len(indices):
        raise ValueError("Invalid deterministic subset inputs")
    order = sorted(
        range(len(indices)),
        key=lambda position: stable_digest(seed, label, hashes[position]),
    )
    return indices[np.asarray(order[:count], dtype=np.int64)]


def write_hash_ledger(root: Path, path: Path) -> None:
    lines: list[str] = []
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        if item.resolve() == path.resolve() or "__pycache__" in item.parts:
            continue
        lines.append(f"{sha256_file(item)}  {item.relative_to(root).as_posix()}")
    atomic_write_text(path, "\n".join(lines) + "\n", root)


__all__ = [
    "ConditionalSmilesTransformer",
    "DERIV_ROOT",
    "REPO_ROOT",
    "STEP1B_ROOT",
    "STEP2B_ROOT",
    "STEP2C_ROOT",
    "STEP2_ROOT",
    "STEP_ROOT",
    "atomic_numpy_save",
    "atomic_numpy_savez",
    "atomic_write_csv",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
    "configure_determinism",
    "decode_tokens",
    "deterministic_subset",
    "ensure_inside",
    "load_decoder",
    "load_json",
    "numeric_summary",
    "protocol",
    "released_train_rows",
    "require_one_gpu",
    "resolve_manifest_inputs",
    "sha256_file",
    "stable_digest",
    "utc_now",
    "validation_embeddings",
    "write_hash_ledger",
]
