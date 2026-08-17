"""Strict path, provenance, hashing, and atomic-output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


RELATIVE_ANALYSIS_PATH = Path(
    "deriv-gen/step-02d-generation-scaling/extra-step-02b-style"
)


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_inside(path: Path, root: Path) -> Path:
    candidate = resolved(path)
    boundary = resolved(root)
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise RuntimeError(f"Refusing path outside analysis root: {candidate}") from exc
    return candidate


def require_analysis_root(path: Path) -> Path:
    root = resolved(path)
    config = root / "config" / "analysis.json"
    protocol = root / "PROTOCOL.md"
    if not config.is_file() or not protocol.is_file():
        raise RuntimeError(f"Not the expected analysis root: {root}")
    return root


def require_repo_root(path: Path) -> Path:
    root = resolved(path)
    required = [
        root / "deriv-gen" / "step-02d-generation-scaling" / "outputs" / "verification.json",
        root / "inference" / "gmolai.py",
        root / "inference" / "models" / "representation-best.pt",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise RuntimeError(f"Invalid repository root; missing: {missing}")
    return root


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(root: Path) -> dict[str, Any]:
    return load_json(root / "config" / "analysis.json")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(target: Path, root: Path) -> Path:
    target = ensure_inside(target, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, value: Any, root: Path) -> None:
    target = ensure_inside(path, root)
    temporary = _temporary_path(target, root)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str, root: Path) -> None:
    target = ensure_inside(path, root)
    temporary = _temporary_path(target, root)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    target = ensure_inside(path, root)
    temporary = _temporary_path(target, root)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, root: Path) -> None:
    target = ensure_inside(path, root)
    temporary = _temporary_path(target, root)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def input_paths(repo_root: Path) -> dict[str, Path | list[Path]]:
    step = repo_root / "deriv-gen" / "step-02d-generation-scaling"
    return {
        "step2d_verification": step / "outputs" / "verification.json",
        "raw_proposals": sorted((step / "outputs" / "raw" / "final").glob("*.parquet")),
        "policy_audit": step / "intermediate" / "final_raw_smiles_policy_audit.parquet",
        "unique_molecules": step / "intermediate" / "final_unique_molecules.parquet",
        "candidate_characterization": step / "outputs" / "tables" / "final_candidate_characterization.parquet",
        "seed_budget_metrics": step / "outputs" / "tables" / "final_seed_budget_metrics.parquet",
        "panel": step / "prepared" / "fresh_validation_panel.csv",
        "conditions": step / "prepared" / "final_conditions.npy",
        "encoder_cli": repo_root / "inference" / "gmolai.py",
        "encoder_checkpoint": repo_root / "inference" / "models" / "representation-best.pt",
        "encoder_calibrator": repo_root / "inference" / "models" / "representation-calibrator.pt",
        "encoder_config": repo_root / "inference" / "models" / "resolved_config.json",
        "encoder_selection": repo_root / "inference" / "models" / "representation_selection.json",
        "encoder_model_source": repo_root / "src" / "gmolai_retrain" / "model.py",
        "optimized_inference_source": repo_root / "src" / "gmolai_retrain" / "fast_inference.py",
    }


def validate_inputs(paths: dict[str, Path | list[Path]]) -> None:
    missing: list[str] = []
    for value in paths.values():
        values = value if isinstance(value, list) else [value]
        if not values:
            missing.append("empty input list")
        missing.extend(str(path) for path in values if not path.is_file())
    if missing:
        raise RuntimeError(f"Required immutable inputs are missing: {missing}")
