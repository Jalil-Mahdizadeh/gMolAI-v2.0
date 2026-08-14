"""Integrity and atomic-output helpers for the Step 2c chemistry audit."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
STEP_ROOT = THIS_FILE.parents[1]
DERIV_ROOT = THIS_FILE.parents[2]
REPO_ROOT = THIS_FILE.parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(str(value) for value in parts).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def ensure_inside(path: Path, root: Path = STEP_ROOT) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise RuntimeError(f"Refusing Step-2c write outside {boundary}: {resolved}")
    return resolved


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path: Path, value: str, root: Path = STEP_ROOT) -> None:
    path = ensure_inside(path, root)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, root: Path = STEP_ROOT) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        root,
    )


def atomic_write_csv(
    path: Path, frame: pd.DataFrame, root: Path = STEP_ROOT
) -> None:
    path = ensure_inside(path, root)
    temporary = _temporary_sibling(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(
    path: Path, frame: pd.DataFrame, root: Path = STEP_ROOT
) -> None:
    path = ensure_inside(path, root)
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, root: Path = STEP_ROOT, **arrays: np.ndarray) -> None:
    path = ensure_inside(path, root)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protocol(root: Path = STEP_ROOT) -> dict[str, Any]:
    return load_json(root / "config" / "protocol.json")


def resolve_manifest_inputs(
    repo_root: Path = REPO_ROOT, root: Path = STEP_ROOT
) -> tuple[dict[str, Path], dict[str, str]]:
    manifest_path = root / "inputs" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Step 2c is not registered")
    manifest = load_json(manifest_path)
    if manifest.get("study_id") != protocol(root).get("study_id"):
        raise RuntimeError("Manifest/protocol study identifiers differ")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for role, record in manifest["files"].items():
        raw = Path(str(record["path"]))
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input {role}: {path}")
        observed = sha256_file(path)
        expected = str(record["sha256"])
        if observed != expected:
            raise RuntimeError(
                f"Frozen input changed for {role}: {observed} != {expected}"
            )
        if root.resolve() in path.resolve().parents and role.startswith("external_"):
            raise RuntimeError(f"External input unexpectedly resides in Step 2c: {path}")
        paths[role] = path
        hashes[role] = observed
    return paths, hashes


def numeric_summary(values: np.ndarray | pd.Series) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q05": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "max": None,
        }
    quantiles = np.quantile(array, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "q05": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q95": float(quantiles[6]),
        "max": float(array.max()),
    }


def write_hash_ledger(root: Path = STEP_ROOT) -> None:
    output_path = root / "outputs" / "SHA256SUMS"
    omitted = {output_path.resolve()}
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in omitted or "__pycache__" in path.parts:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    atomic_write_text(output_path, "\n".join(lines) + "\n", root)
