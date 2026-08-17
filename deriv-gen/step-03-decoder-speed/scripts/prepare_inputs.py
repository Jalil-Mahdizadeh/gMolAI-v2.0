#!/usr/bin/env python3
"""Select and seal the 100 released molecular conditioning vectors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from common import atomic_write_json, load_json, sha256_file, utc_now


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_numpy_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    config_path = step_root / "config" / "benchmark.json"
    config = load_json(config_path)
    panel_path = repo_root / config["source"]["panel"]
    conditions_path = repo_root / config["source"]["conditions"]
    panel_output = step_root / "inputs" / "selected_panel.csv"
    conditions_output = step_root / "inputs" / "selected_conditions.npy"
    metadata_output = step_root / "inputs" / "selection_metadata.json"
    targets = (panel_output, conditions_output, metadata_output)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Prepared inputs already exist: {existing}")
    if not panel_path.is_file() or not conditions_path.is_file():
        raise FileNotFoundError("Frozen Step 02d panel or condition matrix is missing")

    panel = pd.read_csv(panel_path)
    conditions = np.load(conditions_path, mmap_mode="r", allow_pickle=False)
    required = {
        "query_position",
        "target_index",
        "target_hash",
        "seed_canonical_smiles",
        "seed_scaffold",
        "seed_heavy_atoms",
    }
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel columns changed: {sorted(panel.columns)}")
    if conditions.shape != (len(panel), 384) or conditions.dtype != np.float32:
        raise ValueError(f"Unexpected condition matrix: {conditions.shape}, {conditions.dtype}")
    count = int(config["selection"]["count"])
    selection_seed = int(config["selection"]["seed"])
    if count <= 0 or count > len(panel):
        raise ValueError("Invalid selection count")
    rng = np.random.default_rng(selection_seed)
    positions = rng.choice(len(panel), size=count, replace=False)
    selected_panel = panel.iloc[positions].reset_index(drop=True).copy()
    selected_panel.insert(0, "source_panel_row", positions.astype(np.int64))
    selected_panel.insert(0, "benchmark_seed_index", np.arange(count, dtype=np.int64))
    selected_conditions = np.ascontiguousarray(conditions[positions], dtype=np.float32)
    if not np.isfinite(selected_conditions).all():
        raise ValueError("Selected conditions contain non-finite values")
    if selected_panel["target_hash"].duplicated().any():
        raise ValueError("Selected molecular identities are not unique")

    atomic_write_csv(panel_output, selected_panel)
    atomic_numpy_save(conditions_output, selected_conditions)
    metadata = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "source_git_commit": git_head(repo_root),
        "embedding_space": config["embedding_space"],
        "selection": config["selection"],
        "source": {
            "panel_path": config["source"]["panel"],
            "panel_rows": int(len(panel)),
            "panel_sha256": sha256_file(panel_path),
            "conditions_path": config["source"]["conditions"],
            "conditions_shape": list(conditions.shape),
            "conditions_dtype": str(conditions.dtype),
            "conditions_sha256": sha256_file(conditions_path),
        },
        "selected": {
            "rows": int(count),
            "dimensions": int(selected_conditions.shape[1]),
            "source_panel_rows_in_benchmark_order": positions.astype(int).tolist(),
            "panel_sha256": sha256_file(panel_output),
            "conditions_sha256": sha256_file(conditions_output),
        },
        "config_sha256": sha256_file(config_path),
    }
    atomic_write_json(metadata_output, metadata)
    print(f"Prepared {count} inputs at {step_root / 'inputs'}")


if __name__ == "__main__":
    main()
