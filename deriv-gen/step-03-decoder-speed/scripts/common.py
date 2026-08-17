#!/usr/bin/env python3
"""Shared utilities for the isolated Step 03 benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ratio_bootstrap(
    numerators: Any,
    denominators: Any,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    import numpy as np

    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    if numerator.ndim != 1 or numerator.shape != denominator.shape:
        raise ValueError("Bootstrap arrays must be equal one-dimensional vectors")
    if len(numerator) < 2 or np.any(denominator <= 0.0):
        raise ValueError("Bootstrap requires at least two positive-denominator units")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), size=(replicates, len(numerator)))
    ratios = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(ratios, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)
