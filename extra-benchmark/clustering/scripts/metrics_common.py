#!/usr/bin/env python3
"""Exact geometry and atomic array helpers shared by benchmark evaluators."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np


def row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("Representation must be a finite rank-2 matrix")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-12):
        raise RuntimeError("Representation contains a zero-norm vector")
    values /= norms[:, None]
    return values


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_parquet(path: str | Path, frame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _gpu() -> "object":
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exact-neighbor GPU contract violated: {torch.cuda.device_count()}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def exact_knn(
    matrix: np.ndarray,
    k: int,
    *,
    metric: Literal["normalized_euclidean", "binary_tanimoto", "generalized_tanimoto"] = "normalized_euclidean",
    query_block: int = 1024,
) -> np.ndarray:
    """Exact all-vs-all kNN with deterministic lower-index tie breaking."""
    torch = _gpu()
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("kNN input is not a finite matrix")
    n = values.shape[0]
    if not 0 < k < n:
        raise ValueError(f"Invalid k={k} for n={n}")
    device = torch.device("cuda:0")
    reference = torch.as_tensor(values, dtype=torch.float64, device=device)
    if metric == "normalized_euclidean":
        norms = torch.linalg.vector_norm(reference, dim=1)
        if torch.any(torch.abs(norms - 1.0) > 1.0e-9):
            raise RuntimeError("Normalized-Euclidean kNN requires unit rows")
        reference_aux = None
    elif metric == "binary_tanimoto":
        if torch.any((reference != 0.0) & (reference != 1.0)):
            raise RuntimeError("Binary Tanimoto received nonbinary values")
        reference_aux = torch.sum(reference, dim=1)
    elif metric == "generalized_tanimoto":
        if torch.any(reference < 0.0):
            raise RuntimeError("Generalized Tanimoto received negative counts")
        reference_aux = torch.sum(reference * reference, dim=1)
    else:
        raise ValueError(metric)
    penalty = torch.arange(n, dtype=torch.float64, device=device) * 1.0e-12
    output = np.empty((n, k), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, n, query_block):
            stop = min(n, start + query_block)
            query = reference[start:stop]
            dot = query @ reference.T
            if metric == "normalized_euclidean":
                score = dot
            elif metric == "binary_tanimoto":
                denominator = reference_aux[start:stop, None] + reference_aux[None, :] - dot
                score = torch.where(denominator > 0, dot / denominator, -torch.inf)
            else:
                denominator = reference_aux[start:stop, None] + reference_aux[None, :] - dot
                score = torch.where(denominator > 0, dot / denominator, -torch.inf)
            score -= penalty[None, :]
            local = torch.arange(stop - start, device=device)
            score[local, torch.arange(start, stop, device=device)] = -torch.inf
            indices = torch.topk(score, k=k, dim=1, largest=True, sorted=True).indices
            output[start:stop] = indices.cpu().numpy().astype(np.int32, copy=False)
            del dot, score, indices
    del reference, reference_aux, penalty
    torch.cuda.empty_cache()
    return output


def exact_property_knn(matrix: np.ndarray, k: int, query_block: int = 2048) -> np.ndarray:
    """Exact squared-Euclidean neighbors for the low-dimensional property reference."""
    torch = _gpu()
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("Property kNN input is not a finite matrix")
    n = len(values)
    device = torch.device("cuda:0")
    reference = torch.as_tensor(values, dtype=torch.float64, device=device)
    squared = torch.sum(reference * reference, dim=1)
    penalty = torch.arange(n, dtype=torch.float64, device=device) * 1.0e-12
    output = np.empty((n, k), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, n, query_block):
            stop = min(n, start + query_block)
            query = reference[start:stop]
            distance2 = (
                torch.sum(query * query, dim=1)[:, None]
                + squared[None, :]
                - 2.0 * (query @ reference.T)
            )
            score = -torch.clamp(distance2, min=0.0) - penalty[None, :]
            local = torch.arange(stop - start, device=device)
            score[local, torch.arange(start, stop, device=device)] = -torch.inf
            indices = torch.topk(score, k=k, dim=1, largest=True, sorted=True).indices
            output[start:stop] = indices.cpu().numpy().astype(np.int32, copy=False)
    del reference, squared, penalty
    torch.cuda.empty_cache()
    return output

