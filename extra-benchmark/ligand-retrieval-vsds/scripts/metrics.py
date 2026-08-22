#!/usr/bin/env python3
"""Frozen retrieval metrics, tie handling, and deterministic sampling helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def deterministic_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_anchor_sample(
    active_identities: Sequence[str],
    *,
    target_id: str,
    shots: int,
    draw_id: int,
    master_seed: int,
) -> tuple[int, tuple[str, ...]]:
    identities = tuple(sorted(active_identities))
    if len(set(identities)) != len(identities):
        raise ValueError("Active identities must be unique")
    if shots <= 0 or shots > len(identities):
        raise ValueError("Invalid anchor shot count")
    draw_seed = deterministic_seed(master_seed, target_id, shots, draw_id)
    ordered = sorted(
        identities,
        key=lambda identity: (
            hashlib.sha256(
                f"{draw_seed}\x1f{identity}".encode("utf-8")
            ).digest(),
            identity,
        ),
    )
    return draw_seed, tuple(ordered[:shots])


def deterministic_random_scores(
    identities: Iterable[str],
    *,
    target_id: str,
    shots: int,
    draw_id: int,
    master_seed: int,
) -> np.ndarray:
    seed = deterministic_seed("random-control", master_seed, target_id, shots, draw_id)
    maximum = float(1 << 64)
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}\x1f{identity}".encode("utf-8")).digest()[:8],
                "big",
            )
            / maximum
            for identity in identities
        ],
        dtype=np.float64,
    )


def candidate_mask(
    labels: Sequence[int] | np.ndarray,
    scaffolds: Sequence[str] | np.ndarray,
    anchor_indices: Sequence[int],
    *,
    scaffold_excluded: bool,
) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int8)
    scaffolds_array = np.asarray(scaffolds, dtype=object)
    if labels_array.ndim != 1 or scaffolds_array.shape != labels_array.shape:
        raise ValueError("Labels and scaffolds must be aligned one-dimensional arrays")
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if anchors.ndim != 1 or len(set(anchors.tolist())) != anchors.size:
        raise ValueError("Anchor indices must be a unique one-dimensional sequence")
    if anchors.size == 0 or np.any(anchors < 0) or np.any(anchors >= labels_array.size):
        raise ValueError("Anchor index is missing or out of bounds")
    if np.any(labels_array[anchors] != 1):
        raise ValueError("Every anchor must be active")
    mask = np.ones(labels_array.size, dtype=bool)
    mask[anchors] = False
    if scaffold_excluded:
        anchor_scaffolds = {
            str(scaffolds_array[index])
            for index in anchors
            if str(scaffolds_array[index])
        }
        if anchor_scaffolds:
            mask &= np.asarray(
                [str(value) not in anchor_scaffolds for value in scaffolds_array],
                dtype=bool,
            )
    return mask


def fractional_ef_at_fraction(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    fraction: float = 0.01,
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or truth.shape != values.shape or values.size == 0:
        raise ValueError("Scores and labels must be aligned, nonempty vectors")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain non-finite values")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Fraction must be in (0, 1]")
    active_total = int(truth.sum())
    if active_total <= 0:
        raise ValueError("At least one active candidate is required")
    count = int(values.size)
    cutoff = max(1, int(math.ceil(fraction * count)))
    boundary = float(np.partition(values, count - cutoff)[count - cutoff])
    above = values > boundary
    tied = values == boundary
    above_count = int(above.sum())
    tie_count = int(tied.sum())
    seats = cutoff - above_count
    if seats <= 0 or seats > tie_count:
        raise RuntimeError("Invalid boundary tie accounting")
    active_above = int(truth[above].sum())
    active_in_tie = int(truth[tied].sum())
    active_at_cutoff = active_above + (seats / tie_count) * active_in_tie
    realized_fraction = cutoff / count
    enrichment = (active_at_cutoff / active_total) / realized_fraction
    return {
        "candidate_count": count,
        "active_count": active_total,
        "cutoff_k": cutoff,
        "realized_screened_fraction": realized_fraction,
        "boundary_score": boundary,
        "strictly_above_boundary": above_count,
        "boundary_tie_count": tie_count,
        "boundary_tie_seats": seats,
        "actives_at_cutoff_fractional": float(active_at_cutoff),
        "ef1": float(enrichment),
    }


def tie_averaged_bedroc(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    alpha: float = 20.0,
) -> float:
    """BEDROC with each tie block assigned its expected exponential rank mass."""
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if values.ndim != 1 or truth.shape != values.shape or values.size == 0:
        raise ValueError("Scores and labels must be aligned, nonempty vectors")
    if not np.isfinite(values).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("BEDROC requires finite scores and binary labels")
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("BEDROC alpha must be positive")
    count = int(values.size)
    active_total = int(truth.sum())
    if active_total == 0:
        return 0.0
    if active_total == count:
        return 1.0
    order = np.argsort(-values, kind="mergesort")
    ordered_scores = values[order]
    ordered_labels = truth[order]
    sum_exp = 0.0
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        active_in_block = int(ordered_labels[start:stop].sum())
        if active_in_block:
            positions = np.arange(start + 1, stop + 1, dtype=np.float64)
            expected_mass = float(np.exp(-alpha * positions / count).mean())
            sum_exp += active_in_block * expected_mass
        start = stop
    denominator = (1.0 / count) * (
        (1.0 - math.exp(-alpha)) / math.expm1(alpha / count)
    )
    rie = sum_exp / (active_total * denominator)
    ratio = active_total / count
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (
        ratio * (1.0 - math.exp(-alpha))
    )
    rie_min = (1.0 - math.exp(alpha * ratio)) / (
        ratio * (1.0 - math.exp(alpha))
    )
    return float((rie - rie_min) / (rie_max - rie_min))


def compute_metrics(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    ef_fraction: float = 0.01,
    bedroc_alpha: float = 20.0,
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if truth.sum() <= 0 or truth.sum() >= truth.size:
        raise ValueError("Retrieval metrics require both candidate labels")
    ef = fractional_ef_at_fraction(values, truth, fraction=ef_fraction)
    return {
        **ef,
        "bedroc20": tie_averaged_bedroc(values, truth, alpha=bedroc_alpha),
        "roc_auc": float(roc_auc_score(truth, values)),
        "average_precision": float(average_precision_score(truth, values)),
    }

