"""Pure metric selection and paired-bootstrap functions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    unit: str = "fraction"
    denominator_description: str = "all seeds"


def select_reranked(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    """Select one candidate per seed with the frozen target-blind ordering."""
    subset = candidates.loc[candidates["first_proposal_rank"] <= int(budget)].copy()
    if subset.empty:
        return subset
    ordered = subset.sort_values(
        [
            "query_position",
            "latent_relative_l2_to_seed_condition",
            "latent_cosine_to_seed_condition",
            "first_proposal_rank",
            "canonical_smiles",
        ],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates("query_position", keep="first")


def metric_estimate(frame: pd.DataFrame, spec: MetricSpec) -> float:
    if spec.kind == "mean":
        values = frame[str(spec.column)].to_numpy(dtype=np.float64)
        return float(np.mean(values))
    if spec.kind == "nanmean":
        values = frame[str(spec.column)].to_numpy(dtype=np.float64)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    if spec.kind == "ratio":
        numerator = frame[str(spec.numerator)].to_numpy(dtype=np.float64).sum()
        denominator = frame[str(spec.denominator)].to_numpy(dtype=np.float64).sum()
        return float(numerator / denominator) if denominator > 0 else float("nan")
    raise ValueError(f"Unknown metric kind: {spec.kind}")


def paired_bootstrap(
    frame: pd.DataFrame,
    specs: list[MetricSpec],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
    batch_resamples: int = 40,
) -> pd.DataFrame:
    """Return deterministic paired seed-bootstrap intervals for metric specs."""
    if len(frame) == 0:
        raise ValueError("Cannot bootstrap an empty frame")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    alpha = (1.0 - float(confidence_level)) / 2.0
    n = len(frame)
    rng = np.random.default_rng(int(seed))
    estimates = {spec.name: np.empty(resamples, dtype=np.float64) for spec in specs}
    offset = 0
    while offset < resamples:
        current = min(batch_resamples, resamples - offset)
        indices = rng.integers(0, n, size=(current, n), dtype=np.int32)
        for spec in specs:
            if spec.kind in {"mean", "nanmean"}:
                values = frame[str(spec.column)].to_numpy(dtype=np.float64)
                sampled = values[indices]
                if spec.kind == "mean":
                    result = sampled.mean(axis=1)
                else:
                    result = np.nanmean(sampled, axis=1)
            elif spec.kind == "ratio":
                numerator = frame[str(spec.numerator)].to_numpy(dtype=np.float64)
                denominator = frame[str(spec.denominator)].to_numpy(dtype=np.float64)
                num = numerator[indices].sum(axis=1)
                den = denominator[indices].sum(axis=1)
                result = np.divide(
                    num,
                    den,
                    out=np.full(current, np.nan, dtype=np.float64),
                    where=den > 0,
                )
            else:
                raise ValueError(f"Unknown metric kind: {spec.kind}")
            estimates[spec.name][offset : offset + current] = result
        offset += current

    rows = []
    for spec in specs:
        values = estimates[spec.name]
        finite = values[np.isfinite(values)]
        if not len(finite):
            low = high = float("nan")
        else:
            low, high = np.quantile(finite, [alpha, 1.0 - alpha])
        rows.append(
            {
                "metric": spec.name,
                "estimate": metric_estimate(frame, spec),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "confidence_level": float(confidence_level),
                "bootstrap_resamples": int(resamples),
                "unit": spec.unit,
                "denominator": spec.denominator_description,
            }
        )
    return pd.DataFrame(rows)


def derived_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)
