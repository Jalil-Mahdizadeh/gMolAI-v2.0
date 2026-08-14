#!/usr/bin/env python3
"""Freeze the pre-result overlap audit against prior development endpoints."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, sha256_file


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs are invalid")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm)


def target_sets(
    rows: list[dict[str, str]], field: str, value: str
) -> dict[str, set[float]]:
    result: dict[str, set[float]] = {}
    for row in rows:
        if row[field] == value:
            result.setdefault(row["molecule_hash"], set()).add(float(row["target"]))
    return result


def main() -> None:
    protocol = load_protocol()
    tdc_path = BENCHMARK_DIR / "inputs" / "full_labels.tsv"
    prior_path = BENCHMARK_DIR.parent / "moleculenet" / "inputs" / "full_labels.tsv"
    tdc = read_rows(tdc_path)
    prior = read_rows(prior_path)
    development = ("bace", "bbbp", "esol", "freesolv", "lipophilicity")
    prior_sets = {
        dataset: {
            row["molecule_hash"] for row in prior if row["dataset"] == dataset
        }
        for dataset in development
    }
    endpoints: dict[str, Any] = {}
    development_union = set().union(*prior_sets.values())
    for endpoint in protocol["data"]["endpoint_order"]:
        identities = {
            row["molecule_hash"] for row in tdc if row["endpoint"] == endpoint
        }
        endpoints[endpoint] = {
            "tdc_unique_identities": len(identities),
            "overlap_unique_identities": {
                dataset: len(identities & prior_sets[dataset])
                for dataset in development
            },
            "overlap_with_any_development_identity": len(
                identities & development_union
            ),
        }

    direct_pairs = {
        "bbb_martins": "bbbp",
        "lipophilicity_astrazeneca": "lipophilicity",
        "solubility_aqsoldb": "esol",
    }
    direct: dict[str, Any] = {}
    for endpoint, dataset in direct_pairs.items():
        tdc_targets = target_sets(tdc, "endpoint", endpoint)
        prior_targets = target_sets(prior, "dataset", dataset)
        common = sorted(set(tdc_targets) & set(prior_targets))
        comparable = [
            value
            for value in common
            if len(tdc_targets[value]) == 1 and len(prior_targets[value]) == 1
        ]
        left = [next(iter(tdc_targets[value])) for value in comparable]
        right = [next(iter(prior_targets[value])) for value in comparable]
        direct[endpoint] = {
            "prior_dataset": dataset,
            "overlap_unique_identities": len(common),
            "unambiguous_target_pairs": len(comparable),
            "conflicting_tdc_target_identities_in_overlap": sum(
                len(tdc_targets[value]) > 1 for value in common
            ),
            "exact_target_matches": sum(x == y for x, y in zip(left, right)),
            "maximum_absolute_target_difference": max(
                abs(x - y) for x, y in zip(left, right)
            ),
            "target_pearson": pearson(left, right),
            "interpretation": (
                "exact endpoint reuse among unambiguous overlapping identities"
                if all(x == y for x, y in zip(left, right))
                else "strongly related endpoint subset"
            ),
        }

    artifact = {
        "schema_version": 1,
        "status": "frozen_before_representation_execution",
        "purpose": "selection-conditioning audit; no TDC model output was available",
        "tdc_full_labels": {"path": str(tdc_path), "sha256": sha256_file(tdc_path)},
        "prior_moleculenet_full_labels": {
            "path": str(prior_path),
            "sha256": sha256_file(prior_path),
        },
        "prior_development_datasets": list(development),
        "endpoints": endpoints,
        "direct_or_near_endpoint_reuse": direct,
        "predeclared_selection_robust_exclusions": list(direct_pairs),
        "rule": (
            "Report all 22 endpoints and the panel-complete primary summary; also "
            "report a 19-endpoint category-balanced sensitivity excluding the three "
            "directly reused or near-reused development endpoints."
        ),
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "prior_development_overlap.json", artifact)


if __name__ == "__main__":
    main()
