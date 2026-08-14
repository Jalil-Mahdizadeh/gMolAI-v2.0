#!/usr/bin/env python3
"""Independently verify the completed TDC ADMET benchmark artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import statistics
from typing import Any

from benchmark_io import (
    BENCHMARK_DIR,
    load_json,
    load_protocol,
    protocol_digest,
    sha256_file,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12)


def read_csv(name: str) -> list[dict[str, str]]:
    with (BENCHMARK_DIR / "outputs" / name).open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def verify_checksum_ledger() -> int:
    ledger = BENCHMARK_DIR / "outputs" / "SHA256SUMS"
    entries: dict[str, str] = {}
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("  ", 1)
        require(len(fields) == 2, f"Malformed checksum line {number}")
        expected, relative_name = fields
        relative = Path(relative_name)
        require(
            len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"Invalid SHA-256 on line {number}",
        )
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe checksum path on line {number}",
        )
        require(relative_name not in entries, f"Duplicate checksum path: {relative_name}")
        target = BENCHMARK_DIR / relative
        require(target.is_file() and not target.is_symlink(), f"Missing artifact: {target}")
        require(sha256_file(target) == expected, f"Checksum mismatch: {relative_name}")
        entries[relative_name] = expected
    require(len(entries) == 36, f"Expected 36 checksum entries, found {len(entries)}")
    return len(entries)


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        for model in ordered[start:stop]:
            ranks[model] = average
        start = stop
    return ranks


def verify_endpoint_result(
    result: dict[str, Any],
    endpoint: str,
    spec: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    record = result["endpoints"][endpoint]["result"]
    values = [float(value) for value in record["primary"]["values"]]
    require(len(values) == 5 and all(math.isfinite(value) for value in values),
            f"Invalid primary values: {result['model']}/{endpoint}")
    require(close(record["primary"]["mean"], statistics.fmean(values)),
            f"Primary mean mismatch: {result['model']}/{endpoint}")
    require(close(record["primary"]["population_std"], statistics.pstdev(values)),
            f"Primary dispersion mismatch: {result['model']}/{endpoint}")
    require(record["official_metric"] == spec["metric"],
            f"Official metric mismatch: {result['model']}/{endpoint}")
    outcomes = record["per_seed"]
    require([row["seed"] for row in outcomes] == [1, 2, 3, 4, 5],
            f"Seed order mismatch: {result['model']}/{endpoint}")
    for outcome in outcomes:
        if spec["task"] == "regression":
            require(outcome["selected_alpha"] in protocol["evaluation"]["regression"]["alphas"],
                    f"Ridge grid violation: {result['model']}/{endpoint}")
        else:
            require(outcome["selected_c"] in protocol["evaluation"]["classification"]["c_values"],
                    f"Logistic grid violation: {result['model']}/{endpoint}")
        require(all(math.isfinite(float(value)) for value in outcome["test_metrics"].values()),
                f"Non-finite test metric: {result['model']}/{endpoint}")
    strict = record["strict_identity_disjoint_primary"]
    require(strict is not None, f"Missing identity-disjoint result: {result['model']}/{endpoint}")
    require(all(close(left, right) for left, right in zip(values, strict["values"])),
            f"Identity-disjoint result changed: {result['model']}/{endpoint}")


def main() -> None:
    protocol = load_protocol()
    completion = load_json(BENCHMARK_DIR / "state" / "COMPLETE.json")
    status = load_json(BENCHMARK_DIR / "state" / "status.json")
    preflight = load_json(BENCHMARK_DIR / "state" / "preflight.json")
    summary = load_json(BENCHMARK_DIR / "outputs" / "tdc_admet_summary.json")
    manifest = load_json(BENCHMARK_DIR / "inputs" / "common_manifest.json")
    digest = protocol_digest(protocol)

    require(completion.get("status") == "complete", "Completion seal is absent")
    require(status.get("status") == "complete" and status.get("stage") == "complete",
            "Pipeline status is not complete")
    require(preflight.get("status") == "pass", "Preflight did not pass")
    require(preflight["runtime"]["visible_gpu_count"] == 1, "Preflight GPU contract changed")
    require(completion["protocol_sha256"] == digest == summary["protocol_sha256"],
            "Protocol digest mismatch")
    require(sha256_file(BENCHMARK_DIR / "outputs" / "tdc_admet_summary.json")
            == completion["summary_sha256"], "Summary checksum mismatch")
    require(sha256_file(BENCHMARK_DIR / "outputs" / "SHA256SUMS")
            == completion["checksums_sha256"], "Ledger checksum mismatch")
    checksum_entries = verify_checksum_ledger()

    primary_models = list(protocol["comparators"]["model_order"])
    all_models = primary_models + ["descriptor_13"]
    endpoints = list(protocol["data"]["endpoint_order"])
    require(len(primary_models) == 7 and len(endpoints) == 22, "Panel dimensions changed")
    require(manifest["common_unique_identities"] == 43_730, "Common identity count changed")
    require(manifest["common_occurrences"] == 78_131, "Common occurrence count changed")
    for endpoint in endpoints:
        endpoint_manifest = manifest["endpoints"][endpoint]
        require(endpoint_manifest["train_test_identity_overlap"] == 0,
                f"Train/test identity overlap: {endpoint}")
        require(endpoint_manifest["train_test_scaffold_overlap"] == 0,
                f"Train/test scaffold overlap: {endpoint}")

    results: dict[str, dict[str, Any]] = {}
    for model in all_models:
        result = load_json(BENCHMARK_DIR / "outputs" / "results" / f"{model}.json")
        require(result.get("status") == "complete" and result.get("model") == model,
                f"Incomplete model result: {model}")
        require(set(result["endpoints"]) == set(endpoints), f"Endpoint set changed: {model}")
        require(result["common_panel"]["unique_identities"] == 43_730,
                f"Common panel changed: {model}")
        require(result["neural_encoder_training_or_finetuning"] is False,
                f"Neural training flag changed: {model}")
        for endpoint in endpoints:
            verify_endpoint_result(
                result, endpoint, protocol["data"]["endpoints"][endpoint], protocol
            )
        results[model] = result

    descriptor = load_json(
        BENCHMARK_DIR / "outputs" / "embeddings" / "descriptor_13.json"
    )
    amendment = protocol["runtime_amendments"][-1]
    require(descriptor["nan_values"] == amendment["expected_nan_values"] == 12,
            "Descriptor NaN count changed")
    require(descriptor["nan_unique_identities"] == 6, "Descriptor NaN identities changed")
    require(set(descriptor["nan_features"]) == set(amendment["allowed_nan_features"]),
            "Descriptor NaN features changed")

    endpoint_ranks: dict[str, dict[str, float]] = {}
    for endpoint in endpoints:
        metric = protocol["data"]["endpoints"][endpoint]["metric"]
        values = {
            model: float(results[model]["endpoints"][endpoint]["result"]["primary"]["mean"])
            for model in primary_models
        }
        endpoint_ranks[endpoint] = average_ranks(
            values if metric == "mae" else {model: -value for model, value in values.items()}
        )

    model_rows = {row["model"]: row for row in read_csv("model_summary.csv")}
    require(set(model_rows) == set(primary_models), "Model summary membership changed")
    excluded = set(protocol["selection_conditioning"]["direct_or_near_reuse_endpoints"])
    categories = protocol["evaluation"]["categories"]
    full_means: dict[str, float] = {}
    robust_means: dict[str, float] = {}
    for model in primary_models:
        full_categories = []
        robust_categories = []
        all_ranks = []
        for category in categories:
            category_endpoints = [
                endpoint for endpoint in endpoints
                if protocol["data"]["endpoints"][endpoint]["category"] == category
            ]
            robust_endpoints = [
                endpoint for endpoint in category_endpoints if endpoint not in excluded
            ]
            category_ranks = [endpoint_ranks[endpoint][model] for endpoint in category_endpoints]
            robust_ranks = [endpoint_ranks[endpoint][model] for endpoint in robust_endpoints]
            full_categories.append(statistics.fmean(category_ranks))
            robust_categories.append(statistics.fmean(robust_ranks))
            all_ranks.extend(category_ranks)
        full_means[model] = statistics.fmean(full_categories)
        robust_means[model] = statistics.fmean(robust_categories)
        row = model_rows[model]
        require(close(float(row["category_balanced_mean_rank"]), full_means[model]),
                f"Full summary mismatch: {model}")
        require(close(float(row["selection_robust_category_balanced_mean_rank"]),
                      robust_means[model]), f"Robust summary mismatch: {model}")
        require(int(row["rank_one_endpoints"]) == sum(rank == 1.0 for rank in all_ranks),
                f"Endpoint-win count mismatch: {model}")
        require(int(row["top_three_endpoints"]) == sum(rank <= 3.0 for rank in all_ranks),
                f"Top-three count mismatch: {model}")
    full_summary_ranks = average_ranks(full_means)
    robust_summary_ranks = average_ranks(robust_means)
    for model, row in model_rows.items():
        require(close(float(row["category_balanced_rank"]), full_summary_ranks[model]),
                f"Full model rank mismatch: {model}")
        require(close(float(row["selection_robust_category_balanced_rank"]),
                      robust_summary_ranks[model]), f"Robust model rank mismatch: {model}")

    require(len(read_csv("endpoint_primary_metrics.csv")) == 176,
            "Primary endpoint table row count changed")
    require(len(read_csv("all_metrics.csv")) == 600,
            "All-metrics table row count changed")
    require(len(read_csv("category_rank_summary.csv")) == 70,
            "Category table row count changed")
    require(len(read_csv("encoding_runtime_observed.csv")) == 8,
            "Runtime table row count changed")

    print(
        "PASS: tdc-admet-complete; "
        f"checksums={checksum_entries}; models={len(all_models)}; "
        f"endpoints={len(endpoints)}; seed_outcomes={len(all_models) * len(endpoints) * 5}"
    )


if __name__ == "__main__":
    main()
