#!/usr/bin/env python3
"""Freeze all-model common-support manifests after label-blind adapter screening."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    panel_columns,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
    write_csv,
    write_tsv,
)


SCREENED_MODELS = ("morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")
ALL_MODELS = ("gmolai", *SCREENED_MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("classyfire", "qmugs"))
    parser.add_argument("--attempt-size", type=int)
    return parser.parse_args()


def load_screens(tag: str, panel: Path, rows: list[dict[str, str]]):
    input_hash = sha256_file(panel)
    accepted: dict[str, set[int]] = {"gmolai": set(range(len(rows)))}
    reports: dict[str, dict[str, Any]] = {}
    for model in SCREENED_MODELS:
        path = BENCHMARK_DIR / "artifacts" / "screens" / f"{model}-{tag}.json"
        report = load_json(path)
        if report.get("status") != "ok" or report.get("model") != model:
            raise RuntimeError(f"Invalid screen report: {path}")
        if report.get("input_sha256") != input_hash or int(report.get("rows", -1)) != len(rows):
            raise RuntimeError(f"Screen input binding differs: {path}")
        indices = [int(value) for value in report.get("accepted_indices", [])]
        index_set = set(indices)
        if len(indices) != len(index_set) or any(value < 0 or value >= len(rows) for value in index_set):
            raise RuntimeError(f"Invalid accepted-index set: {path}")
        if len(index_set) != int(report.get("accepted", -1)):
            raise RuntimeError(f"Accepted count differs: {path}")
        accepted[model] = index_set
        reports[model] = report
    return accepted, reports


def coverage_rows(
    rows: list[dict[str, str]], accepted: dict[str, set[int]], common: set[int]
) -> list[dict[str, Any]]:
    result = []
    for model in ALL_MODELS:
        count = len(accepted[model])
        result.append({
            "model": model,
            "attempted": len(rows),
            "accepted": count,
            "rejected": len(rows) - count,
            "coverage_fraction": count / max(1, len(rows)),
            "all_model_common": len(common),
            "missing_after_common_support": count - len(common),
            "nonfinite_vectors": 0,
            "zero_norm_vectors": 0,
        })
    return result


def assemble_classyfire(protocol: dict[str, Any]) -> None:
    panel = BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_candidates.tsv"
    rows = read_panel_tsv(panel)
    accepted, reports = load_screens("classyfire", panel, rows)
    common = set.intersection(*accepted.values())
    labels = sorted({row["subclass"] for row in rows})
    if len(labels) != 25:
        raise RuntimeError("Prepared structural panel does not contain 25 subclasses")
    common_counts = Counter(rows[index]["subclass"] for index in common)
    m = min(common_counts[label] for label in labels)
    if m <= int(protocol["structural_evaluation"]["geometry_metric"]["k"]):
        raise RuntimeError("Balanced ClassyFire subclasses are too small for leave-self-out kNN")
    selected_indices: set[int] = set()
    for label in labels:
        candidates = sorted(
            (index for index in common if rows[index]["subclass"] == label),
            key=lambda index: rows[index]["molecule_hash"],
        )
        selected_indices.update(candidates[:m])
    selected_source = sorted(selected_indices, key=lambda index: rows[index]["molecule_hash"])
    final_rows = [dict(rows[index], panel_index=output_index) for output_index, index in enumerate(selected_source)]
    output = BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv"
    write_tsv(output, final_rows, panel_columns(panel))
    coverage = coverage_rows(rows, accepted, common)
    coverage_path = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_coverage.csv"
    write_csv(coverage_path, coverage, tuple(coverage[0]))
    subclass_rows = []
    for label in labels:
        attempted_indices = {index for index, row in enumerate(rows) if row["subclass"] == label}
        for model in ALL_MODELS:
            count = len(attempted_indices & accepted[model])
            subclass_rows.append({
                "subclass": label, "model": model,
                "attempted": len(attempted_indices), "accepted": count,
                "rejected": len(attempted_indices) - count,
                "all_model_common_before_balance": common_counts[label],
                "frozen_balanced_rows": m,
            })
    subclass_path = BENCHMARK_DIR / "outputs" / "tables" / "classyfire_coverage_by_subclass.csv"
    write_csv(subclass_path, subclass_rows, tuple(subclass_rows[0]))
    report = {
        "schema_version": 1, "status": "ok", "benchmark": "classyfire",
        "attempted_rows": len(rows), "all_model_common_before_balance": len(common),
        "balance_per_subclass": m, "final_rows": len(final_rows),
        "subclasses": labels, "common_counts_before_balance": dict(common_counts),
        "source_panel": str(panel), "source_panel_sha256": sha256_file(panel),
        "final_panel": str(output), "final_panel_sha256": sha256_file(output),
        "final_identity_sha256": sha256_lines(row["molecule_hash"] for row in final_rows),
        "coverage_table": str(coverage_path), "coverage_table_sha256": sha256_file(coverage_path),
        "subclass_coverage_table": str(subclass_path),
        "subclass_coverage_table_sha256": sha256_file(subclass_path),
        "screen_reports": {model: reports[model]["input_sha256"] for model in SCREENED_MODELS},
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "classyfire_common.json", report)
    print(json.dumps(report, sort_keys=True))


def assemble_qmugs(protocol: dict[str, Any], attempt_size: int | None) -> None:
    if attempt_size is None:
        raise ValueError("--attempt-size is required for QMugs")
    rules = protocol["common_support"]
    initial = int(rules["qmugs_initial_attempt"])
    increment = int(rules["qmugs_increment"])
    maximum = int(rules["qmugs_maximum_attempt"])
    if attempt_size < initial or attempt_size > maximum or (attempt_size - initial) % increment:
        raise ValueError("QMugs attempt size violates the frozen expansion sequence")
    panel = BENCHMARK_DIR / "inputs" / "prepared" / f"qmugs_attempt_{attempt_size:06d}.tsv"
    rows = read_panel_tsv(panel)
    if len(rows) != attempt_size:
        raise RuntimeError("QMugs attempt panel length differs from its name")
    tag = f"qmugs_{attempt_size:06d}"
    accepted, reports = load_screens(tag, panel, rows)
    common = set.intersection(*accepted.values())
    target = int(rules["qmugs_target_common"])
    if len(common) < target and attempt_size < maximum:
        report = {
            "schema_version": 1, "status": "expansion_required",
            "attempt_size": attempt_size, "common_rows": len(common),
            "next_attempt_size": attempt_size + increment,
        }
        atomic_write_json(BENCHMARK_DIR / "state" / "qmugs_expansion_required.json", report)
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(3)
    selected_count = min(target, len(common))
    selected_source = sorted(common, key=lambda index: rows[index]["molecule_hash"])[:selected_count]
    final_rows = [dict(rows[index], panel_index=output_index) for output_index, index in enumerate(selected_source)]
    output = BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv"
    write_tsv(output, final_rows, panel_columns(panel))
    coverage = coverage_rows(rows, accepted, common)
    coverage_path = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_coverage.csv"
    write_csv(coverage_path, coverage, tuple(coverage[0]))
    report = {
        "schema_version": 1,
        "status": "ok" if len(common) >= target else "coverage_failure_disclosed",
        "benchmark": "qmugs", "attempted_rows": len(rows),
        "all_model_common_before_truncation": len(common), "target_rows": target,
        "final_rows": len(final_rows), "coverage_failure": len(common) < target,
        "source_panel": str(panel), "source_panel_sha256": sha256_file(panel),
        "final_panel": str(output), "final_panel_sha256": sha256_file(output),
        "final_identity_sha256": sha256_lines(row["molecule_hash"] for row in final_rows),
        "coverage_table": str(coverage_path), "coverage_table_sha256": sha256_file(coverage_path),
        "screen_reports": {model: reports[model]["input_sha256"] for model in SCREENED_MODELS},
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "qmugs_common.json", report)
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if args.benchmark == "classyfire":
        assemble_classyfire(protocol)
    else:
        assemble_qmugs(protocol, args.attempt_size)


if __name__ == "__main__":
    main()
