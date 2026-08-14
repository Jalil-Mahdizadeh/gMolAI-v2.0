#!/usr/bin/env python3
"""Assemble the all-representation common panel without changing TDC roles."""

from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    identity_set_sha256,
    load_json,
    load_protocol,
    read_labels_tsv,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
    write_labels_tsv,
    write_panel_tsv,
)


def split_key(endpoint: str, seed: int, role: str) -> str:
    return f"{endpoint}__seed{seed:02d}__{role}"


def metric_defined(task: str, values: np.ndarray) -> bool:
    if len(values) < 2 or not np.isfinite(values).all():
        return False
    if task == "classification":
        return set(values.astype(int)) == {0, 1}
    return len(np.unique(values)) >= 2


def remap(values: np.ndarray, old_to_new: dict[int, int]) -> np.ndarray:
    return np.asarray(
        [old_to_new[int(value)] for value in values if int(value) in old_to_new],
        dtype=np.int64,
    )


def main() -> None:
    protocol = load_protocol()
    prepared = load_json(BENCHMARK_DIR / "inputs" / "prepared_manifest.json")
    full_panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "full_panel.tsv")
    full_labels = read_labels_tsv(BENCHMARK_DIR / "inputs" / "full_labels.tsv")
    if prepared["full_panel"]["sha256"] != sha256_file(
        BENCHMARK_DIR / "inputs" / "full_panel.tsv"
    ):
        raise RuntimeError("Prepared full panel changed")
    if prepared["full_labels"]["sha256"] != sha256_file(
        BENCHMARK_DIR / "inputs" / "full_labels.tsv"
    ):
        raise RuntimeError("Prepared labels changed")

    screen_models = tuple(protocol["comparators"]["screen_models"])
    accepted_by_model: dict[str, set[int]] = {
        "gmolai": set(range(len(full_panel)))
    }
    reports: dict[str, dict[str, Any]] = {}
    full_panel_hash = sha256_file(BENCHMARK_DIR / "inputs" / "full_panel.tsv")
    for model in screen_models:
        report = load_json(BENCHMARK_DIR / "state" / f"{model}-screen.json")
        if report.get("status") != "ok" or report.get("model") != model:
            raise RuntimeError(f"Invalid support screen for {model}")
        if report.get("input_sha256") != full_panel_hash:
            raise RuntimeError(f"{model} screened a different panel")
        accepted = [int(value) for value in report["accepted_indices"]]
        if accepted != sorted(set(accepted)) or any(
            value < 0 or value >= len(full_panel) for value in accepted
        ):
            raise RuntimeError(f"Invalid accepted indices for {model}")
        accepted_by_model[model] = set(accepted)
        reports[model] = report

    common_old = set(range(len(full_panel)))
    for accepted in accepted_by_model.values():
        common_old &= accepted
    common_indices = sorted(common_old)
    if not common_indices:
        raise RuntimeError("All-representation common panel is empty")
    old_panel_to_new = {old: new for new, old in enumerate(common_indices)}

    common_panel: list[dict[str, Any]] = []
    for new, old in enumerate(common_indices):
        row = dict(full_panel[old])
        row["panel_index"] = new
        common_panel.append(row)

    common_labels: list[dict[str, Any]] = []
    for row in full_labels:
        old_panel = int(row["panel_index"])
        if old_panel not in old_panel_to_new:
            continue
        updated = dict(row)
        updated["occurrence_index"] = len(common_labels)
        updated["panel_index"] = old_panel_to_new[old_panel]
        updated["original_panel_index"] = old_panel
        common_labels.append(updated)

    panel_path = BENCHMARK_DIR / "inputs" / "common_panel.tsv"
    labels_path = BENCHMARK_DIR / "inputs" / "common_labels.tsv"
    write_panel_tsv(panel_path, common_panel)
    write_labels_tsv(labels_path, common_labels)

    coverage_rows: list[dict[str, Any]] = []
    endpoint_order = tuple(protocol["data"]["endpoint_order"])
    for model in protocol["comparators"]["model_order"]:
        rejected_reason = {
            int(row["panel_index"]): row["reason"]
            for row in reports.get(model, {}).get("rejections", [])
        }
        for endpoint in (*endpoint_order, "all"):
            occurrences = [
                row for row in full_labels if endpoint == "all" or row["endpoint"] == endpoint
            ]
            unique = {int(row["panel_index"]) for row in occurrences}
            accepted_unique = unique & accepted_by_model[model]
            accepted_occurrences = sum(
                int(row["panel_index"]) in accepted_by_model[model]
                for row in occurrences
            )
            reason_counts = Counter(
                rejected_reason[index]
                for index in unique
                if index in rejected_reason
            )
            coverage_rows.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "occurrences": len(occurrences),
                    "accepted_occurrences": accepted_occurrences,
                    "occurrence_coverage": accepted_occurrences / max(1, len(occurrences)),
                    "unique_identities": len(unique),
                    "accepted_unique_identities": len(accepted_unique),
                    "unique_identity_coverage": len(accepted_unique) / max(1, len(unique)),
                    "rejection_reasons": json.dumps(
                        dict(sorted(reason_counts.items())), sort_keys=True
                    ),
                }
            )
    atomic_write_csv(
        BENCHMARK_DIR / "outputs" / "coverage.csv",
        coverage_rows,
        (
            "endpoint",
            "model",
            "occurrences",
            "accepted_occurrences",
            "occurrence_coverage",
            "unique_identities",
            "accepted_unique_identities",
            "unique_identity_coverage",
            "rejection_reasons",
        ),
    )

    with np.load(
        BENCHMARK_DIR / "inputs" / "full_split_indices.npz", allow_pickle=False
    ) as source:
        full_splits = {key: source[key] for key in source.files}
    common_splits: dict[str, np.ndarray] = {}
    endpoint_manifest: dict[str, Any] = {}
    for endpoint in endpoint_order:
        full_endpoint = [row for row in full_labels if row["endpoint"] == endpoint]
        common_endpoint = [row for row in common_labels if row["endpoint"] == endpoint]
        old_local_to_new: dict[int, int] = {}
        new_local = 0
        for old_local, row in enumerate(full_endpoint):
            if int(row["panel_index"]) in old_panel_to_new:
                old_local_to_new[old_local] = new_local
                new_local += 1
        if new_local != len(common_endpoint):
            raise RuntimeError(f"Occurrence remapping failed for {endpoint}")
        split_manifest = []
        for seed in protocol["evaluation"]["seeds"]:
            arrays = {}
            for role in ("train", "valid", "train_val", "test"):
                key = split_key(endpoint, int(seed), role)
                arrays[role] = remap(full_splits[key], old_local_to_new)
                common_splits[key] = arrays[role]
            universe = set(range(len(common_endpoint)))
            if set(arrays["train"]) & set(arrays["valid"]):
                raise RuntimeError(f"Train/valid overlap for {endpoint} seed {seed}")
            if set(arrays["train"]) | set(arrays["valid"]) != set(arrays["train_val"]):
                raise RuntimeError(f"Train/valid no longer partitions train_val for {endpoint}")
            if set(arrays["train_val"]) & set(arrays["test"]):
                raise RuntimeError(f"Train/test occurrence overlap for {endpoint}")
            if set(arrays["train_val"]) | set(arrays["test"]) != universe:
                raise RuntimeError(f"Roles no longer partition {endpoint}")
            task = common_endpoint[0]["task"]
            targets = np.asarray([float(row["target"]) for row in common_endpoint])
            for role in ("train", "valid", "train_val", "test"):
                if not metric_defined(task, targets[arrays[role]]):
                    raise RuntimeError(
                        f"Undefined {role} metric for {endpoint} seed {seed}"
                    )
            train_scaffolds = {
                common_endpoint[int(i)]["scaffold"] for i in arrays["train"]
            }
            valid_scaffolds = {
                common_endpoint[int(i)]["scaffold"] for i in arrays["valid"]
            }
            if train_scaffolds & valid_scaffolds:
                raise RuntimeError(f"Scaffold leakage in validation for {endpoint}")
            split_manifest.append(
                {
                    "seed": int(seed),
                    "train_occurrences": len(arrays["train"]),
                    "valid_occurrences": len(arrays["valid"]),
                    "train_val_occurrences": len(arrays["train_val"]),
                    "test_occurrences": len(arrays["test"]),
                    "train_identity_set_sha256": identity_set_sha256(
                        common_endpoint[int(i)]["molecule_hash"] for i in arrays["train"]
                    ),
                    "valid_identity_set_sha256": identity_set_sha256(
                        common_endpoint[int(i)]["molecule_hash"] for i in arrays["valid"]
                    ),
                }
            )
        train_rows = [row for row in common_endpoint if row["source_role"] == "train_val"]
        test_rows = [row for row in common_endpoint if row["source_role"] == "test"]
        train_identities = {row["molecule_hash"] for row in train_rows}
        test_identities = {row["molecule_hash"] for row in test_rows}
        train_scaffolds = {row["scaffold"] for row in train_rows}
        test_scaffolds = {row["scaffold"] for row in test_rows}
        endpoint_manifest[endpoint] = {
            "full_policy_accepted_occurrences": len(full_endpoint),
            "common_occurrences": len(common_endpoint),
            "removed_occurrences": len(full_endpoint) - len(common_endpoint),
            "common_unique_identities": len({row["molecule_hash"] for row in common_endpoint}),
            "train_val_occurrences": len(train_rows),
            "test_occurrences": len(test_rows),
            "train_test_identity_overlap": len(train_identities & test_identities),
            "train_test_identity_overlap_sha256": identity_set_sha256(
                train_identities & test_identities
            ),
            "train_test_scaffold_overlap": len(train_scaffolds & test_scaffolds),
            "split_manifest": split_manifest,
        }

    split_path = BENCHMARK_DIR / "inputs" / "common_split_indices.npz"
    atomic_savez(split_path, common_splits)
    manifest = {
        "schema_version": 1,
        "status": "all_representation_common_panel_frozen",
        "models": protocol["comparators"]["model_order"],
        "full_unique_identities": len(full_panel),
        "common_unique_identities": len(common_panel),
        "removed_unique_identities": len(full_panel) - len(common_panel),
        "full_occurrences": len(full_labels),
        "common_occurrences": len(common_labels),
        "removed_occurrences": len(full_labels) - len(common_labels),
        "ordered_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in common_panel
        ),
        "identity_set_sha256": identity_set_sha256(
            row["molecule_hash"] for row in common_panel
        ),
        "endpoints": endpoint_manifest,
        "common_panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
        "common_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "split_indices": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "coverage": {
            "path": str(BENCHMARK_DIR / "outputs" / "coverage.csv"),
            "sha256": sha256_file(BENCHMARK_DIR / "outputs" / "coverage.csv"),
        },
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "common_manifest.json", manifest)
    print(
        json.dumps(
            {
                "full_unique": len(full_panel),
                "common_unique": len(common_panel),
                "full_occurrences": len(full_labels),
                "common_occurrences": len(common_labels),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
