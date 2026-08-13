#!/usr/bin/env python3
"""Build the all-model common panel while preserving frozen split roles."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
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
from prepare_datasets import array_key


def remap(values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray(
        [mapping[int(value)] for value in values if int(value) in mapping],
        dtype=np.int64,
    )


def role_digest(dataset_rows: list[dict[str, str]], indices: np.ndarray) -> str:
    return identity_set_sha256(
        dataset_rows[int(index)]["molecule_hash"] for index in indices
    )


def validate_split(
    dataset: str,
    task: str,
    rows: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    outer: int,
) -> dict[str, Any]:
    train = arrays[array_key(dataset, outer, "train")]
    test = arrays[array_key(dataset, outer, "test")]
    universe = set(range(len(rows)))
    if set(train) & set(test) or set(train) | set(test) != universe:
        raise RuntimeError(f"{dataset} outer {outer} does not partition the common panel")
    train_groups = {rows[int(index)]["scaffold_group"] for index in train}
    test_groups = {rows[int(index)]["scaffold_group"] for index in test}
    if train_groups & test_groups:
        raise RuntimeError(f"{dataset} outer {outer} leaks scaffold groups")
    targets = np.asarray([float(row["target"]) for row in rows])
    if task == "classification":
        for role, indices in (("train", train), ("test", test)):
            if set(targets[indices].astype(int)) != {0, 1}:
                raise RuntimeError(f"{dataset} outer {outer} {role} lost a class")
    inner_manifest = []
    for fold in range(3):
        fit = arrays[array_key(dataset, outer, "fit", fold)]
        validation = arrays[array_key(dataset, outer, "validation", fold)]
        if set(fit) & set(validation) or set(fit) | set(validation) != set(train):
            raise RuntimeError(f"{dataset} outer {outer} inner {fold} changed roles")
        fit_groups = {rows[int(index)]["scaffold_group"] for index in fit}
        validation_groups = {
            rows[int(index)]["scaffold_group"] for index in validation
        }
        if fit_groups & validation_groups:
            raise RuntimeError(f"{dataset} outer {outer} inner {fold} leaks scaffolds")
        if task == "classification":
            for role, indices in (("fit", fit), ("validation", validation)):
                if set(targets[indices].astype(int)) != {0, 1}:
                    raise RuntimeError(
                        f"{dataset} outer {outer} inner {fold} {role} lost a class"
                    )
        inner_manifest.append(
            {
                "fold": fold,
                "fit_molecules": int(len(fit)),
                "validation_molecules": int(len(validation)),
                "fit_identity_set_sha256": role_digest(rows, fit),
                "validation_identity_set_sha256": role_digest(rows, validation),
            }
        )
    return {
        "outer_index": outer,
        "train_molecules": int(len(train)),
        "test_molecules": int(len(test)),
        "train_scaffold_groups": len(train_groups),
        "test_scaffold_groups": len(test_groups),
        "train_identity_set_sha256": role_digest(rows, train),
        "test_identity_set_sha256": role_digest(rows, test),
        "inner_folds": inner_manifest,
    }


def main() -> None:
    protocol = load_protocol()
    manifest = load_json(BENCHMARK_DIR / "inputs" / "dataset_manifest.json")
    full_panel = read_panel_tsv(BENCHMARK_DIR / "inputs" / "full_panel.tsv")
    full_labels = read_labels_tsv(BENCHMARK_DIR / "inputs" / "full_labels.tsv")
    if len(full_panel) != len(full_labels) or len(full_panel) != int(manifest["rows"]):
        raise RuntimeError("Full panel and labels disagree")
    for panel, label in zip(full_panel, full_labels):
        if panel["molecule_hash"] != label["molecule_hash"]:
            raise RuntimeError("Full panel/label identity order changed")

    screen_models = tuple(protocol["comparators"]["screen_models"])
    accepted_by_model: dict[str, set[int]] = {"gmolai": set(range(len(full_panel)))}
    reports: dict[str, dict[str, Any]] = {}
    for model in screen_models:
        report = load_json(BENCHMARK_DIR / "state" / f"{model}-screen.json")
        if report.get("status") != "ok" or report.get("model") != model:
            raise RuntimeError(f"Invalid coverage screen for {model}")
        if report.get("input_sha256") != sha256_file(
            BENCHMARK_DIR / "inputs" / "full_panel.tsv"
        ):
            raise RuntimeError(f"{model} screen was run on a different panel")
        accepted = [int(value) for value in report["accepted_indices"]]
        if accepted != sorted(set(accepted)) or any(
            value < 0 or value >= len(full_panel) for value in accepted
        ):
            raise RuntimeError(f"{model} screen has invalid accepted indices")
        accepted_by_model[model] = set(accepted)
        reports[model] = report

    common = set(range(len(full_panel)))
    for accepted in accepted_by_model.values():
        common &= accepted
    common_indices = sorted(common)
    if not common_indices:
        raise RuntimeError("All-model common panel is empty")

    common_panel: list[dict[str, Any]] = []
    common_labels: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(common_indices):
        panel = dict(full_panel[old_index])
        label = dict(full_labels[old_index])
        panel["panel_index"] = new_index
        label["panel_index"] = new_index
        label["original_panel_index"] = old_index
        common_panel.append(panel)
        common_labels.append(label)
    common_panel_path = BENCHMARK_DIR / "inputs" / "common_panel.tsv"
    common_labels_path = BENCHMARK_DIR / "inputs" / "common_labels.tsv"
    write_panel_tsv(common_panel_path, common_panel)
    write_labels_tsv(common_labels_path, common_labels)

    coverage_rows: list[dict[str, Any]] = []
    dataset_order = tuple(protocol["datasets"]["order"])
    for model in ("gmolai", *screen_models):
        rejected_reasons = {
            int(row["panel_index"]): row["reason"]
            for row in reports.get(model, {}).get("rejections", [])
        }
        for dataset in (*dataset_order, "all"):
            indices = [
                index
                for index, row in enumerate(full_labels)
                if dataset == "all" or row["dataset"] == dataset
            ]
            accepted = sum(index in accepted_by_model[model] for index in indices)
            reason_counts = Counter(
                rejected_reasons[index]
                for index in indices
                if index in rejected_reasons
            )
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "rows": len(indices),
                    "accepted": accepted,
                    "rejected": len(indices) - accepted,
                    "coverage_fraction": accepted / max(1, len(indices)),
                    "rejection_reasons": json.dumps(
                        dict(sorted(reason_counts.items())), sort_keys=True
                    ),
                }
            )
    coverage_path = BENCHMARK_DIR / "outputs" / "coverage.csv"
    atomic_write_csv(
        coverage_path,
        coverage_rows,
        (
            "dataset",
            "model",
            "rows",
            "accepted",
            "rejected",
            "coverage_fraction",
            "rejection_reasons",
        ),
    )

    with np.load(
        BENCHMARK_DIR / "inputs" / "full_split_indices.npz", allow_pickle=False
    ) as source:
        full_splits = {key: source[key] for key in source.files}
    common_splits: dict[str, np.ndarray] = {}
    dataset_manifest: dict[str, Any] = {}
    for dataset in dataset_order:
        full_dataset_rows = [row for row in full_labels if row["dataset"] == dataset]
        common_dataset_rows = [
            row for row in common_labels if row["dataset"] == dataset
        ]
        allowed_old = {int(row["dataset_index"]) for row in common_dataset_rows}
        old_to_new = {
            int(row["dataset_index"]): new
            for new, row in enumerate(common_dataset_rows)
        }
        if len(old_to_new) != len(common_dataset_rows):
            raise RuntimeError(f"{dataset} common dataset indices are not unique")
        for outer in range(int(protocol["evaluation"]["outer_splits"])):
            for role in ("train", "test"):
                key = array_key(dataset, outer, role)
                common_splits[key] = remap(full_splits[key], old_to_new)
            for fold in range(3):
                for role in ("fit", "validation"):
                    key = array_key(dataset, outer, role, fold)
                    common_splits[key] = remap(full_splits[key], old_to_new)
        split_manifest = [
            validate_split(
                dataset,
                common_dataset_rows[0]["task"],
                common_dataset_rows,
                common_splits,
                outer,
            )
            for outer in range(int(protocol["evaluation"]["outer_splits"]))
        ]
        dataset_manifest[dataset] = {
            "full_rows": len(full_dataset_rows),
            "common_rows": len(common_dataset_rows),
            "removed_rows": len(full_dataset_rows) - len(common_dataset_rows),
            "common_identity_set_sha256": identity_set_sha256(
                row["molecule_hash"] for row in common_dataset_rows
            ),
            "split_assignment_rule": (
                "intersection with frozen full-panel roles; no reassignment"
            ),
            "split_identity_manifest": split_manifest,
            "accepted_original_dataset_indices": sorted(allowed_old),
        }

    common_split_path = BENCHMARK_DIR / "inputs" / "common_split_indices.npz"
    atomic_savez(common_split_path, common_splits)
    common_manifest = {
        "schema_version": 1,
        "status": "all_model_common_panel_frozen",
        "models": ["gmolai", *screen_models],
        "rows": len(common_panel),
        "removed_rows": len(full_panel) - len(common_panel),
        "ordered_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in common_panel
        ),
        "identity_occurrences_not_globally_deduplicated": True,
        "panel": {
            "path": str(common_panel_path),
            "sha256": sha256_file(common_panel_path),
        },
        "labels": {
            "path": str(common_labels_path),
            "sha256": sha256_file(common_labels_path),
        },
        "split_indices": {
            "path": str(common_split_path),
            "sha256": sha256_file(common_split_path),
        },
        "coverage": {"path": str(coverage_path), "sha256": sha256_file(coverage_path)},
        "datasets": dataset_manifest,
    }
    atomic_write_json(
        BENCHMARK_DIR / "inputs" / "common_manifest.json", common_manifest
    )
    print(
        json.dumps(
            {
                "full_rows": len(full_panel),
                "common_rows": len(common_panel),
                "datasets": {
                    key: value["common_rows"] for key, value in dataset_manifest.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
