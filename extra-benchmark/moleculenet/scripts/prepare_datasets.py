#!/usr/bin/env python3
"""Reconstruct the frozen MoleculeNet/HIV molecules and split assignments."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_savez,
    atomic_write_json,
    identity_set_sha256,
    load_json,
    load_protocol,
    sha256_file,
    sha256_lines,
    write_labels_tsv,
    write_panel_tsv,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gmolai_retrain.config import load_config  # noqa: E402
from gmolai_retrain.downstream import (  # noqa: E402
    MOLECULENET_DATASETS,
    _inner_group_folds,
    _prepare_dataset_records,
    _scaffold_splits,
)
from gmolai_retrain.downstream_audit import _split_identity_manifest  # noqa: E402


def array_key(dataset: str, outer: int, role: str, fold: int | None = None) -> str:
    if fold is None:
        return f"{dataset}__outer{outer:02d}__{role}"
    return f"{dataset}__outer{outer:02d}__inner{fold}__{role}"


def reference_splits(
    dataset_name: str,
    reference: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray, int]],
) -> None:
    value = reference["datasets"].get(dataset_name)
    if value is None:
        raise RuntimeError(f"Reference benchmark lacks {dataset_name}")
    observed = [
        (int(seed), int(len(train)), int(len(test))) for train, test, seed in splits
    ]
    feature_results = value["feature_results"]
    feature = (
        "molecule_embedding"
        if "molecule_embedding" in feature_results
        else next(iter(feature_results))
    )
    expected = [
        (int(row["outer_seed"]), int(row["train"]), int(row["test"]))
        for row in feature_results[feature]["per_split"]
    ]
    if observed != expected:
        raise RuntimeError(f"{dataset_name} outer split seeds/counts changed")


def main() -> None:
    protocol = load_protocol()
    dataset_order = tuple(protocol["datasets"]["order"])
    cfg = load_config(REPOSITORY_ROOT / protocol["sources"]["config"]["path"])
    if cfg["_config_hash"] != protocol["sources"]["config"]["config_hash"]:
        raise RuntimeError("Canonical configuration hash changed")
    if cfg["_descriptor_schema_hash"] != protocol["sources"]["config"]["descriptor_schema_hash"]:
        raise RuntimeError("Descriptor schema hash changed")

    development_reference = load_json(
        REPOSITORY_ROOT / protocol["references"]["development"]["path"]
    )
    hiv_reference = load_json(REPOSITORY_ROOT / protocol["references"]["hiv"]["path"])
    descriptor_reference = load_json(
        REPOSITORY_ROOT / protocol["references"]["descriptor_control"]["path"]
    )

    panel_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    split_arrays: dict[str, np.ndarray] = {}
    manifest_datasets: dict[str, Any] = {}
    datasets_dir = REPOSITORY_ROOT / protocol["datasets"]["directory"]

    for dataset_name in dataset_order:
        spec = MOLECULENET_DATASETS[dataset_name]
        expected_source = protocol["datasets"]["sources"][dataset_name]
        source = datasets_dir / spec["filename"]
        if source.name != expected_source["filename"]:
            raise RuntimeError(f"{dataset_name} source filename changed")
        if sha256_file(source) != expected_source["sha256"]:
            raise RuntimeError(f"{dataset_name} source hash changed")

        dataset = _prepare_dataset_records(source, spec, cfg)
        if len(set(dataset.molecule_hashes)) != len(dataset.molecule_hashes):
            raise RuntimeError(f"{dataset_name} has duplicate canonical identity hashes")
        by_hash: dict[str, str] = {}
        for molecule_hash, smiles in zip(dataset.molecule_hashes, dataset.canonical_smiles):
            previous = by_hash.setdefault(molecule_hash, smiles)
            if previous != smiles:
                raise RuntimeError(f"SHA-256 collision in {dataset_name}")

        reference = hiv_reference if dataset_name == "hiv" else development_reference
        reference_dataset = reference["datasets"][dataset_name]
        for field in (
            "molecules",
            "scaffold_groups",
            "duplicate_rows_collapsed",
            "conflicting_duplicate_rows_rejected",
        ):
            if dataset.preparation.get(field) != reference_dataset["preparation"].get(field):
                raise RuntimeError(f"{dataset_name} preparation mismatch for {field}")

        split_count = int(protocol["evaluation"]["outer_splits"])
        splits = _scaffold_splits(
            dataset.scaffold_groups,
            dataset.targets,
            task=spec["task"],
            count=split_count,
            seed=int(protocol["evaluation"]["seed"]),
        )
        reference_splits(dataset_name, reference, splits)
        split_identity = _split_identity_manifest(dataset, splits, task=spec["task"])
        if dataset_name in descriptor_reference["datasets"]:
            frozen_identity = descriptor_reference["datasets"][dataset_name][
                "split_reconstruction"
            ]["identity_manifest"]
            if split_identity != frozen_identity:
                raise RuntimeError(
                    f"{dataset_name} outer/inner split identities differ from the frozen audit"
                )

        global_start = len(panel_rows)
        for dataset_index, (
            smiles,
            molecule_hash,
            source_bucket,
            scaffold_group,
            target,
        ) in enumerate(
            zip(
                dataset.canonical_smiles,
                dataset.molecule_hashes,
                dataset.source_buckets,
                dataset.scaffold_groups,
                dataset.targets,
            )
        ):
            panel_index = len(panel_rows)
            panel_rows.append(
                {
                    "panel_index": panel_index,
                    "graph_id": f"{dataset_name}:{dataset_index}",
                    "source_bucket": int(source_bucket),
                    "molecule_hash": molecule_hash,
                    "canonical_smiles": smiles,
                    "scaffold": str(scaffold_group),
                }
            )
            label_rows.append(
                {
                    "panel_index": panel_index,
                    "original_panel_index": panel_index,
                    "dataset": dataset_name,
                    "dataset_index": dataset_index,
                    "molecule_hash": molecule_hash,
                    "target": format(float(target), ".17g"),
                    "task": spec["task"],
                    "scaffold_group": str(scaffold_group),
                }
            )

        for outer, (train, test, seed) in enumerate(splits):
            split_arrays[array_key(dataset_name, outer, "train")] = train.astype(np.int64)
            split_arrays[array_key(dataset_name, outer, "test")] = test.astype(np.int64)
            inner = _inner_group_folds(
                train,
                dataset.scaffold_groups,
                dataset.targets,
                task=spec["task"],
                seed=int(seed) + 1,
            )
            for fold, (fit, validation) in enumerate(inner):
                split_arrays[array_key(dataset_name, outer, "fit", fold)] = fit.astype(np.int64)
                split_arrays[array_key(dataset_name, outer, "validation", fold)] = validation.astype(np.int64)

        manifest_datasets[dataset_name] = {
            "task": spec["task"],
            "source": {"path": str(source), "sha256": sha256_file(source)},
            "preparation": dataset.preparation,
            "global_panel_start": global_start,
            "global_panel_stop": len(panel_rows),
            "ordered_identity_sha256": sha256_lines(dataset.molecule_hashes),
            "identity_set_sha256": identity_set_sha256(dataset.molecule_hashes),
            "split_identity_manifest": split_identity,
        }

    panel_path = BENCHMARK_DIR / "inputs" / "full_panel.tsv"
    labels_path = BENCHMARK_DIR / "inputs" / "full_labels.tsv"
    split_path = BENCHMARK_DIR / "inputs" / "full_split_indices.npz"
    write_panel_tsv(panel_path, panel_rows)
    write_labels_tsv(labels_path, label_rows)
    atomic_savez(split_path, split_arrays)
    manifest = {
        "schema_version": 1,
        "status": "frozen_source_splits_reconstructed",
        "datasets": manifest_datasets,
        "dataset_order": list(dataset_order),
        "rows": len(panel_rows),
        "ordered_identity_sha256": sha256_lines(row["molecule_hash"] for row in panel_rows),
        "full_panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
        "full_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "split_indices": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "reference_validation": (
            "Five development datasets match the rev4 identity audit exactly; HIV "
            "matches the authoritative accepted outer seeds and molecule counts."
        ),
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "dataset_manifest.json", manifest)
    print(json.dumps({"rows": len(panel_rows), "datasets": {k: v["preparation"]["molecules"] for k, v in manifest_datasets.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
