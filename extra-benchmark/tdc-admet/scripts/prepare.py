#!/usr/bin/env python3
"""Fetch, verify, prepare, and split the pinned TDC ADMET snapshot."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path, PurePosixPath
from random import Random
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_savez,
    atomic_write_json,
    identity_set_sha256,
    load_protocol,
    sha256_file,
    sha256_lines,
    write_labels_tsv,
    write_panel_tsv,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gmolai_retrain.chem import CanonicalMolecule, canonicalize  # noqa: E402
from gmolai_retrain.config import load_config  # noqa: E402


def split_key(endpoint: str, seed: int, role: str) -> str:
    return f"{endpoint}__seed{seed:02d}__{role}"


def read_source(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("Drug_ID", "Drug", "Y"):
            raise RuntimeError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path} has {len(rows)} rows; expected {expected_rows}")
    return rows


def safe_fetch(protocol: dict[str, Any]) -> dict[str, Any]:
    data = protocol["data"]
    source_root = REPOSITORY_ROOT / "work" / "tdc-admet" / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    archive = source_root / data["archive_filename"]
    if archive.exists():
        if archive.is_symlink() or sha256_file(archive) != data["archive_sha256"]:
            raise RuntimeError(f"Existing archive is unsafe or has the wrong hash: {archive}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=archive.name + ".", suffix=".partial", dir=source_root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                with urllib.request.urlopen(data["download_url"], timeout=120) as response:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                    handle.flush()
            if temporary.stat().st_size != int(data["archive_bytes"]):
                raise RuntimeError("Downloaded TDC archive byte count changed")
            if sha256_file(temporary) != data["archive_sha256"]:
                raise RuntimeError("Downloaded TDC archive hash changed")
            temporary.replace(archive)
        except Exception:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            finally:
                raise

    destination = source_root / "admet_group"
    if not destination.exists():
        temporary_root = Path(tempfile.mkdtemp(prefix="extract-", dir=source_root))
        try:
            expected_csv = {
                f"admet_group/{endpoint}/{role}.csv"
                for endpoint in data["endpoint_order"]
                for role in ("train_val", "test")
            }
            observed_csv: set[str] = set()
            with tarfile.open(archive, "r:gz") as handle:
                for member in handle.getmembers():
                    pure = PurePosixPath(member.name)
                    if pure.is_absolute() or ".." in pure.parts:
                        raise RuntimeError(f"Unsafe archive member: {member.name}")
                    if member.issym() or member.islnk() or member.isdev():
                        raise RuntimeError(f"Unsupported archive member: {member.name}")
                    if member.isdir() or member.name == "admet_group/.DS_Store":
                        continue
                    if not member.isfile() or member.name not in expected_csv:
                        raise RuntimeError(f"Unexpected archive member: {member.name}")
                    source = handle.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Could not read archive member: {member.name}")
                    target = temporary_root.joinpath(*pure.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    observed_csv.add(member.name)
            if observed_csv != expected_csv:
                missing = sorted(expected_csv - observed_csv)
                raise RuntimeError(f"TDC archive is incomplete: {missing}")
            temporary_root.joinpath("admet_group").replace(destination)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    files: dict[str, Any] = {}
    for endpoint in data["endpoint_order"]:
        spec = data["endpoints"][endpoint]
        files[endpoint] = {}
        for role in ("train_val", "test"):
            path = destination / endpoint / f"{role}.csv"
            expected_hash = spec[f"{role}_sha256"]
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise RuntimeError(f"Source hash changed for {endpoint}/{role}")
            rows = read_source(path, int(spec[f"{role}_rows"]))
            files[endpoint][role] = {
                "path": str(path),
                "sha256": observed_hash,
                "rows": len(rows),
                "columns": ["Drug_ID", "Drug", "Y"],
            }
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "source": {
            "title": data["title"],
            "doi": data["doi"],
            "snapshot_created_with": data["snapshot_created_with"],
            "audit_repository": data["audit_repository"],
            "audit_repository_commit": data["audit_repository_commit"],
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
        },
        "endpoint_order": data["endpoint_order"],
        "files": files,
        "total_occurrences": sum(
            files[endpoint][role]["rows"]
            for endpoint in data["endpoint_order"]
            for role in ("train_val", "test")
        ),
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "source_manifest.json", manifest)
    return manifest


def tdc_scaffold_split(smiles: list[str], seed: int) -> tuple[set[int], set[int]]:
    """Exact membership reproduction of PyTDC 0.4.1 create_scaffold_split."""
    scaffolds: dict[str, set[int]] = {}
    error_count = 0
    for index, value in enumerate(smiles):
        try:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError("RDKit parse failure")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule, includeChirality=False
            )
            scaffolds.setdefault(scaffold, set()).add(index)
        except Exception:
            error_count += 1
    usable = len(smiles) - error_count
    train_size = int(usable * 0.875)
    validation_size = int(usable * 0.125)
    test_size = usable - train_size - validation_size
    index_sets = list(scaffolds.values())
    big = [
        values
        for values in index_sets
        if len(values) > validation_size / 2 or len(values) > test_size / 2
    ]
    small = [values for values in index_sets if values not in big]
    random = Random(seed)
    random.shuffle(big)
    random.shuffle(small)
    train: list[int] = []
    validation: list[int] = []
    for values in big + small:
        ordered = sorted(values)
        if len(train) + len(ordered) <= train_size:
            train.extend(ordered)
        else:
            validation.extend(ordered)
    if set(train) & set(validation) or len(train) + len(validation) != usable:
        raise RuntimeError("Local PyTDC scaffold split reproduction failed")
    return set(train), set(validation)


def scaffold(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return "PARSE_FAILURE"
    core = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule, includeChirality=False
    )
    return f"MURCKO:{core}"


def metric_defined(task: str, targets: list[float]) -> bool:
    if not targets or not all(math.isfinite(value) for value in targets):
        return False
    if task == "classification":
        return set(int(value) for value in targets) == {0, 1}
    return len(targets) >= 2 and len(set(targets)) >= 2


def main() -> None:
    protocol = load_protocol()
    source_manifest = safe_fetch(protocol)
    cfg = load_config(REPOSITORY_ROOT / protocol["gmolai"]["config"]["path"])
    if cfg["_config_hash"] != protocol["gmolai"]["config"]["config_hash"]:
        raise RuntimeError("Canonicalization configuration hash changed")
    policy = cfg["data"]["canonicalization"]

    identity_to_panel: dict[str, int] = {}
    identity_to_smiles: dict[str, str] = {}
    panel_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    split_arrays: dict[str, np.ndarray] = {}
    endpoint_manifests: dict[str, Any] = {}
    data = protocol["data"]
    source_dir = REPOSITORY_ROOT / data["source_directory"]

    for endpoint in data["endpoint_order"]:
        spec = data["endpoints"][endpoint]
        raw_by_role = {
            role: read_source(
                source_dir / endpoint / f"{role}.csv", int(spec[f"{role}_rows"])
            )
            for role in ("train_val", "test")
        }
        raw_splits = {
            seed: tdc_scaffold_split(
                [row["Drug"] for row in raw_by_role["train_val"]], seed
            )
            for seed in protocol["evaluation"]["seeds"]
        }
        endpoint_rows: list[dict[str, Any]] = []
        rejections: Counter[str] = Counter()
        role_accepted: Counter[str] = Counter()
        for role in ("train_val", "test"):
            for source_index, row in enumerate(raw_by_role[role]):
                try:
                    target = float(row["Y"])
                except (TypeError, ValueError):
                    rejections[f"{role}:invalid_target"] += 1
                    continue
                if not math.isfinite(target):
                    rejections[f"{role}:invalid_target"] += 1
                    continue
                if spec["task"] == "classification" and target not in (0.0, 1.0):
                    rejections[f"{role}:nonbinary_target"] += 1
                    continue
                canonical = canonicalize(
                    str(row["Drug"]),
                    isomeric_smiles=bool(policy["isomeric_smiles"]),
                    fragment_policy=str(policy["fragment_policy"]),
                    allowed_elements=set(policy["allowed_elements"]),
                    min_atoms=int(policy["min_atoms"]),
                    max_atoms=int(policy["max_atoms"]),
                    buckets=int(cfg["data"]["hash_buckets"]),
                    split_cfg=cfg["data"]["split"],
                )
                if not isinstance(canonical, CanonicalMolecule):
                    rejections[f"{role}:{canonical.reason}"] += 1
                    continue
                existing = identity_to_smiles.setdefault(
                    canonical.molecule_hash, canonical.smiles
                )
                if existing != canonical.smiles:
                    raise RuntimeError("SHA-256 identity collision")
                panel_index = identity_to_panel.get(canonical.molecule_hash)
                if panel_index is None:
                    panel_index = len(panel_rows)
                    identity_to_panel[canonical.molecule_hash] = panel_index
                    panel_rows.append(
                        {
                            "panel_index": panel_index,
                            "graph_id": f"tdc:{canonical.molecule_hash[:16]}",
                            "source_bucket": int(canonical.bucket),
                            "molecule_hash": canonical.molecule_hash,
                            "canonical_smiles": canonical.smiles,
                            "scaffold": (
                                f"SCAFFOLD:{canonical.scaffold}"
                                if canonical.scaffold
                                else f"ACYCLIC:{canonical.nonisomeric_smiles}"
                            ),
                        }
                    )
                endpoint_rows.append(
                    {
                        "occurrence_index": -1,
                        "panel_index": panel_index,
                        "original_panel_index": panel_index,
                        "endpoint": endpoint,
                        "source_role": role,
                        "source_row_index": source_index,
                        "drug_id": str(row["Drug_ID"]),
                        "molecule_hash": canonical.molecule_hash,
                        "target": format(target, ".17g"),
                        "task": spec["task"],
                        "official_metric": spec["metric"],
                        "category": spec["category"],
                        "scaffold": scaffold(str(row["Drug"])),
                    }
                )
                role_accepted[role] += 1

        endpoint_start = len(label_rows)
        for row in endpoint_rows:
            row["occurrence_index"] = len(label_rows)
            label_rows.append(row)
        train_val_local = np.asarray(
            [i for i, row in enumerate(endpoint_rows) if row["source_role"] == "train_val"],
            dtype=np.int64,
        )
        test_local = np.asarray(
            [i for i, row in enumerate(endpoint_rows) if row["source_role"] == "test"],
            dtype=np.int64,
        )
        split_manifest: list[dict[str, Any]] = []
        for seed in protocol["evaluation"]["seeds"]:
            raw_train, raw_validation = raw_splits[int(seed)]
            train = np.asarray(
                [
                    i
                    for i, row in enumerate(endpoint_rows)
                    if row["source_role"] == "train_val"
                    and int(row["source_row_index"]) in raw_train
                ],
                dtype=np.int64,
            )
            validation = np.asarray(
                [
                    i
                    for i, row in enumerate(endpoint_rows)
                    if row["source_role"] == "train_val"
                    and int(row["source_row_index"]) in raw_validation
                ],
                dtype=np.int64,
            )
            if set(train) & set(validation) or set(train) | set(validation) != set(train_val_local):
                raise RuntimeError(f"Filtered split roles changed for {endpoint} seed {seed}")
            train_targets = [float(endpoint_rows[int(i)]["target"]) for i in train]
            validation_targets = [
                float(endpoint_rows[int(i)]["target"]) for i in validation
            ]
            test_targets = [float(endpoint_rows[int(i)]["target"]) for i in test_local]
            if not all(
                metric_defined(spec["task"], values)
                for values in (train_targets, validation_targets, test_targets)
            ):
                raise RuntimeError(
                    f"Undefined metric after policy filtering: {endpoint} seed {seed}"
                )
            split_arrays[split_key(endpoint, int(seed), "train")] = train
            split_arrays[split_key(endpoint, int(seed), "valid")] = validation
            split_arrays[split_key(endpoint, int(seed), "train_val")] = train_val_local
            split_arrays[split_key(endpoint, int(seed), "test")] = test_local
            split_manifest.append(
                {
                    "seed": int(seed),
                    "train_occurrences": len(train),
                    "valid_occurrences": len(validation),
                    "test_occurrences": len(test_local),
                    "train_source_row_sha256": sha256_lines(
                        endpoint_rows[int(i)]["source_row_index"] for i in train
                    ),
                    "valid_source_row_sha256": sha256_lines(
                        endpoint_rows[int(i)]["source_row_index"] for i in validation
                    ),
                }
            )

        train_identities = {
            row["molecule_hash"] for row in endpoint_rows if row["source_role"] == "train_val"
        }
        test_identities = {
            row["molecule_hash"] for row in endpoint_rows if row["source_role"] == "test"
        }
        train_scaffolds = {
            row["scaffold"] for row in endpoint_rows if row["source_role"] == "train_val"
        }
        test_scaffolds = {
            row["scaffold"] for row in endpoint_rows if row["source_role"] == "test"
        }
        endpoint_manifests[endpoint] = {
            "category": spec["category"],
            "task": spec["task"],
            "official_metric": spec["metric"],
            "raw": {
                "train_val": len(raw_by_role["train_val"]),
                "test": len(raw_by_role["test"]),
            },
            "policy_accepted": dict(role_accepted),
            "policy_rejections": dict(sorted(rejections.items())),
            "occurrence_start": endpoint_start,
            "occurrence_stop": len(label_rows),
            "unique_identities": len({row["molecule_hash"] for row in endpoint_rows}),
            "train_val_unique_identities": len(train_identities),
            "test_unique_identities": len(test_identities),
            "train_test_identity_overlap": len(train_identities & test_identities),
            "train_test_identity_overlap_sha256": identity_set_sha256(
                train_identities & test_identities
            ),
            "train_test_scaffold_overlap": len(train_scaffolds & test_scaffolds),
            "split_manifest": split_manifest,
        }

    panel_path = BENCHMARK_DIR / "inputs" / "full_panel.tsv"
    labels_path = BENCHMARK_DIR / "inputs" / "full_labels.tsv"
    splits_path = BENCHMARK_DIR / "inputs" / "full_split_indices.npz"
    write_panel_tsv(panel_path, panel_rows)
    write_labels_tsv(labels_path, label_rows)
    atomic_savez(splits_path, split_arrays)
    audit_spec = protocol["selection_conditioning"]["audit"]
    audit_path = REPOSITORY_ROOT / audit_spec["path"]
    if sha256_file(audit_path) != audit_spec["sha256"]:
        raise RuntimeError("Prior development-overlap audit hash changed")
    if sha256_file(labels_path) != audit_spec["tdc_full_labels_sha256"]:
        raise RuntimeError("Prepared TDC identities no longer match the frozen overlap audit")
    manifest = {
        "schema_version": 1,
        "status": "policy_filtered_official_roles_frozen",
        "source_manifest_sha256": sha256_file(
            BENCHMARK_DIR / "inputs" / "source_manifest.json"
        ),
        "canonicalization": protocol["evaluation"]["canonicalization"],
        "source_occurrences": source_manifest["total_occurrences"],
        "accepted_occurrences": len(label_rows),
        "unique_panel_identities": len(panel_rows),
        "ordered_panel_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in panel_rows
        ),
        "endpoints": endpoint_manifests,
        "full_panel": {"path": str(panel_path), "sha256": sha256_file(panel_path)},
        "full_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "split_indices": {"path": str(splits_path), "sha256": sha256_file(splits_path)},
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "prepared_manifest.json", manifest)
    print(
        json.dumps(
            {
                "source_occurrences": source_manifest["total_occurrences"],
                "accepted_occurrences": len(label_rows),
                "unique_identities": len(panel_rows),
                "identity_overlap_endpoints": {
                    endpoint: value["train_test_identity_overlap"]
                    for endpoint, value in endpoint_manifests.items()
                    if value["train_test_identity_overlap"]
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
