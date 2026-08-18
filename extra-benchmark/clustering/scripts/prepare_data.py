#!/usr/bin/env python3
"""Prepare ClassyFire-25 and QMugs identity manifests without model outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import duckdb
from rdkit import Chem

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    canonicalize_external,
    load_protocol,
    require_regular_file,
    sha256_file,
    sha256_lines,
    write_csv,
    write_tsv,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from gmolai_retrain.chem import CanonicalMolecule  # noqa: E402
from gmolai_retrain.config import load_config  # noqa: E402


CLASSY_COLUMNS = (
    "panel_index", "molecule_hash", "canonical_smiles", "subclass",
    "source_index", "source_inchikey", "heavy_atom_count",
)
QMUGS_COLUMNS = (
    "panel_index", "molecule_hash", "canonical_smiles", "chembl_id",
    "conf_id", "heavy_atom_count", "DFT_TOTAL_ENERGY", "DFT_HOMO_ENERGY",
    "DFT_HOMO_LUMO_GAP", "DFT_DIPOLE_TOT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("both", "classyfire", "qmugs"), default="both"
    )
    return parser.parse_args()


def prepare_classyfire(protocol: dict[str, Any], cfg: dict[str, Any]) -> None:
    source = protocol["data"]["classyfire25"]
    path = BENCHMARK_DIR / source["path"]
    require_regular_file(path, source["sha256"])
    started = time.perf_counter()
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempted_by_class: Counter[str] = Counter()
    accepted_before_dedup_by_class: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ("inchikey", "SMILES", "Kingdom", "Superclass", "Class", "Subclass", "mass")
        if tuple(reader.fieldnames or ()) != expected:
            raise RuntimeError(f"Unexpected ClassyFire schema: {reader.fieldnames}")
        for source_index, source_row in enumerate(reader):
            subclass = source_row["Subclass"].strip()
            attempted_by_class[subclass] += 1
            canonical = canonicalize_external(source_row["SMILES"], cfg)
            if not isinstance(canonical, CanonicalMolecule):
                rejection_reasons[canonical.reason] += 1
                continue
            molecule = Chem.MolFromSmiles(canonical.smiles)
            if molecule is None:
                raise RuntimeError("Canonical ClassyFire SMILES failed to reparse")
            row = {
                "panel_index": -1,
                "molecule_hash": canonical.molecule_hash,
                "canonical_smiles": canonical.smiles,
                "subclass": subclass,
                "source_index": source_index,
                "source_inchikey": source_row["inchikey"],
                "heavy_atom_count": molecule.GetNumHeavyAtoms(),
            }
            by_identity[canonical.molecule_hash].append(row)
            accepted_before_dedup_by_class[subclass] += 1
    if sum(attempted_by_class.values()) != int(source["expected_rows"]):
        raise RuntimeError("ClassyFire source row count changed")
    conflicting: set[str] = set()
    duplicates = 0
    eligible: list[dict[str, Any]] = []
    for identity, occurrences in by_identity.items():
        labels = {row["subclass"] for row in occurrences}
        if len(labels) != 1:
            conflicting.add(identity)
            continue
        duplicates += len(occurrences) - 1
        eligible.append(min(occurrences, key=lambda row: int(row["source_index"])))
    eligible.sort(key=lambda row: row["molecule_hash"])
    for index, row in enumerate(eligible):
        row["panel_index"] = index
    output = BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_candidates.tsv"
    write_tsv(output, eligible, CLASSY_COLUMNS)
    eligible_by_class = Counter(row["subclass"] for row in eligible)
    classes = sorted(attempted_by_class)
    if len(classes) != int(source["expected_subclasses"]):
        raise RuntimeError("ClassyFire subclass count changed")
    class_rows = [
        {
            "subclass": label,
            "source_attempted": attempted_by_class[label],
            "accepted_before_deduplication": accepted_before_dedup_by_class[label],
            "eligible_unique": eligible_by_class[label],
        }
        for label in classes
    ]
    class_path = BENCHMARK_DIR / "audit" / "classyfire_preparation_by_subclass.csv"
    write_csv(class_path, class_rows, tuple(class_rows[0]))
    report = {
        "schema_version": 1,
        "status": "ok",
        "source": str(path),
        "source_sha256": sha256_file(path),
        "attempted": sum(attempted_by_class.values()),
        "canonicalization_rejected": sum(rejection_reasons.values()),
        "canonicalization_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_occurrences_removed": duplicates,
        "conflicting_label_identities_removed": len(conflicting),
        "eligible_unique": len(eligible),
        "eligible_identity_sha256": sha256_lines(row["molecule_hash"] for row in eligible),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "by_subclass": str(class_path),
        "by_subclass_sha256": sha256_file(class_path),
        "explicit_hydrogens_removed_before_identity": True,
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audit" / "classyfire_preparation.json", report)
    print(json.dumps(report, sort_keys=True))


def qmugs_reader(path: Path):
    escaped = str(path).replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads=16")
    connection.execute("SET preserve_insertion_order=false")
    query = f"""
        SELECT chembl_id, conf_id, smiles, heavy_atoms,
               DFT_TOTAL_ENERGY, DFT_HOMO_ENERGY,
               DFT_HOMO_LUMO_GAP, DFT_DIPOLE_TOT
        FROM read_csv_auto('{escaped}', header=true, sample_size=-1)
        WHERE coalesce(significant_negative_wavenumbers, true) = false
          AND coalesce(nonunique_smiles, true) = false
          AND isfinite(DFT_TOTAL_ENERGY)
          AND isfinite(DFT_HOMO_ENERGY)
          AND isfinite(DFT_HOMO_LUMO_GAP)
          AND isfinite(DFT_DIPOLE_TOT)
          AND DFT_DIPOLE_TOT >= 0
        QUALIFY row_number() OVER (
          PARTITION BY chembl_id ORDER BY DFT_TOTAL_ENERGY ASC, conf_id ASC
        ) = 1
        ORDER BY chembl_id
    """
    return connection, connection.execute(query).fetch_record_batch(rows_per_batch=8192)


def prepare_qmugs(protocol: dict[str, Any], cfg: dict[str, Any]) -> None:
    source = protocol["data"]["qmugs"]
    path = BENCHMARK_DIR / source["path"]
    require_regular_file(path, source["sha256"])
    started = time.perf_counter()
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_reasons: Counter[str] = Counter()
    selected_conformers = 0
    heavy_atom_mismatches = 0
    connection, batches = qmugs_reader(path)
    try:
        for batch in batches:
            columns = batch.to_pydict()
            for values in zip(*(columns[name] for name in columns)):
                source_row = dict(zip(columns, values))
                selected_conformers += 1
                canonical = canonicalize_external(str(source_row["smiles"]), cfg)
                if not isinstance(canonical, CanonicalMolecule):
                    rejection_reasons[canonical.reason] += 1
                    continue
                numeric = {
                    key: float(source_row[key])
                    for key in (
                        "DFT_TOTAL_ENERGY", "DFT_HOMO_ENERGY",
                        "DFT_HOMO_LUMO_GAP", "DFT_DIPOLE_TOT",
                    )
                }
                if not all(math.isfinite(value) for value in numeric.values()):
                    rejection_reasons["nonfinite_after_selection"] += 1
                    continue
                molecule = Chem.MolFromSmiles(canonical.smiles)
                if molecule is None:
                    raise RuntimeError("Canonical QMugs SMILES failed to reparse")
                heavy_atoms = int(molecule.GetNumHeavyAtoms())
                if int(source_row["heavy_atoms"]) != heavy_atoms:
                    heavy_atom_mismatches += 1
                row = {
                    "panel_index": -1,
                    "molecule_hash": canonical.molecule_hash,
                    "canonical_smiles": canonical.smiles,
                    "chembl_id": str(source_row["chembl_id"]),
                    "conf_id": str(source_row["conf_id"]),
                    "heavy_atom_count": heavy_atoms,
                    **{key: format(value, ".17g") for key, value in numeric.items()},
                }
                by_identity[canonical.molecule_hash].append(row)
    finally:
        connection.close()
    conflicting: set[str] = set()
    duplicates = 0
    eligible: list[dict[str, Any]] = []
    for identity, occurrences in by_identity.items():
        chembl_ids = {row["chembl_id"] for row in occurrences}
        if len(chembl_ids) != 1:
            conflicting.add(identity)
            continue
        duplicates += len(occurrences) - 1
        eligible.append(
            min(
                occurrences,
                key=lambda row: (float(row["DFT_TOTAL_ENERGY"]), row["conf_id"]),
            )
        )
    eligible.sort(key=lambda row: row["molecule_hash"])
    for index, row in enumerate(eligible):
        row["panel_index"] = index
    output = BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_eligible.tsv"
    write_tsv(output, eligible, QMUGS_COLUMNS)
    initial = int(protocol["common_support"]["qmugs_initial_attempt"])
    attempt_rows = [dict(row, panel_index=index) for index, row in enumerate(eligible[:initial])]
    attempt = BENCHMARK_DIR / "inputs" / "prepared" / f"qmugs_attempt_{initial:06d}.tsv"
    write_tsv(attempt, attempt_rows, QMUGS_COLUMNS)
    report = {
        "schema_version": 1,
        "status": "ok",
        "source": str(path),
        "source_sha256": sha256_file(path),
        "source_rows": int(source["expected_rows"]),
        "source_chembl_ids": int(source["expected_chembl_ids"]),
        "selected_lowest_energy_conformers_after_row_filters": selected_conformers,
        "canonicalization_rejected": sum(rejection_reasons.values()),
        "canonicalization_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_same_chembl_identity_occurrences_removed": duplicates,
        "cross_chembl_conflicting_identities_removed": len(conflicting),
        "source_vs_recomputed_heavy_atom_mismatches": heavy_atom_mismatches,
        "eligible_unique": len(eligible),
        "eligible_identity_sha256": sha256_lines(row["molecule_hash"] for row in eligible),
        "eligible_output": str(output),
        "eligible_output_sha256": sha256_file(output),
        "initial_attempt_rows": len(attempt_rows),
        "initial_attempt": str(attempt),
        "initial_attempt_sha256": sha256_file(attempt),
        "explicit_hydrogens_removed_before_identity": True,
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audit" / "qmugs_preparation.json", report)
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    cfg = load_config(REPOSITORY_ROOT / protocol["gmolai"]["config"]["path"])
    if args.dataset in ("both", "classyfire"):
        prepare_classyfire(protocol, cfg)
    if args.dataset in ("both", "qmugs"):
        prepare_qmugs(protocol, cfg)


if __name__ == "__main__":
    main()

