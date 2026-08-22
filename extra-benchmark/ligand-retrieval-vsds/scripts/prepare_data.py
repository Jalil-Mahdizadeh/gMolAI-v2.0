#!/usr/bin/env python3
"""Prepare and audit the 71-target TrueDecoy_gap panel without model outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

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


XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EXPECTED_HEADERS = (
    "Entry",
    "Reviewed",
    "Entry Name",
    "Organism",
    "AlphaFoldDB",
    "PDB entry",
    "Target class",
    "active_num",
    "inactive_num",
    "min_activity_act (Ki, Kd, IC50 nM)",
    "max_activity_ina (Ki, Kd, IC50 nM)",
)
TARGET_COLUMNS = (
    "target_id",
    "reviewed",
    "entry_name",
    "organism",
    "alphafold_id",
    "pdb_entry",
    "target_class",
    "source_active_count",
    "source_inactive_count",
    "least_potent_active_nM",
    "most_potent_inactive_nM",
)
MOLECULE_COLUMNS = (
    "panel_index",
    "molecule_hash",
    "canonical_smiles",
    "nonisomeric_smiles",
    "inchikey",
    "scaffold",
    "atom_count",
)
MEMBERSHIP_COLUMNS = (
    "membership_index",
    "target_id",
    "label",
    "molecule_hash",
    "canonical_smiles",
    "nonisomeric_smiles",
    "inchikey",
    "scaffold",
    "atom_count",
    "source_record_name",
    "source_record_index",
)


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def read_gap_workbook(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(XLSX_NS + "t"))
            for item in strings_root.findall(XLSX_NS + "si")
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    matrix: list[list[str]] = []
    for row in sheet.findall(".//" + XLSX_NS + "row"):
        values = [""] * len(EXPECTED_HEADERS)
        for cell in row.findall(XLSX_NS + "c"):
            index = _column_index(str(cell.get("r", "")))
            if index >= len(values):
                continue
            value_node = cell.find(XLSX_NS + "v")
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            values[index] = value
        matrix.append(values)
    if tuple(matrix[0]) != EXPECTED_HEADERS:
        raise RuntimeError(f"Unexpected TrueDecoy_gap workbook schema: {matrix[0]}")
    rows = []
    for values in matrix[1:]:
        row = dict(zip(EXPECTED_HEADERS, values))
        rows.append(
            {
                "target_id": row["Entry"],
                "reviewed": row["Reviewed"],
                "entry_name": row["Entry Name"],
                "organism": row["Organism"],
                "alphafold_id": row["AlphaFoldDB"],
                "pdb_entry": row["PDB entry"],
                "target_class": row["Target class"],
                "source_active_count": str(int(float(row["active_num"]))),
                "source_inactive_count": str(int(float(row["inactive_num"]))),
                "least_potent_active_nM": row[
                    "min_activity_act (Ki, Kd, IC50 nM)"
                ],
                "most_potent_inactive_nM": row[
                    "max_activity_ina (Ki, Kd, IC50 nM)"
                ],
            }
        )
    return rows


def canonicalize_record(molecule: Chem.Mol, cfg: dict[str, Any]):
    try:
        without_hydrogens = Chem.RemoveHs(molecule, sanitize=True)
        smiles = Chem.MolToSmiles(
            without_hydrogens, canonical=True, isomericSmiles=True
        )
    except Exception:
        return None, "explicit_hydrogen_removal_failure"
    canonical = canonicalize_external(smiles, cfg)
    if not isinstance(canonical, CanonicalMolecule):
        return None, canonical.reason
    normalized = Chem.MolFromSmiles(canonical.smiles)
    if normalized is None:
        return None, "canonical_reparse_failure"
    try:
        inchikey = Chem.MolToInchiKey(normalized)
    except Exception:
        return None, "inchikey_failure"
    if not inchikey:
        return None, "inchikey_failure"
    return (canonical, inchikey), None


def prepare_target(
    target: dict[str, str], cfg: dict[str, Any], source_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_id = target["target_id"]
    occurrences: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    raw_by_label: Counter[str] = Counter()
    accepted_before_dedup: Counter[str] = Counter()
    for label, filename, expected_key in (
        ("active", "actives.sdf", "source_active_count"),
        ("inactive_or_lower_affinity", "inactives.sdf", "source_inactive_count"),
    ):
        path = source_root / target_id / filename
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing extracted source SDF: {path}")
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        if len(supplier) != int(target[expected_key]):
            raise RuntimeError(
                f"Source record count differs for {target_id}/{filename}: "
                f"{len(supplier)} != {target[expected_key]}"
            )
        for source_index, molecule in enumerate(supplier):
            raw_by_label[label] += 1
            if molecule is None:
                rejection_reasons[f"{label}:sdf_parse_failure"] += 1
                continue
            value, reason = canonicalize_record(molecule, cfg)
            if value is None:
                rejection_reasons[f"{label}:{reason}"] += 1
                continue
            canonical, inchikey = value
            source_name = (
                molecule.GetProp("_Name").strip()
                if molecule.HasProp("_Name")
                else ""
            )
            occurrences.append(
                {
                    "target_id": target_id,
                    "label": label,
                    "molecule_hash": canonical.molecule_hash,
                    "canonical_smiles": canonical.smiles,
                    "nonisomeric_smiles": canonical.nonisomeric_smiles,
                    "inchikey": inchikey,
                    "scaffold": canonical.scaffold,
                    "atom_count": canonical.atom_count,
                    "source_record_name": source_name,
                    "source_record_index": source_index,
                }
            )
            accepted_before_dedup[label] += 1
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_inchikey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occurrences:
        by_hash[row["molecule_hash"]].append(row)
        by_inchikey[row["inchikey"]].append(row)
    conflicting_hashes = {
        key for key, rows in by_hash.items() if len({row["label"] for row in rows}) > 1
    }
    conflicting_inchikeys = {
        key
        for key, rows in by_inchikey.items()
        if len({row["label"] for row in rows}) > 1
    }
    nonconflicting = [
        row
        for row in occurrences
        if row["molecule_hash"] not in conflicting_hashes
        and row["inchikey"] not in conflicting_inchikeys
    ]
    by_label_inchikey: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in nonconflicting:
        by_label_inchikey[(row["label"], row["inchikey"])].append(row)
    retained = [
        min(
            rows,
            key=lambda row: (
                row["molecule_hash"],
                row["source_record_name"],
                int(row["source_record_index"]),
            ),
        )
        for rows in by_label_inchikey.values()
    ]
    by_retained_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        by_retained_hash[row["molecule_hash"]].append(row)
    final_rows = [
        min(
            rows,
            key=lambda row: (
                row["inchikey"],
                row["source_record_name"],
                int(row["source_record_index"]),
            ),
        )
        for rows in by_retained_hash.values()
    ]
    if any(len({row["label"] for row in rows}) > 1 for rows in by_retained_hash.values()):
        raise RuntimeError(f"Label contradiction survived deduplication for {target_id}")
    final_rows.sort(key=lambda row: (row["label"], row["molecule_hash"]))
    final_counts = Counter(row["label"] for row in final_rows)
    report = {
        "target_id": target_id,
        "target_class": target["target_class"],
        "raw_active": raw_by_label["active"],
        "raw_inactive_or_lower_affinity": raw_by_label[
            "inactive_or_lower_affinity"
        ],
        "canonical_active_before_dedup": accepted_before_dedup["active"],
        "canonical_inactive_before_dedup": accepted_before_dedup[
            "inactive_or_lower_affinity"
        ],
        "conflicting_canonical_identities": len(conflicting_hashes),
        "conflicting_full_inchikeys": len(conflicting_inchikeys),
        "duplicate_occurrences_removed": len(nonconflicting) - len(final_rows),
        "prepared_active": final_counts["active"],
        "prepared_inactive_or_lower_affinity": final_counts[
            "inactive_or_lower_affinity"
        ],
        "canonicalization_rejected": sum(rejection_reasons.values()),
        "rejection_reasons_json": json.dumps(
            dict(sorted(rejection_reasons.items())), sort_keys=True
        ),
    }
    return final_rows, report


def main() -> None:
    started = time.perf_counter()
    protocol = load_protocol()
    archive = protocol["data"]["archive"]
    gap = protocol["data"]["gap_definition"]
    archive_path = BENCHMARK_DIR / archive["path"]
    workbook_path = BENCHMARK_DIR / gap["path"]
    require_regular_file(archive_path, archive["sha256"], archive["bytes"])
    require_regular_file(workbook_path, gap["sha256"], gap["bytes"])
    targets = read_gap_workbook(workbook_path)
    if len(targets) != int(gap["expected_targets"]):
        raise RuntimeError("TrueDecoy_gap target count changed")
    if len({row["target_id"] for row in targets}) != len(targets):
        raise RuntimeError("Duplicate target identifier in TrueDecoy_gap workbook")
    if sum(int(row["source_active_count"]) for row in targets) != int(
        gap["expected_active_records"]
    ):
        raise RuntimeError("Workbook active total changed")
    if sum(int(row["source_inactive_count"]) for row in targets) != int(
        gap["expected_inactive_or_lower_affinity_records"]
    ):
        raise RuntimeError("Workbook inactive/lower-affinity total changed")
    cfg = load_config(REPOSITORY_ROOT / protocol["gmolai"]["config"]["path"])
    source_root = (
        BENCHMARK_DIR
        / "inputs/raw/extracted/VSDS_vd/TrueDecoy set"
    )
    memberships: list[dict[str, Any]] = []
    target_reports: list[dict[str, Any]] = []
    for target in targets:
        rows, report = prepare_target(target, cfg, source_root)
        memberships.extend(rows)
        target_reports.append(report)
    memberships.sort(
        key=lambda row: (row["target_id"], row["label"], row["molecule_hash"])
    )
    for index, row in enumerate(memberships):
        row["membership_index"] = index
    molecule_by_hash: dict[str, dict[str, Any]] = {}
    for row in memberships:
        candidate = {
            "panel_index": -1,
            "molecule_hash": row["molecule_hash"],
            "canonical_smiles": row["canonical_smiles"],
            "nonisomeric_smiles": row["nonisomeric_smiles"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
            "atom_count": row["atom_count"],
        }
        existing = molecule_by_hash.setdefault(row["molecule_hash"], candidate)
        if any(existing[key] != candidate[key] for key in candidate if key != "panel_index"):
            raise RuntimeError("Cross-target molecule metadata contradiction")
    molecules = sorted(molecule_by_hash.values(), key=lambda row: row["molecule_hash"])
    for index, row in enumerate(molecules):
        row["panel_index"] = index
    prepared = BENCHMARK_DIR / "inputs" / "prepared"
    target_path = prepared / "targets.tsv"
    membership_path = prepared / "memberships_candidates.tsv"
    molecule_path = prepared / "molecule_candidates.tsv"
    write_tsv(target_path, targets, TARGET_COLUMNS)
    write_tsv(membership_path, memberships, MEMBERSHIP_COLUMNS)
    write_tsv(molecule_path, molecules, MOLECULE_COLUMNS)
    by_target_path = BENCHMARK_DIR / "audits" / "data_preparation_by_target.csv"
    write_csv(by_target_path, target_reports, tuple(target_reports[0]))
    result = {
        "schema_version": 1,
        "status": "ok",
        "dataset": "VSDS-vd v3 TrueDecoy_gap",
        "archive_sha256": sha256_file(archive_path),
        "gap_workbook_sha256": sha256_file(workbook_path),
        "targets": len(targets),
        "raw_active_records": sum(int(row["source_active_count"]) for row in targets),
        "raw_inactive_or_lower_affinity_records": sum(
            int(row["source_inactive_count"]) for row in targets
        ),
        "prepared_memberships": len(memberships),
        "prepared_unique_molecules": len(molecules),
        "prepared_active_memberships": sum(row["label"] == "active" for row in memberships),
        "prepared_inactive_or_lower_affinity_memberships": sum(
            row["label"] == "inactive_or_lower_affinity" for row in memberships
        ),
        "targets_path": str(target_path),
        "targets_sha256": sha256_file(target_path),
        "memberships_path": str(membership_path),
        "memberships_sha256": sha256_file(membership_path),
        "molecules_path": str(molecule_path),
        "molecules_sha256": sha256_file(molecule_path),
        "ordered_molecule_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in molecules
        ),
        "by_target_path": str(by_target_path),
        "by_target_sha256": sha256_file(by_target_path),
        "label_contradictions_retained": 0,
        "explicit_hydrogens_removed_before_identity": True,
        "performance_inspected": False,
        "wall_seconds": time.perf_counter() - started,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audits" / "data_preparation.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

