"""Exception-safe application of the unchanged gMolAI chemistry policy."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem, RDLogger

from gmolai_retrain.chem import Rejection, canonicalize


RDLogger.DisableLog("rdApp.*")
_POLICY_CONFIG: dict[str, Any] | None = None


def _initialize(resolved_config: dict[str, Any]) -> None:
    global _POLICY_CONFIG
    _POLICY_CONFIG = resolved_config
    RDLogger.DisableLog("rdApp.*")


def _descriptors(molecule: Chem.Mol) -> dict[str, Any]:
    elements: dict[str, int] = {}
    heavy_atoms = 0
    heteroatoms = 0
    aromatic_atoms = 0
    formal_charge = 0
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        elements[symbol] = elements.get(symbol, 0) + 1
        atomic_number = atom.GetAtomicNum()
        heavy_atoms += int(atomic_number > 1)
        heteroatoms += int(atomic_number not in (1, 6))
        aromatic_atoms += int(atom.GetIsAromatic())
        formal_charge += int(atom.GetFormalCharge())
    return {
        "atom_count": int(molecule.GetNumAtoms()),
        "heavy_atom_count": int(heavy_atoms),
        "bond_count": int(molecule.GetNumBonds()),
        "ring_count": int(molecule.GetRingInfo().NumRings()),
        "heteroatom_count": int(heteroatoms),
        "formal_charge": int(formal_charge),
        "aromatic_atom_count": int(aromatic_atoms),
        "aromatic_bond_count": int(
            sum(int(bond.GetIsAromatic()) for bond in molecule.GetBonds())
        ),
        "element_counts_json": json.dumps(
            elements, sort_keys=True, separators=(",", ":")
        ),
    }


def _empty_record(raw: str) -> dict[str, Any]:
    return {
        "raw_smiles": raw,
        "rdkit_valid": False,
        "policy_accepted": False,
        "policy_rejection": "rdkit_parse_failure",
        "canonical_smiles": "",
        "molecule_hash": "",
        "scaffold": "",
        "policy_split_assignment": "",
        "raw_equals_canonical": False,
        "atom_count": -1,
        "heavy_atom_count": -1,
        "bond_count": -1,
        "ring_count": -1,
        "heteroatom_count": -1,
        "formal_charge": 0,
        "aromatic_atom_count": -1,
        "aromatic_bond_count": -1,
        "element_counts_json": "{}",
    }


def _worker(raw_smiles: str) -> dict[str, Any]:
    if _POLICY_CONFIG is None:
        raise RuntimeError("Policy worker was not initialized")
    raw = str(raw_smiles)
    record = _empty_record(raw)
    try:
        parsed = Chem.MolFromSmiles(raw) if raw else None
    except Exception as error:
        record["policy_rejection"] = f"rdkit_parse_exception:{type(error).__name__}"
        return record
    if parsed is None:
        return record
    record["rdkit_valid"] = True
    data = _POLICY_CONFIG["data"]
    policy = data["canonicalization"]
    try:
        result = canonicalize(
            raw,
            isomeric_smiles=bool(policy["isomeric_smiles"]),
            fragment_policy=str(policy["fragment_policy"]),
            allowed_elements={str(item) for item in policy["allowed_elements"]},
            min_atoms=int(policy["min_atoms"]),
            max_atoms=int(policy["max_atoms"]),
            buckets=int(data["hash_buckets"]),
            split_cfg=data["split"],
        )
    except Exception as error:
        record["rdkit_valid"] = False
        record["policy_rejection"] = f"rdkit_sanitization_exception:{type(error).__name__}"
        return record
    if isinstance(result, Rejection):
        record["policy_rejection"] = str(result.reason)
        return record
    try:
        canonical_molecule = Chem.MolFromSmiles(result.smiles)
        if canonical_molecule is None:
            raise ValueError("canonical output did not parse")
        descriptors = _descriptors(canonical_molecule)
    except Exception as error:
        record["rdkit_valid"] = False
        record["policy_rejection"] = f"canonical_sanitization_exception:{type(error).__name__}"
        return record
    record.update(
        {
            "policy_accepted": True,
            "policy_rejection": "",
            "canonical_smiles": str(result.smiles),
            "molecule_hash": str(result.molecule_hash),
            "scaffold": str(result.scaffold),
            "policy_split_assignment": str(result.split),
            "raw_equals_canonical": raw == str(result.smiles),
            **descriptors,
        }
    )
    return record


def audit_raw_smiles(
    raw_smiles: Sequence[str], *, resolved_config: dict[str, Any], workers: int
) -> pd.DataFrame:
    values = [str(value) for value in raw_smiles]
    if values != sorted(set(values)):
        raise RuntimeError("Policy-audit input must be sorted and unique")
    with ProcessPoolExecutor(
        max_workers=int(workers), initializer=_initialize, initargs=(resolved_config,)
    ) as executor:
        records = list(executor.map(_worker, values, chunksize=256))
    frame = pd.DataFrame.from_records(records)
    if len(frame) != len(values) or frame["raw_smiles"].tolist() != values:
        raise RuntimeError("Exception-safe policy audit lost input ordering")
    return frame


__all__ = ["audit_raw_smiles"]
