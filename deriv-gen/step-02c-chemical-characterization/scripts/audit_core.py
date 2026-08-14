"""Chemistry primitives used by the frozen Step 2c candidate audit."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from common import numeric_summary


RDLogger.DisableLog("rdApp.*")

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
REPO_SOURCE = REPO_ROOT / "src"
if str(REPO_SOURCE) not in sys.path:
    sys.path.insert(0, str(REPO_SOURCE))

from gmolai_retrain.chem import Rejection, canonicalize  # noqa: E402


_POLICY_CONFIG: dict[str, Any] | None = None


def _initialize_policy_worker(resolved_config: dict[str, Any]) -> None:
    global _POLICY_CONFIG
    _POLICY_CONFIG = resolved_config
    RDLogger.DisableLog("rdApp.*")


def molecular_descriptors(molecule: Chem.Mol) -> dict[str, Any]:
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
    aromatic_bonds = sum(int(bond.GetIsAromatic()) for bond in molecule.GetBonds())
    return {
        "atom_count": int(molecule.GetNumAtoms()),
        "heavy_atom_count": int(heavy_atoms),
        "bond_count": int(molecule.GetNumBonds()),
        "ring_count": int(molecule.GetRingInfo().NumRings()),
        "heteroatom_count": int(heteroatoms),
        "formal_charge": int(formal_charge),
        "aromatic_atom_count": int(aromatic_atoms),
        "aromatic_bond_count": int(aromatic_bonds),
        "element_counts_json": json.dumps(elements, sort_keys=True, separators=(",", ":")),
    }


def _policy_audit_worker(raw_smiles: str) -> dict[str, Any]:
    if _POLICY_CONFIG is None:
        raise RuntimeError("Policy worker was not initialized")
    raw = str(raw_smiles)
    parsed = Chem.MolFromSmiles(raw) if raw else None
    rdkit_valid = parsed is not None
    data = _POLICY_CONFIG["data"]
    policy = data["canonicalization"]
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
    record: dict[str, Any] = {
        "raw_smiles": raw,
        "rdkit_valid": bool(rdkit_valid),
        "policy_accepted": not isinstance(result, Rejection),
        "policy_rejection": result.reason if isinstance(result, Rejection) else "",
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
    if isinstance(result, Rejection):
        return record
    canonical_molecule = Chem.MolFromSmiles(result.smiles)
    if canonical_molecule is None:
        raise RuntimeError(f"Policy output failed to parse: {result.smiles}")
    record.update(
        {
            "canonical_smiles": str(result.smiles),
            "molecule_hash": str(result.molecule_hash),
            "scaffold": str(result.scaffold),
            "policy_split_assignment": str(result.split),
            "raw_equals_canonical": raw == str(result.smiles),
            **molecular_descriptors(canonical_molecule),
        }
    )
    return record


def audit_raw_smiles(
    raw_smiles: Sequence[str], *, resolved_config: dict[str, Any], workers: int
) -> pd.DataFrame:
    values = [str(value) for value in raw_smiles]
    if values != sorted(set(values)):
        raise RuntimeError("Policy-audit input must be sorted and unique")
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_initialize_policy_worker,
        initargs=(resolved_config,),
    ) as executor:
        for completed, record in enumerate(
            executor.map(_policy_audit_worker, values, chunksize=256), start=1
        ):
            records.append(record)
            if completed % 50_000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"  policy-audited {completed:,}/{len(values):,} "
                    f"({completed / elapsed:,.0f} strings/s)",
                    flush=True,
                )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != len(values) or frame["raw_smiles"].tolist() != values:
        raise RuntimeError("Policy-audit output lost input ordering")
    return frame


def mmp_explanations(
    candidate_pairs: pd.DataFrame,
    fragments: pd.DataFrame,
    *,
    settings: dict[str, Any],
    threads: int,
    temporary_dir: Path,
) -> pd.DataFrame:
    required_pair_columns = {
        "candidate_row_id",
        "query_position",
        "seed_structure_index",
        "candidate_structure_index",
    }
    if not required_pair_columns.issubset(candidate_pairs.columns):
        raise RuntimeError("Candidate-pair table lacks MMP join columns")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET memory_limit='200GB'")
        escaped = str(temporary_dir).replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped}'")
        connection.register("candidate_pairs", candidate_pairs)
        connection.register("fragments", fragments)
        result = connection.execute(
            f"""
            WITH eligible AS (
                SELECT
                    p.candidate_row_id::BIGINT AS candidate_row_id,
                    p.query_position::BIGINT AS query_position,
                    seed_fragment.core AS core,
                    seed_fragment.substituent AS seed_substituent,
                    candidate_fragment.substituent AS candidate_substituent,
                    seed_fragment.core_heavy_atoms::INTEGER AS core_heavy_atoms,
                    seed_fragment.substituent_heavy_atoms::INTEGER
                        AS seed_substituent_heavy_atoms,
                    candidate_fragment.substituent_heavy_atoms::INTEGER
                        AS candidate_substituent_heavy_atoms,
                    seed_fragment.parent_heavy_atoms::INTEGER
                        AS seed_parent_heavy_atoms,
                    candidate_fragment.parent_heavy_atoms::INTEGER
                        AS candidate_parent_heavy_atoms
                FROM candidate_pairs AS p
                INNER JOIN fragments AS seed_fragment
                    ON seed_fragment.molecule_index = p.seed_structure_index
                INNER JOIN fragments AS candidate_fragment
                    ON candidate_fragment.molecule_index = p.candidate_structure_index
                   AND candidate_fragment.core = seed_fragment.core
                WHERE seed_fragment.substituent != candidate_fragment.substituent
                  AND abs(
                    seed_fragment.substituent_heavy_atoms
                    - candidate_fragment.substituent_heavy_atoms
                  ) <= {int(settings['max_variable_heavy_atom_delta'])}
                  AND abs(
                    seed_fragment.parent_heavy_atoms
                    - candidate_fragment.parent_heavy_atoms
                  ) <= {int(settings['max_parent_heavy_atom_delta'])}
            ), ranked AS (
                SELECT
                    *,
                    candidate_substituent_heavy_atoms
                        - seed_substituent_heavy_atoms AS variable_heavy_atom_delta,
                    candidate_parent_heavy_atoms
                        - seed_parent_heavy_atoms AS parent_heavy_atom_delta,
                    seed_substituent || '>>' || candidate_substituent
                        AS seed_to_candidate_transform,
                    least(seed_substituent, candidate_substituent) || '>>'
                        || greatest(seed_substituent, candidate_substituent)
                        AS undirected_transform,
                    CASE
                        WHEN candidate_substituent_heavy_atoms
                             > seed_substituent_heavy_atoms
                            THEN 'substituent_growth'
                        WHEN candidate_substituent_heavy_atoms
                             < seed_substituent_heavy_atoms
                            THEN 'substituent_truncation'
                        ELSE 'equal_heavy_atom_replacement'
                    END AS mmp_edit_class,
                    count(*) OVER (PARTITION BY candidate_row_id)
                        AS mmp_explanation_count,
                    row_number() OVER (
                        PARTITION BY candidate_row_id
                        ORDER BY
                            core_heavy_atoms DESC,
                            seed_substituent_heavy_atoms
                                + candidate_substituent_heavy_atoms ASC,
                            core ASC,
                            seed_substituent ASC,
                            candidate_substituent ASC
                    ) AS explanation_rank
                FROM eligible
            )
            SELECT *, explanation_rank = 1 AS is_primary_explanation
            FROM ranked
            ORDER BY candidate_row_id, explanation_rank
            """
        ).fetchdf()
    finally:
        connection.close()
    if not result.empty:
        result["candidate_row_id"] = result["candidate_row_id"].astype(np.int64)
        result["query_position"] = result["query_position"].astype(np.int64)
        result["mmp_explanation_count"] = result["mmp_explanation_count"].astype(
            np.int32
        )
        result["explanation_rank"] = result["explanation_rank"].astype(np.int32)
    return result


def _prefixed_summary(prefix: str, values: np.ndarray) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in numeric_summary(values).items()}


def compute_morgan_and_diversity(
    candidates: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    radius: int,
    bits: int,
    progress_every: int = 1000,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], float]:
    if candidates["candidate_row_id"].tolist() != list(range(len(candidates))):
        raise RuntimeError("Candidate row identifiers must be contiguous")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(bits), includeChirality=False
    )
    morgan_to_seed = np.full(len(candidates), np.nan, dtype=np.float32)
    grouped = {int(key): value for key, value in candidates.groupby("query_position", sort=False)}
    all_pairwise_parts: list[np.ndarray] = []
    nonseed_pairwise_parts: list[np.ndarray] = []
    all_offsets = np.zeros(len(seeds) + 1, dtype=np.int64)
    nonseed_offsets = np.zeros(len(seeds) + 1, dtype=np.int64)
    seed_rows: list[dict[str, Any]] = []
    maximum_stored_difference = 0.0
    started = time.monotonic()

    for query_position in range(len(seeds)):
        seed = seeds.iloc[query_position]
        seed_molecule = Chem.MolFromSmiles(str(seed["seed_canonical_smiles"]))
        if seed_molecule is None:
            raise RuntimeError(f"Seed failed to parse at query {query_position}")
        seed_fp = generator.GetFingerprint(seed_molecule)
        current = grouped.get(query_position)
        if current is None:
            current = candidates.iloc[0:0]
        fps = []
        for smiles in current["canonical_smiles"].astype(str):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"Retained canonical candidate failed to parse: {smiles}")
            fps.append(generator.GetFingerprint(molecule))
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(seed_fp, fps), dtype=np.float32
        )
        row_ids = current["candidate_row_id"].to_numpy(dtype=np.int64)
        morgan_to_seed[row_ids] = similarities
        if len(current):
            stored = current["morgan_similarity_to_target"].to_numpy(dtype=np.float32)
            maximum_stored_difference = max(
                maximum_stored_difference,
                float(np.max(np.abs(similarities - stored))),
            )

        all_pairwise_values: list[float] = []
        for first in range(len(fps) - 1):
            all_pairwise_values.extend(
                DataStructs.BulkTanimotoSimilarity(fps[first], fps[first + 1 :])
            )
        all_values = np.asarray(all_pairwise_values, dtype=np.float32)
        all_pairwise_parts.append(all_values)
        all_offsets[query_position + 1] = all_offsets[query_position] + len(all_values)

        nonseed_mask = ~current["is_seed_identity"].to_numpy(dtype=bool)
        nonseed_fps = [fp for fp, keep in zip(fps, nonseed_mask) if keep]
        nonseed_pairwise_values: list[float] = []
        for first in range(len(nonseed_fps) - 1):
            nonseed_pairwise_values.extend(
                DataStructs.BulkTanimotoSimilarity(
                    nonseed_fps[first], nonseed_fps[first + 1 :]
                )
            )
        nonseed_values = np.asarray(nonseed_pairwise_values, dtype=np.float32)
        nonseed_pairwise_parts.append(nonseed_values)
        nonseed_offsets[query_position + 1] = (
            nonseed_offsets[query_position] + len(nonseed_values)
        )

        nonseed_seed_similarity = similarities[nonseed_mask]
        seed_rows.append(
            {
                "query_position": query_position,
                **_prefixed_summary("seed_candidate_morgan", nonseed_seed_similarity),
                **_prefixed_summary("within_set_pairwise_morgan", all_values),
                **_prefixed_summary(
                    "within_nonseed_pairwise_morgan", nonseed_values
                ),
                "within_set_pairwise_fraction_ge_0_90": (
                    float(np.mean(all_values >= 0.90)) if len(all_values) else math.nan
                ),
                "within_set_pairwise_fraction_eq_1": (
                    float(np.mean(all_values == 1.0)) if len(all_values) else math.nan
                ),
            }
        )
        if (query_position + 1) % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"  Morgan/diversity {query_position + 1:,}/{len(seeds):,} "
                f"({(query_position + 1) / elapsed:,.0f} seeds/s)",
                flush=True,
            )

    if not np.isfinite(morgan_to_seed).all():
        raise RuntimeError("Morgan similarity was not assigned to every candidate")
    all_pairwise = (
        np.concatenate(all_pairwise_parts)
        if all_pairwise_parts
        else np.empty(0, dtype=np.float32)
    )
    nonseed_pairwise = (
        np.concatenate(nonseed_pairwise_parts)
        if nonseed_pairwise_parts
        else np.empty(0, dtype=np.float32)
    )
    arrays = {
        "all_candidate_pairwise_morgan": all_pairwise,
        "all_candidate_query_offsets": all_offsets,
        "nonseed_candidate_pairwise_morgan": nonseed_pairwise,
        "nonseed_candidate_query_offsets": nonseed_offsets,
    }
    return (
        morgan_to_seed,
        pd.DataFrame(seed_rows),
        arrays,
        maximum_stored_difference,
    )


def histogram_table(values: np.ndarray, *, bins: int, population: str) -> pd.DataFrame:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, range=(0, 1))
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "population": population,
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts.astype(np.int64),
            "fraction": counts.astype(np.float64) / total,
        }
    )
