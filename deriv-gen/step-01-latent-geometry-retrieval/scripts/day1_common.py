"""Shared, deterministic helpers for the Day-1 derivative feasibility study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMMPA


RDLogger.DisableLog("rdApp.*")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"Refusing to write outside {resolved_root}: {resolved}")
    return resolved


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path: Path, text: str, write_root: Path) -> None:
    path = ensure_within(path, write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, write_root: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        write_root,
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame, write_root: Path) -> None:
    path = ensure_within(path, write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, write_root: Path) -> None:
    path = ensure_within(path, write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, write_root: Path, **arrays: np.ndarray) -> None:
    path = ensure_within(path, write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1 or resamples <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=np.float64)
    batch = min(200, resamples)
    written = 0
    while written < resamples:
        count = min(batch, resamples - written)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        boot[written : written + count] = array[indices].mean(axis=1)
        written += count
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def effective_rank(eigenvalues: np.ndarray) -> dict[str, float | int]:
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Covariance spectrum has no positive mass")
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy_rank = float(np.exp(-(positive * np.log(positive)).sum()))
    participation = float(total * total / max(float(np.square(values).sum()), 1e-30))
    descending = values[::-1]
    cumulative = np.cumsum(descending) / total

    def components_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    return {
        "effective_rank": entropy_rank,
        "participation_ratio": participation,
        "top_eigenvalue_fraction": float(descending[0] / total),
        "variance_top_10": float(cumulative[min(9, len(cumulative) - 1)]),
        "variance_top_30": float(cumulative[min(29, len(cumulative) - 1)]),
        "variance_top_50": float(cumulative[min(49, len(cumulative) - 1)]),
        "components_90pct": components_for(0.90),
        "components_95pct": components_for(0.95),
    }


def covariance_eigendecomposition(
    train: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(train, dtype=torch.float32, device=device)
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(centered.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    values = eigenvalues.detach().cpu().numpy().astype(np.float64)
    vectors = eigenvectors.detach().cpu().numpy().astype(np.float64)
    del covariance, eigenvalues, eigenvectors, centered, tensor
    torch.cuda.empty_cache()
    return values, vectors


def topk_l2(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    k: int,
    device: torch.device,
    batch_size: int,
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("Query and bank must be aligned two-dimensional matrices")
    if k <= 0 or k >= bank.shape[0]:
        raise ValueError(f"Invalid top-k {k} for bank with {bank.shape[0]} rows")
    bank_tensor = torch.as_tensor(bank, dtype=torch.float32, device=device)
    bank_norm = torch.square(bank_tensor).sum(dim=1).unsqueeze(0)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    for offset in range(0, query.shape[0], batch_size):
        stop = min(offset + batch_size, query.shape[0])
        batch = torch.as_tensor(query[offset:stop], dtype=torch.float32, device=device)
        distances = (
            torch.square(batch).sum(dim=1, keepdim=True)
            + bank_norm
            - 2.0 * (batch @ bank_tensor.T)
        )
        distances.clamp_(min=0.0)
        if exclude_indices is not None:
            excluded = np.asarray(exclude_indices[offset:stop], dtype=np.int64)
            valid = np.flatnonzero(excluded >= 0)
            if len(valid):
                rows = torch.as_tensor(valid, dtype=torch.long, device=device)
                columns = torch.as_tensor(excluded[valid], dtype=torch.long, device=device)
                distances[rows, columns] = torch.inf
        values, indices = torch.topk(distances, k=k, dim=1, largest=False, sorted=True)
        all_indices.append(indices.detach().cpu().numpy().astype(np.int64))
        all_distances.append(
            torch.sqrt(values).detach().cpu().numpy().astype(np.float32)
        )
        del batch, distances, values, indices
    del bank_tensor, bank_norm
    torch.cuda.empty_cache()
    return np.concatenate(all_indices), np.concatenate(all_distances)


def _heavy_atoms(molecule: Chem.Mol) -> int:
    return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def _fragment_worker(
    task: tuple[int, str, str, dict[str, Any]]
) -> tuple[int, int, list[tuple[int, str, str, int, int, int]]]:
    index, molecule_hash, smiles, settings = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, -1, []
    parent_heavy = _heavy_atoms(molecule)
    unique: set[tuple[str, str]] = set()
    records: list[tuple[int, str, str, int, int, int]] = []
    try:
        fragmentations = rdMMPA.FragmentMol(
            molecule, maxCuts=1, resultsAsMols=False
        )
    except Exception:
        fragmentations = ()
    for core_text, chains_text in fragmentations:
        pieces: list[str] = []
        for text in (core_text, chains_text):
            if text:
                pieces.extend(piece for piece in text.split(".") if piece)
        if len(pieces) != 2:
            continue
        parsed: list[tuple[str, int]] = []
        for piece in pieces:
            fragment = Chem.MolFromSmiles(piece)
            if fragment is None:
                parsed = []
                break
            canonical = Chem.MolToSmiles(
                fragment, canonical=True, isomericSmiles=True
            )
            parsed.append((canonical, _heavy_atoms(fragment)))
        if len(parsed) != 2 or parsed[0][1] == parsed[1][1]:
            continue
        parsed.sort(key=lambda value: (-value[1], value[0]))
        (core, core_heavy), (variable, variable_heavy) = parsed
        if core_heavy < int(settings["min_core_heavy_atoms"]):
            continue
        if not (
            int(settings["min_variable_heavy_atoms"])
            <= variable_heavy
            <= int(settings["max_variable_heavy_atoms"])
        ):
            continue
        if core_heavy / max(parent_heavy, 1) < float(settings["min_core_fraction"]):
            continue
        identity = (core, variable)
        if identity in unique:
            continue
        unique.add(identity)
        records.append(
            (index, core, variable, core_heavy, variable_heavy, parent_heavy)
        )
    records.sort(key=lambda value: (value[1], value[2]))
    return index, parent_heavy, records


@dataclass
class FragmentationResult:
    fragments: pd.DataFrame
    heavy_atoms: np.ndarray
    statistics: dict[str, int]


def fragment_molecules(
    hashes: Sequence[str],
    smiles: Sequence[str],
    *,
    settings: dict[str, Any],
    workers: int,
) -> FragmentationResult:
    tasks = [
        (index, str(molecule_hash), str(smile), settings)
        for index, (molecule_hash, smile) in enumerate(zip(hashes, smiles))
    ]
    rows: list[tuple[int, str, str, int, int, int]] = []
    heavy = np.full(len(tasks), -1, dtype=np.int16)
    parse_failures = 0
    molecules_with_fragments = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(_fragment_worker, tasks, chunksize=128)
        for index, parent_heavy, records in iterator:
            heavy[index] = int(parent_heavy)
            if parent_heavy < 0:
                parse_failures += 1
            if records:
                molecules_with_fragments += 1
                rows.extend(records)
    frame = pd.DataFrame(
        rows,
        columns=[
            "molecule_index",
            "core",
            "substituent",
            "core_heavy_atoms",
            "substituent_heavy_atoms",
            "parent_heavy_atoms",
        ],
    )
    if not frame.empty:
        frame = frame.drop_duplicates(
            ["molecule_index", "core", "substituent"]
        ).sort_values(["core", "substituent", "molecule_index"], ignore_index=True)
    return FragmentationResult(
        fragments=frame,
        heavy_atoms=heavy,
        statistics={
            "molecules": len(tasks),
            "parse_failures": parse_failures,
            "molecules_with_eligible_fragments": molecules_with_fragments,
            "eligible_fragmentations": len(frame),
        },
    )


@dataclass
class PairingResult:
    pairs: pd.DataFrame
    statistics: dict[str, int]


def build_mmp_pairs(
    fragments: pd.DataFrame,
    hashes: Sequence[str],
    *,
    settings: dict[str, Any],
    seed: int,
) -> PairingResult:
    pair_rows: list[dict[str, Any]] = []
    skipped_large_groups = 0
    capped_groups = 0
    eligible_groups = 0
    for core, group in fragments.groupby("core", sort=True, observed=True):
        group = group.drop_duplicates("molecule_index").sort_values(
            ["substituent", "molecule_index"]
        )
        records = list(group.itertuples(index=False))
        if len(records) < 2:
            continue
        if len(records) > int(settings["max_core_group_size"]):
            skipped_large_groups += 1
            continue
        eligible_groups += 1
        candidates: list[tuple[str, dict[str, Any]]] = []
        seen_transforms: set[str] = set()
        for first_index in range(len(records) - 1):
            for second_index in range(first_index + 1, len(records)):
                first = records[first_index]
                second = records[second_index]
                if first.substituent == second.substituent:
                    continue
                if (
                    abs(
                        int(first.substituent_heavy_atoms)
                        - int(second.substituent_heavy_atoms)
                    )
                    > int(settings["max_variable_heavy_atom_delta"])
                ):
                    continue
                if (
                    abs(int(first.parent_heavy_atoms) - int(second.parent_heavy_atoms))
                    > int(settings["max_parent_heavy_atom_delta"])
                ):
                    continue
                lhs, rhs = (
                    (first, second)
                    if first.substituent < second.substituent
                    else (second, first)
                )
                transform = f"{lhs.substituent}>>{rhs.substituent}"
                if transform in seen_transforms:
                    continue
                seen_transforms.add(transform)
                lhs_index = int(lhs.molecule_index)
                rhs_index = int(rhs.molecule_index)
                identity = stable_digest(
                    seed,
                    core,
                    transform,
                    hashes[lhs_index],
                    hashes[rhs_index],
                )
                candidates.append(
                    (
                        identity,
                        {
                            "pair_id": identity,
                            "core": str(core),
                            "transform": transform,
                            "lhs_index": lhs_index,
                            "rhs_index": rhs_index,
                            "lhs_hash": str(hashes[lhs_index]),
                            "rhs_hash": str(hashes[rhs_index]),
                            "lhs_substituent": str(lhs.substituent),
                            "rhs_substituent": str(rhs.substituent),
                            "lhs_parent_heavy_atoms": int(lhs.parent_heavy_atoms),
                            "rhs_parent_heavy_atoms": int(rhs.parent_heavy_atoms),
                        },
                    )
                )
        candidates.sort(key=lambda item: item[0])
        maximum = int(settings["max_pairs_per_core"])
        if len(candidates) > maximum:
            candidates = candidates[:maximum]
            capped_groups += 1
        pair_rows.extend(item[1] for item in candidates)
    frame = pd.DataFrame(pair_rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["transform", "core", "pair_id"], ignore_index=True
        )
    return PairingResult(
        pairs=frame,
        statistics={
            "pairs": len(frame),
            "transformations": int(frame["transform"].nunique()) if len(frame) else 0,
            "eligible_core_groups": eligible_groups,
            "skipped_large_core_groups": skipped_large_groups,
            "capped_core_groups": capped_groups,
        },
    )


def core_sets(fragments: pd.DataFrame, molecule_count: int) -> list[set[str]]:
    result = [set() for _ in range(molecule_count)]
    for row in fragments[["molecule_index", "core"]].itertuples(index=False):
        result[int(row.molecule_index)].add(str(row.core))
    return result


def diverse_indices(
    hashes: Sequence[str],
    scaffolds: Sequence[str],
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    ranked = sorted(
        range(len(hashes)), key=lambda index: stable_digest(seed, hashes[index])
    )
    selected: list[int] = []
    seen_scaffolds: set[str] = set()
    for index in ranked:
        scaffold = str(scaffolds[index])
        key = scaffold if scaffold else f"__empty__{hashes[index]}"
        if key in seen_scaffolds:
            continue
        seen_scaffolds.add(key)
        selected.append(index)
        if len(selected) == maximum:
            break
    if len(selected) < maximum:
        selected_set = set(selected)
        selected.extend(index for index in ranked if index not in selected_set)
    return np.asarray(selected[:maximum], dtype=np.int64)


def make_fingerprints(smiles: Sequence[str]) -> tuple[list[Chem.Mol], list[Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    molecules: list[Chem.Mol] = []
    fingerprints: list[Any] = []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid cached molecule at row {index}: {value!r}")
        molecules.append(molecule)
        fingerprints.append(generator.GetFingerprint(molecule))
    return molecules, fingerprints


def tanimoto(first: Any, second: Any) -> float:
    return float(DataStructs.TanimotoSimilarity(first, second))


def unit_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return np.asarray(value, dtype=np.float64) / norm


def covariance_sample(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(size=len(eigenvalues))
    return eigenvectors @ (np.sqrt(np.clip(eigenvalues, 0.0, None)) * noise)


def local_covariance_sample(
    neighbor_vectors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    centered = neighbor_vectors - neighbor_vectors.mean(axis=0, keepdims=True)
    weights = rng.normal(size=centered.shape[0])
    return weights @ centered / math.sqrt(max(1, centered.shape[0] - 1))


def one_cut_related(core_lookup: Sequence[set[str]], first: int, second: int) -> bool:
    return bool(core_lookup[first].intersection(core_lookup[second]))


def hash_ledger(root: Path, destination: Path, write_root: Path) -> None:
    destination = ensure_within(destination, write_root)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != destination.resolve()
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in files]
    atomic_write_text(destination, "\n".join(lines) + "\n", write_root)

