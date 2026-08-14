"""Shared deterministic utilities for the scaled latent-space selection study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
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
    return hashlib.sha256(
        "\x1f".join(str(value) for value in parts).encode("utf-8")
    ).hexdigest()


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside {resolved_root}: {resolved}")
    return resolved


def _temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path: Path, value: str, root: Path) -> None:
    path = ensure_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, root: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        root,
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    path = ensure_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, root: Path) -> None:
    path = ensure_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, root: Path, **arrays: np.ndarray) -> None:
    path = ensure_within(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def payload_array(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Payload field {key!r} is not a tensor")
    return value.detach().cpu().numpy()


def load_validate_manifest(
    repo_root: Path, step_root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    forbidden = ("test-partition", "test-standardized", "moleculenet", "hiv")
    for role, record in manifest["files"].items():
        raw = Path(record["path"])
        path = raw if raw.is_absolute() else repo_root / raw
        if not path.is_file():
            raise FileNotFoundError(f"Missing immutable input {role}: {path}")
        observed = sha256_file(path)
        if observed != str(record["sha256"]):
            raise RuntimeError(
                f"Input hash mismatch for {role}: {observed} != {record['sha256']}"
            )
        if role != "container" and any(token in str(path).lower() for token in forbidden):
            raise RuntimeError(f"Forbidden input path: {path}")
        if step_root.resolve() in path.resolve().parents:
            raise RuntimeError(f"Manifest input must be external to study: {path}")
        paths[role] = path
        hashes[role] = observed
    return paths, hashes


def chemical_records(
    payload: dict[str, Any],
    cache_records: dict[str, list[str]],
    work_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    hashes = [str(value) for value in payload["molecule_hashes"]]
    buckets = payload_array(payload, "source_buckets").reshape(-1).astype(np.int64)
    if len(hashes) != len(buckets):
        raise RuntimeError("Payload hashes and source buckets are not row-aligned")
    records = {
        value: (str(cache_records[value][0]), str(cache_records[value][1] or ""))
        for value in hashes
        if value in cache_records
    }
    missing_by_bucket: dict[int, list[str]] = defaultdict(list)
    for value, bucket in zip(hashes, buckets):
        if value not in records:
            missing_by_bucket[int(bucket)].append(value)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=8")
        for position, (bucket, wanted_hashes) in enumerate(
            sorted(missing_by_bucket.items()), start=1
        ):
            parquet_path = work_dir / "deduplicated" / f"bucket-{bucket:04d}.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(parquet_path)
            connection.register(
                "wanted_hashes",
                pa.table({"molecule_hash": sorted(set(wanted_hashes))}),
            )
            rows = connection.execute(
                """
                SELECT d.molecule_hash, d.canonical_smiles, d.scaffold
                FROM read_parquet(?) AS d
                INNER JOIN wanted_hashes AS w USING (molecule_hash)
                WHERE d.split IN ('train', 'validation')
                """,
                [str(parquet_path)],
            ).fetchall()
            connection.unregister("wanted_hashes")
            for molecule_hash, smiles, scaffold in rows:
                records[str(molecule_hash)] = (str(smiles), str(scaffold or ""))
            if position % 32 == 0:
                print(
                    f"  chemical join buckets {position}/{len(missing_by_bucket)}",
                    flush=True,
                )
    finally:
        connection.close()
    missing = [value for value in hashes if value not in records]
    if missing:
        raise RuntimeError(f"Missing chemical records for {len(missing)} payload rows")
    return (
        hashes,
        [records[value][0] for value in hashes],
        [records[value][1] for value in hashes],
    )


def _heavy_atoms(molecule: Chem.Mol) -> int:
    return sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())


def _fragment_worker(
    task: tuple[int, str, dict[str, Any]]
) -> tuple[int, int, list[tuple[int, str, str, int, int, int]]]:
    index, smiles, settings = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return index, -1, []
    parent_heavy = _heavy_atoms(molecule)
    unique: set[tuple[str, str]] = set()
    rows: list[tuple[int, str, str, int, int, int]] = []
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
                pieces.extend(value for value in text.split(".") if value)
        if len(pieces) != 2:
            continue
        parsed: list[tuple[str, int]] = []
        for piece in pieces:
            fragment = Chem.MolFromSmiles(piece)
            if fragment is None:
                parsed = []
                break
            parsed.append(
                (
                    Chem.MolToSmiles(
                        fragment, canonical=True, isomericSmiles=True
                    ),
                    _heavy_atoms(fragment),
                )
            )
        if len(parsed) != 2 or parsed[0][1] == parsed[1][1]:
            continue
        parsed.sort(key=lambda value: (-value[1], value[0]))
        (core, core_heavy), (substituent, substituent_heavy) = parsed
        if core_heavy < int(settings["min_core_heavy_atoms"]):
            continue
        if not (
            int(settings["min_variable_heavy_atoms"])
            <= substituent_heavy
            <= int(settings["max_variable_heavy_atoms"])
        ):
            continue
        if core_heavy / max(parent_heavy, 1) < float(settings["min_core_fraction"]):
            continue
        identity = (core, substituent)
        if identity in unique:
            continue
        unique.add(identity)
        rows.append(
            (
                index,
                core,
                substituent,
                core_heavy,
                substituent_heavy,
                parent_heavy,
            )
        )
    rows.sort(key=lambda value: (value[1], value[2]))
    return index, parent_heavy, rows


def fragment_molecules(
    smiles: Sequence[str],
    *,
    settings: dict[str, Any],
    workers: int,
    progress_every: int = 100_000,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    tasks = (
        (index, str(value), settings) for index, value in enumerate(smiles)
    )
    rows: list[tuple[int, str, str, int, int, int]] = []
    heavy = np.full(len(smiles), -1, dtype=np.int16)
    parse_failures = 0
    with_fragments = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, (index, parent_heavy, fragments) in enumerate(
            executor.map(_fragment_worker, tasks, chunksize=256), start=1
        ):
            heavy[index] = int(parent_heavy)
            if parent_heavy < 0:
                parse_failures += 1
            if fragments:
                with_fragments += 1
                rows.extend(fragments)
            if completed % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"  fragmented {completed:,}/{len(smiles):,} "
                    f"({completed / elapsed:,.0f} mol/s)",
                    flush=True,
                )
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
        ).sort_values(
            ["core", "substituent", "molecule_index"], ignore_index=True
        )
    statistics = {
        "molecules": len(smiles),
        "parse_failures": parse_failures,
        "molecules_with_eligible_fragments": with_fragments,
        "eligible_fragmentations": len(frame),
    }
    return frame, heavy, statistics


def unit_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize degenerate vector")
    return array / norm


def covariance_eigendecomposition(
    values: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    mean = tensor.mean(dim=0, keepdim=True)
    centered = tensor - mean
    covariance = centered.T @ centered / float(centered.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    result = (
        eigenvalues.detach().cpu().numpy().astype(np.float64),
        eigenvectors.detach().cpu().numpy().astype(np.float64),
    )
    del tensor, mean, centered, covariance, eigenvalues, eigenvectors
    torch.cuda.empty_cache()
    return result


def covariance_sample(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    coefficients = rng.normal(size=len(eigenvalues)) * np.sqrt(
        np.clip(eigenvalues, 0.0, None)
    )
    return eigenvectors @ coefficients


def local_covariance_sample(
    centered_neighbors: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    matrix = np.asarray(centered_neighbors, dtype=np.float64)
    coefficients = rng.normal(size=matrix.shape[0])
    return coefficients @ matrix / math.sqrt(max(matrix.shape[0], 1))


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
        raise ValueError("Query and bank matrices are not dimensionally aligned")
    if not 0 < k < bank.shape[0]:
        raise ValueError(f"Invalid top-k {k} for {bank.shape[0]} candidates")
    bank_tensor = torch.as_tensor(bank, dtype=torch.float32, device=device)
    bank_norm = torch.square(bank_tensor).sum(dim=1).unsqueeze(0)
    result_indices: list[np.ndarray] = []
    result_distances: list[np.ndarray] = []
    for offset in range(0, len(query), batch_size):
        stop = min(offset + batch_size, len(query))
        batch = torch.as_tensor(
            query[offset:stop], dtype=torch.float32, device=device
        )
        distances = (
            torch.square(batch).sum(dim=1, keepdim=True)
            + bank_norm
            - 2.0 * batch @ bank_tensor.T
        )
        distances.clamp_(min=0.0)
        if exclude_indices is not None:
            excluded = np.asarray(exclude_indices[offset:stop], dtype=np.int64)
            valid = np.flatnonzero(excluded >= 0)
            if len(valid):
                distances[
                    torch.as_tensor(valid, dtype=torch.long, device=device),
                    torch.as_tensor(
                        excluded[valid], dtype=torch.long, device=device
                    ),
                ] = torch.inf
        values, indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True
        )
        result_indices.append(indices.cpu().numpy().astype(np.int64))
        result_distances.append(torch.sqrt(values).cpu().numpy().astype(np.float32))
        del batch, distances, values, indices
    del bank_tensor, bank_norm
    torch.cuda.empty_cache()
    return np.concatenate(result_indices), np.concatenate(result_distances)


def make_fingerprints(smiles: Sequence[str]) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result: list[Any] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(f"Cannot fingerprint immutable SMILES: {value}")
        result.append(generator.GetFingerprint(molecule))
    return result


def tanimoto(first: Any, second: Any) -> float:
    return float(DataStructs.TanimotoSimilarity(first, second))


def core_sets(fragments: pd.DataFrame, size: int) -> list[set[str]]:
    result = [set() for _ in range(size)]
    for row in fragments.itertuples(index=False):
        result[int(row.molecule_index)].add(str(row.core))
    return result


def requested_target_sets(
    fragments: pd.DataFrame,
) -> dict[tuple[str, str], set[int]]:
    result: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in fragments.itertuples(index=False):
        result[(str(row.core), str(row.substituent))].add(
            int(row.molecule_index)
        )
    return dict(result)


def one_cut_related(sets: list[set[str]], first: int, second: int) -> bool:
    return bool(sets[first].intersection(sets[second]))


def support_tier(value: int) -> str:
    if value >= 20:
        return "20+"
    if value >= 10:
        return "10-19"
    if value >= 5:
        return "5-9"
    return "2-4"


def hash_ledger(root: Path, *, exclude: set[str] | None = None) -> str:
    excluded = exclude or set()
    lines: list[str] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    return "\n".join(lines) + "\n"
