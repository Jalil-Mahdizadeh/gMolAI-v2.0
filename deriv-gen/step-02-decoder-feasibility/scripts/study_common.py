"""Shared integrity, tokenization, chemistry, and metric utilities for Step 2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BYTE_OFFSET = 3
ASCII_VALUES = 128
VOCAB_SIZE = BYTE_OFFSET + ASCII_VALUES


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
    expected = root.resolve()
    if resolved != expected and expected not in resolved.parents:
        raise RuntimeError(f"Refusing write outside {expected}: {resolved}")
    return resolved


def _temporary(path: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(raw)


def atomic_write_text(path: Path, value: str, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, root: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        root,
    )


def atomic_write_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Any, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_numpy_save(path: Path, array: np.ndarray, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_numpy_savez(path: Path, root: Path, **arrays: np.ndarray) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(target)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


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
        if observed != record["sha256"]:
            raise RuntimeError(
                f"Immutable input changed for {role}: {observed} != {record['sha256']}"
            )
        if role != "container" and any(token in str(path).lower() for token in forbidden):
            raise RuntimeError(f"Forbidden input path: {path}")
        if step_root.resolve() in path.resolve().parents:
            raise RuntimeError(f"Manifest input must be external to Step 2: {path}")
        paths[role] = path
        hashes[role] = observed
    if manifest.get("selected_embedding_space") != "released_hybrid_w3":
        raise RuntimeError("Manifest embedding-space identity changed")
    if hashes["checkpoint"] != hashes["packaged_checkpoint"]:
        raise RuntimeError("Training and packaged checkpoint identities differ")
    if hashes["calibrator"] != hashes["packaged_calibrator"]:
        raise RuntimeError("Training and packaged calibrator identities differ")
    if hashes["resolved_config"] != hashes["packaged_resolved_config"]:
        raise RuntimeError("Training and packaged resolved configs differ")
    decision = json.loads(paths["step1b_decision"].read_text(encoding="utf-8"))
    if decision.get("selected_edit_control_space") != "released_hybrid_w3":
        raise RuntimeError("Step-1b did not select released_hybrid_w3")
    return paths, hashes


def encode_smiles(smiles: str, maximum_bytes: int) -> list[int]:
    raw = smiles.encode("ascii", errors="strict")
    if len(raw) > maximum_bytes:
        raise ValueError(f"SMILES is {len(raw)} bytes, above ceiling {maximum_bytes}")
    return [BOS_TOKEN, *[int(value) + BYTE_OFFSET for value in raw], EOS_TOKEN]


def token_matrix(values: Sequence[str], maximum_bytes: int) -> np.ndarray:
    encoded = [encode_smiles(str(value), maximum_bytes) for value in values]
    width = max(len(value) for value in encoded)
    result = np.full((len(encoded), width), PAD_TOKEN, dtype=np.uint8)
    for index, row in enumerate(encoded):
        result[index, : len(row)] = row
    return result


def decode_tokens(tokens: Sequence[int]) -> tuple[str, str]:
    raw: list[int] = []
    saw_eos = False
    for value in tokens:
        value = int(value)
        if value == EOS_TOKEN:
            saw_eos = True
            break
        if value in {PAD_TOKEN, BOS_TOKEN}:
            return "", f"reserved_token_{value}"
        byte = value - BYTE_OFFSET
        if byte < 0 or byte >= ASCII_VALUES:
            return "", "out_of_range_token"
        raw.append(byte)
    if not saw_eos:
        return "", "missing_eos"
    try:
        return bytes(raw).decode("ascii", errors="strict"), ""
    except UnicodeDecodeError:
        return "", "non_ascii"


def payload_array(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Payload field {key!r} is not a tensor")
    return value.detach().cpu().numpy()


def released_train_embeddings(
    raw_payload: dict[str, Any],
    calibrator: dict[str, Any],
) -> np.ndarray:
    raw = payload_array(raw_payload, "embeddings").astype(np.float32)
    mean = payload_array(calibrator, "coordinate_mean").astype(np.float32)
    scale = payload_array(calibrator, "coordinate_scale").astype(np.float32)
    if raw.shape[1] != 384 or mean.shape != (384,) or scale.shape != (384,):
        raise RuntimeError("Frozen embedding/calibrator dimensions changed")
    result = np.ascontiguousarray(
        (raw - mean[None, :]) / scale[None, :], dtype=np.float32
    )
    result[:, 256:] *= 3.0
    if not np.isfinite(result).all():
        raise RuntimeError("Released train embeddings contain non-finite values")
    return result


def validation_embeddings(payload: dict[str, Any]) -> np.ndarray:
    metadata = payload["metadata"]
    if metadata.get("split") != "validation":
        raise RuntimeError("Validation payload split identity changed")
    if float(metadata["embedding_parameters"]["mean_node_weight"]) != 3.0:
        raise RuntimeError("Validation payload is not released weight 3")
    values = payload_array(payload, "embeddings").astype(np.float32)
    if values.shape != (50_000, 384) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid validation embedding matrix: {values.shape}")
    return np.ascontiguousarray(values)


def scaffold_group_keys(molecules: pd.DataFrame) -> list[str]:
    return [
        f"SCAFFOLD:{str(row.scaffold)}"
        if str(row.scaffold)
        else f"ACYCLIC:{str(row.molecule_hash)}"
        for row in molecules.itertuples(index=False)
    ]


def deterministic_panel_indices(
    hashes: Sequence[str], count: int, seed: int, label: str
) -> np.ndarray:
    if count <= 0 or count > len(hashes):
        raise ValueError("Invalid deterministic panel size")
    ranked = sorted(
        range(len(hashes)),
        key=lambda index: stable_digest(seed, label, hashes[index]),
    )
    return np.asarray(ranked[:count], dtype=np.int64)


def derangement(size: int, seed: int, label: str) -> np.ndarray:
    if size < 2:
        raise ValueError("Derangement requires at least two rows")
    rng = np.random.default_rng(
        int(stable_digest(seed, label)[:16], 16)
    )
    order = rng.permutation(size)
    result = np.empty(size, dtype=np.int64)
    result[order] = np.roll(order, -1)
    if np.any(result == np.arange(size)):
        raise RuntimeError("Deterministic derangement has a fixed point")
    return result


def topk_l2(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    k: int,
    device: torch.device,
    batch_size: int,
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bank_tensor = torch.as_tensor(bank, dtype=torch.float32, device=device)
    bank_norm = torch.square(bank_tensor).sum(dim=1).unsqueeze(0)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    for offset in range(0, len(query), batch_size):
        stop = min(offset + batch_size, len(query))
        current = torch.as_tensor(
            query[offset:stop], dtype=torch.float32, device=device
        )
        distances = (
            torch.square(current).sum(dim=1, keepdim=True)
            + bank_norm
            - 2.0 * current @ bank_tensor.T
        )
        distances.clamp_(min=0.0)
        if exclude_indices is not None:
            excluded = np.asarray(exclude_indices[offset:stop], dtype=np.int64)
            rows = torch.arange(stop - offset, device=device)
            distances[
                rows,
                torch.as_tensor(excluded, dtype=torch.long, device=device),
            ] = torch.inf
        values, indices = torch.topk(
            distances, k=k, dim=1, largest=False, sorted=True
        )
        all_indices.append(indices.cpu().numpy().astype(np.int64))
        all_distances.append(torch.sqrt(values).cpu().numpy().astype(np.float32))
    del bank_tensor, bank_norm
    torch.cuda.empty_cache()
    return np.concatenate(all_indices), np.concatenate(all_distances)


def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=1)
    denominator = np.linalg.norm(first.astype(np.float64), axis=1) * np.linalg.norm(
        second.astype(np.float64), axis=1
    )
    result = np.full(len(first), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def make_fingerprints(smiles: Sequence[str]) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result: list[Any] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise RuntimeError(f"Cannot fingerprint canonical target: {value}")
        result.append(generator.GetFingerprint(molecule))
    return result


def tanimoto(first: Any, second: Any) -> float:
    return float(DataStructs.TanimotoSimilarity(first, second))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    point = float(np.mean(array))
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    chunk = 100
    position = 0
    while position < resamples:
        take = min(chunk, resamples - position)
        indices = rng.integers(0, len(array), size=(take, len(array)))
        estimates[position : position + take] = array[indices].mean(axis=1)
        position += take
    return (
        point,
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def hash_ledger(root: Path, exclude: set[str] | None = None) -> str:
    omitted = exclude or set()
    lines: list[str] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in omitted:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    return "\n".join(lines) + "\n"
