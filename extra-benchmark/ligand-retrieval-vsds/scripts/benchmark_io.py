#!/usr/bin/env python3
"""Fail-closed I/O, identity, and protocol helpers for the LBVS benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[1]
PROTOCOL_PATH = BENCHMARK_DIR / "protocol.json"
REQUIRED_PANEL_COLUMNS = ("panel_index", "molecule_hash", "canonical_smiles")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise RuntimeError("Unsupported LBVS protocol schema")
    if protocol.get("protocol_status") != "frozen_before_representation_execution":
        raise RuntimeError("LBVS protocol is not frozen before representation execution")
    return protocol


def protocol_digest(protocol: dict[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def require_regular_file(
    path: str | Path,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> str:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError(f"Required regular file is missing or unsafe: {target}")
    if expected_bytes is not None and target.stat().st_size != int(expected_bytes):
        raise RuntimeError(
            f"Byte-size mismatch for {target}: expected {expected_bytes}, "
            f"observed {target.stat().st_size}"
        )
    observed = sha256_file(target)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {target}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return observed


def atomic_write_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


def _write_delimited(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    columns: Sequence[str],
    delimiter: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(columns),
                delimiter=delimiter,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_tsv(
    path: str | Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]
) -> None:
    _write_delimited(path, rows, columns, "\t")


def write_csv(
    path: str | Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]
) -> None:
    _write_delimited(path, rows, columns, ",")


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def columns_of(path: str | Path, delimiter: str = "\t") -> tuple[str, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        columns = csv.DictReader(handle, delimiter=delimiter).fieldnames
    if columns is None:
        raise RuntimeError(f"No header in {path}")
    return tuple(columns)


def read_panel_tsv(path: str | Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    fields = columns_of(path)
    if not all(column in fields for column in REQUIRED_PANEL_COLUMNS):
        raise RuntimeError(f"Missing required panel columns in {path}: {fields}")
    seen: set[str] = set()
    for expected, row in enumerate(rows):
        if int(row["panel_index"]) != expected:
            raise RuntimeError(f"Non-contiguous panel index at row {expected} in {path}")
        identity = row["molecule_hash"]
        expected_identity = hashlib.sha256(
            row["canonical_smiles"].encode("utf-8")
        ).hexdigest()
        if identity != expected_identity:
            raise RuntimeError(f"Identity hash mismatch at row {expected} in {path}")
        if identity in seen:
            raise RuntimeError(f"Duplicate molecule identity in {path}: {identity}")
        seen.add(identity)
    return rows


def canonicalize_external(raw_smiles: str, cfg: dict[str, Any]):
    """Remove explicit hydrogens, then apply the repository identity policy."""
    from rdkit import Chem

    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from gmolai_retrain.chem import Rejection, canonicalize

    molecule = Chem.MolFromSmiles(str(raw_smiles))
    if molecule is None:
        return Rejection("parse_failure_before_hydrogen_removal")
    try:
        molecule = Chem.RemoveHs(molecule, sanitize=True)
        normalized = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
    except Exception:
        return Rejection("explicit_hydrogen_removal_failure")
    policy = cfg["data"]["canonicalization"]
    return canonicalize(
        normalized,
        isomeric_smiles=bool(policy["isomeric_smiles"]),
        fragment_policy=str(policy["fragment_policy"]),
        allowed_elements=set(policy["allowed_elements"]),
        min_atoms=int(policy["min_atoms"]),
        max_atoms=int(policy["max_atoms"]),
        buckets=int(cfg["data"]["hash_buckets"]),
        split_cfg=cfg["data"]["split"],
    )

