#!/usr/bin/env python3
"""Fail-closed I/O helpers for the TDC ADMET benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = BENCHMARK_DIR.parents[1]
PROTOCOL_PATH = BENCHMARK_DIR / "protocol.json"

PANEL_COLUMNS = (
    "panel_index",
    "graph_id",
    "source_bucket",
    "molecule_hash",
    "canonical_smiles",
    "scaffold",
)
LABEL_COLUMNS = (
    "occurrence_index",
    "panel_index",
    "original_panel_index",
    "endpoint",
    "source_role",
    "source_row_index",
    "drug_id",
    "molecule_hash",
    "target",
    "task",
    "official_metric",
    "category",
    "scaffold",
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def identity_set_sha256(values: Iterable[str]) -> str:
    return sha256_lines(sorted(set(str(value) for value in values)))


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def load_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("schema_version") != 1:
        raise RuntimeError("Unsupported TDC ADMET protocol schema")
    if protocol.get("protocol_status") != "frozen_before_representation_execution":
        raise RuntimeError("TDC ADMET protocol is not frozen")
    return protocol


def protocol_digest(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_hash(path: str | Path, expected: str) -> str:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError(f"Required regular file is missing: {target}")
    observed = sha256_file(target)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {target}: expected {expected}, observed {observed}"
        )
    return observed


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(
    path: str | Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
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


def _write_tsv(
    path: str | Path, rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
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


def write_panel_tsv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_tsv(path, list(rows), PANEL_COLUMNS)


def write_labels_tsv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_tsv(path, list(rows), LABEL_COLUMNS)


def _read_tsv(path: str | Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        return list(reader)


def read_panel_tsv(path: str | Path) -> list[dict[str, str]]:
    rows = _read_tsv(path, PANEL_COLUMNS)
    for expected, row in enumerate(rows):
        if int(row["panel_index"]) != expected:
            raise RuntimeError(f"Non-contiguous panel index in {path} at {expected}")
    return rows


def read_labels_tsv(path: str | Path) -> list[dict[str, str]]:
    rows = _read_tsv(path, LABEL_COLUMNS)
    for expected, row in enumerate(rows):
        if int(row["occurrence_index"]) != expected:
            raise RuntimeError(f"Non-contiguous occurrence index in {path} at {expected}")
    return rows


def atomic_savez(path: str | Path, arrays: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            import numpy as np

            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
