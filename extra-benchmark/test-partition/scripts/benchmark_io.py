#!/usr/bin/env python3
"""Shared fail-closed I/O helpers for the locked-test benchmark."""

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


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
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
        raise RuntimeError("Unsupported benchmark protocol schema")
    if protocol.get("protocol_status") != "frozen_before_locked_comparator_execution":
        raise RuntimeError("Benchmark protocol is not frozen")
    return protocol


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
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


def write_panel_tsv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
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
                fieldnames=PANEL_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_panel_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PANEL_COLUMNS:
            raise RuntimeError(
                f"Unexpected panel columns in {path}: {reader.fieldnames!r}"
            )
        rows = list(reader)
    for expected, row in enumerate(rows):
        if int(row["panel_index"]) != expected:
            raise RuntimeError(f"Non-contiguous panel index in {path} at row {expected}")
    return rows


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


def protocol_digest(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
