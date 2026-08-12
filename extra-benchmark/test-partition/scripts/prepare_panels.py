#!/usr/bin/env python3
"""Materialize canonical SMILES for the immutable benchmark identities."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import torch

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_protocol,
    require_hash,
    sha256_file,
    sha256_lines,
    write_panel_tsv,
)


HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-rows", type=int, default=128)
    return parser.parse_args()


def load_embedding_payload(path: Path, expected_rows: int, split: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "metadata",
        "embeddings",
        "standardized_descriptor_targets",
        "graph_ids",
        "source_buckets",
        "molecule_hashes",
    }
    if not isinstance(payload, dict) or required - set(payload):
        raise RuntimeError(f"Malformed authoritative embedding payload: {path}")
    rows = int(payload["embeddings"].shape[0])
    if rows != expected_rows or payload["metadata"].get("split") != split:
        raise RuntimeError(
            f"Authoritative {split} payload identity mismatch: rows={rows}, "
            f"metadata split={payload['metadata'].get('split')!r}"
        )
    aligned_lengths = {
        rows,
        int(payload["standardized_descriptor_targets"].shape[0]),
        int(payload["graph_ids"].numel()),
        int(payload["source_buckets"].numel()),
        len(payload["molecule_hashes"]),
    }
    if aligned_lengths != {rows}:
        raise RuntimeError(f"Misaligned rows in {path}: {sorted(aligned_lengths)}")
    hashes = [str(value) for value in payload["molecule_hashes"]]
    if any(not HASH_PATTERN.fullmatch(value) for value in hashes):
        raise RuntimeError(f"Invalid molecular SHA-256 identity in {path}")
    if len(set(hashes)) != rows:
        raise RuntimeError(f"Duplicate molecular identity in {path}")
    if not torch.isfinite(payload["embeddings"]).all():
        raise RuntimeError(f"Non-finite authoritative embeddings in {path}")
    return payload


def join_chemical_records(
    payloads: dict[str, dict[str, Any]], work_dir: Path
) -> dict[str, tuple[str, str]]:
    wanted_by_bucket: dict[int, set[str]] = defaultdict(set)
    for payload in payloads.values():
        for molecule_hash, bucket in zip(
            payload["molecule_hashes"], payload["source_buckets"].view(-1).tolist()
        ):
            wanted_by_bucket[int(bucket)].add(str(molecule_hash))

    records: dict[str, tuple[str, str]] = {}
    connection = duckdb.connect(":memory:")
    try:
        for bucket, hashes in sorted(wanted_by_bucket.items()):
            parquet = work_dir / "deduplicated" / f"bucket-{bucket:04d}.parquet"
            if not parquet.is_file() or parquet.is_symlink():
                raise FileNotFoundError(parquet)
            ordered_hashes = sorted(hashes)
            connection.register("wanted", pa.table({"molecule_hash": ordered_hashes}))
            rows = connection.execute(
                """
                SELECT d.molecule_hash, d.canonical_smiles, d.scaffold
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet)],
            ).fetchall()
            connection.unregister("wanted")
            for molecule_hash, canonical_smiles, scaffold in rows:
                key = str(molecule_hash)
                smiles = str(canonical_smiles)
                observed = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
                if observed != key:
                    raise RuntimeError(
                        f"Collision-safe identity validation failed for {key}"
                    )
                value = (smiles, str(scaffold or ""))
                previous = records.setdefault(key, value)
                if previous != value:
                    raise RuntimeError(f"Conflicting canonical record for {key}")
    finally:
        connection.close()

    all_hashes = {
        str(value)
        for payload in payloads.values()
        for value in payload["molecule_hashes"]
    }
    missing = sorted(all_hashes - set(records))
    if missing:
        raise RuntimeError(f"Failed to join {len(missing)} immutable identities")
    return records


def panel_rows(
    payload: dict[str, Any], records: dict[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    result = []
    for index, (graph_id, bucket, molecule_hash) in enumerate(
        zip(
            payload["graph_ids"].view(-1).tolist(),
            payload["source_buckets"].view(-1).tolist(),
            payload["molecule_hashes"],
        )
    ):
        smiles, scaffold = records[str(molecule_hash)]
        result.append(
            {
                "panel_index": index,
                "graph_id": int(graph_id),
                "source_bucket": int(bucket),
                "molecule_hash": str(molecule_hash),
                "canonical_smiles": smiles,
                "scaffold": scaffold,
            }
        )
    return result


def qualification_indices(rows: list[dict[str, Any]], count: int) -> np.ndarray:
    if count <= 0 or count > len(rows):
        raise ValueError("qualification row count is outside the validation panel")
    lengths = np.asarray([len(str(row["canonical_smiles"])) for row in rows])
    hashes = np.asarray([str(row["molecule_hash"]) for row in rows], dtype=object)
    order = np.lexsort((hashes, lengths))
    positions = np.linspace(0, len(order) - 1, num=count, dtype=np.int64)
    selected = np.unique(order[positions])
    if len(selected) != count:
        raise RuntimeError("Qualification fixture selection produced duplicate indices")
    return np.sort(selected)


def write_named_panel(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tsv = BENCHMARK_DIR / "inputs" / f"{name}.tsv"
    smi = BENCHMARK_DIR / "inputs" / f"{name}.smi"
    write_panel_tsv(tsv, rows)
    atomic_write_text(smi, "".join(f"{row['canonical_smiles']}\n" for row in rows))
    return {
        "rows": len(rows),
        "tsv": str(tsv.relative_to(REPOSITORY_ROOT)),
        "tsv_sha256": sha256_file(tsv),
        "smiles": str(smi.relative_to(REPOSITORY_ROOT)),
        "smiles_sha256": sha256_file(smi),
        "ordered_identity_sha256": sha256_lines(
            str(row["molecule_hash"]) for row in rows
        ),
        "canonical_smiles_sha256": sha256_lines(
            str(row["canonical_smiles"]) for row in rows
        ),
        "unique_identities": len({str(row["molecule_hash"]) for row in rows}),
    }


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    work_dir = Path(protocol["repository"]["work_dir"])
    specifications = protocol["authoritative_panels"]
    payloads: dict[str, dict[str, Any]] = {}
    for name in ("train", "validation", "test"):
        specification = specifications[name]
        path = REPOSITORY_ROOT / specification["path"]
        require_hash(path, specification["sha256"])
        payloads[name] = load_embedding_payload(
            path, int(specification["rows"]), str(specification["split"])
        )

    records = join_chemical_records(payloads, work_dir)
    train_rows = panel_rows(payloads["train"], records)
    validation_rows = panel_rows(payloads["validation"], records)
    test_rows = panel_rows(payloads["test"], records)
    selected = qualification_indices(validation_rows, args.qualification_rows)
    fixture_rows = [
        {**validation_rows[int(index)], "panel_index": fixture_index}
        for fixture_index, index in enumerate(selected)
    ]

    outputs = {
        "train_10k": write_named_panel("train_10k", train_rows),
        "validation_qualification": write_named_panel(
            "validation_qualification", fixture_rows
        ),
        "test_50k": write_named_panel("test_50k", test_rows),
    }
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "identity_rule": "SHA-256 of canonical isomeric SMILES",
        "collision_safe_validation": True,
        "source_dataset_manifest_sha256": specifications["dataset_manifest"]["sha256"],
        "source_graph_manifest_sha256": specifications["graph_manifest"]["sha256"],
        "qualification_selection": {
            "method": "length_stratified_even_positions_with_hash_tiebreak",
            "source": "pretraining_validation_partition",
            "source_rows": len(validation_rows),
            "selected_source_indices_sha256": sha256_lines(
                str(int(index)) for index in selected
            ),
        },
        "panels": outputs,
    }
    atomic_write_json(BENCHMARK_DIR / "inputs" / "panel_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
