#!/usr/bin/env python3
"""Audit an embedding matrix against its immutable ordered identity panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR, atomic_write_json, load_json, load_protocol,
    read_panel_tsv, sha256_file, sha256_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True, choices=("classyfire", "qmugs"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    rows = read_panel_tsv(args.input)
    metadata = load_json(args.metadata)
    protocol = load_protocol()
    if args.model in protocol["models"]:
        expected_dimension = int(protocol["models"][args.model]["dimension"])
    elif args.model == "descriptor13":
        expected_dimension = int(protocol["descriptor_diagnostic"]["dimension"])
    else:
        raise ValueError(f"Unknown model {args.model}")
    identity_sha = sha256_lines(row["molecule_hash"] for row in rows)
    if metadata.get("status") != "ok" or metadata.get("model") != args.model:
        raise RuntimeError("Embedding metadata identity differs")
    if metadata.get("input_sha256") != sha256_file(args.input):
        raise RuntimeError("Embedding metadata input hash differs")
    if metadata.get("ordered_identity_sha256") != identity_sha:
        raise RuntimeError("Embedding ordered identity digest differs")
    if metadata.get("output_sha256") != sha256_file(args.embedding):
        raise RuntimeError("Embedding output hash differs")
    matrix = np.load(args.embedding, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(rows), expected_dimension) or matrix.dtype != np.float32:
        raise RuntimeError(f"Unexpected embedding shape/dtype: {matrix.shape}/{matrix.dtype}")
    nonfinite = 0
    zero_norm = 0
    for start in range(0, len(rows), 4096):
        batch = np.asarray(matrix[start:start + 4096], dtype=np.float64)
        nonfinite += int(np.count_nonzero(~np.isfinite(batch).all(axis=1)))
        zero_norm += int(np.count_nonzero(np.linalg.norm(batch, axis=1) <= 1e-12))
    if nonfinite or zero_norm:
        raise RuntimeError(f"Invalid vectors: nonfinite={nonfinite}, zero_norm={zero_norm}")
    result = {
        "schema_version": 1, "status": "ok", "benchmark": args.benchmark,
        "model": args.model, "rows": len(rows), "dimension": expected_dimension,
        "dtype": "float32", "nonfinite_vectors": nonfinite,
        "zero_norm_vectors": zero_norm, "input_sha256": sha256_file(args.input),
        "ordered_identity_sha256": identity_sha,
        "embedding_sha256": sha256_file(args.embedding),
        "metadata_sha256": sha256_file(args.metadata),
    }
    output = BENCHMARK_DIR / "audit" / f"embedding-{args.model}-{args.benchmark}.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

