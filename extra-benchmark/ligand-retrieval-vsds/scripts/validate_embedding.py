#!/usr/bin/env python3
"""Validate one representation export against its model-specific identity panel."""

from __future__ import annotations

import argparse
import json

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    load_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.model not in protocol["models"]["primary_order"]:
        raise ValueError(f"Unknown model: {args.model}")
    panel = BENCHMARK_DIR / "inputs/prepared/model_panels" / f"{args.model}.tsv"
    matrix_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{args.model}.npy"
    metadata_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{args.model}.json"
    rows = read_panel_tsv(panel)
    metadata = load_json(metadata_path)
    expected_dimension = int(protocol["models"][args.model]["dimension"])
    identity_sha = sha256_lines(row["molecule_hash"] for row in rows)
    if metadata.get("status") != "ok" or metadata.get("model") != args.model:
        raise RuntimeError("Embedding metadata model/status differs")
    if metadata.get("input_sha256") != sha256_file(panel):
        raise RuntimeError("Embedding metadata input hash differs")
    if metadata.get("ordered_identity_sha256") != identity_sha:
        raise RuntimeError("Embedding ordered identity digest differs")
    if metadata.get("output_sha256") != sha256_file(matrix_path):
        raise RuntimeError("Embedding output hash differs")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(rows), expected_dimension) or matrix.dtype != np.float32:
        raise RuntimeError(f"Unexpected embedding shape/dtype: {matrix.shape}/{matrix.dtype}")
    nonfinite = 0
    zero_norm = 0
    nonbinary = 0
    for start in range(0, len(rows), 2048):
        batch = np.asarray(matrix[start : start + 2048])
        finite_rows = np.isfinite(batch).all(axis=1)
        nonfinite += int(np.count_nonzero(~finite_rows))
        norms = np.linalg.norm(batch.astype(np.float64), axis=1)
        zero_norm += int(np.count_nonzero(norms <= 1.0e-12))
        if args.model == "morgan":
            nonbinary += int(np.count_nonzero((batch != 0.0) & (batch != 1.0)))
    if nonfinite or zero_norm or nonbinary:
        raise RuntimeError(
            f"Invalid {args.model} vectors: nonfinite={nonfinite}, "
            f"zero_norm={zero_norm}, nonbinary_values={nonbinary}"
        )
    result = {
        "schema_version": 1,
        "status": "ok",
        "model": args.model,
        "rows": len(rows),
        "dimension": expected_dimension,
        "dtype": "float32",
        "nonfinite_vectors": nonfinite,
        "zero_norm_vectors": zero_norm,
        "nonbinary_values": nonbinary if args.model == "morgan" else None,
        "input_sha256": sha256_file(panel),
        "ordered_identity_sha256": identity_sha,
        "embedding_sha256": sha256_file(matrix_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    output = BENCHMARK_DIR / "audits" / f"embedding-{args.model}.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

