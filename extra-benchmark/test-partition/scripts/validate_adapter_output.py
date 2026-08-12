#!/usr/bin/env python3
"""Validate a frozen adapter's NPY matrix and provenance sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_io import load_json, load_protocol, read_panel_tsv, sha256_file, sha256_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if args.model not in protocol["comparators"]:
        raise KeyError(f"Unknown frozen comparator: {args.model}")
    specification = protocol["comparators"][args.model]
    rows = read_panel_tsv(args.input)
    metadata = load_json(args.metadata)
    expected_identity = sha256_lines(row["molecule_hash"] for row in rows)
    checks = {
        "status": metadata.get("status") == "ok",
        "execution": metadata.get("execution") == "inference_only",
        "model": metadata.get("model") == args.model,
        "input_sha256": metadata.get("input_sha256") == sha256_file(args.input),
        "ordered_identity_sha256": metadata.get("ordered_identity_sha256")
        == expected_identity,
        "rows": int(metadata.get("rows", -1)) == len(rows),
        "dimension": int(metadata.get("dimension", -1))
        == int(specification["dimension"]),
        "batch_size": int(metadata.get("batch_size", -1))
        == int(specification["batch_size"]),
        "dtype": metadata.get("dtype") == "float32",
        "output_sha256": metadata.get("output_sha256")
        == sha256_file(args.embedding),
        "deterministic": metadata.get("fixed_batch_deterministic_repeat") is True,
    }
    if args.model not in {"morgan", "gmolai"}:
        checks["single_gpu"] = int(metadata.get("visible_gpu_count", -1)) == 1
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Adapter metadata validation failed: {failed}")

    matrix = np.load(args.embedding, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(rows), int(specification["dimension"]))
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise RuntimeError(
            f"Embedding matrix contract failed: {matrix.shape}/{matrix.dtype}, "
            f"expected {expected_shape}/float32"
        )
    block = 4096
    for start in range(0, len(rows), block):
        values = np.asarray(matrix[start : start + block])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite embedding values at rows {start}:{start+block}")
        norms = np.linalg.norm(values.astype(np.float64), axis=1)
        if np.any(norms <= 1.0e-12):
            raise RuntimeError(f"Zero-norm embedding values at rows {start}:{start+block}")
    report = {
        "status": "ok",
        "model": args.model,
        "rows": len(rows),
        "dimension": int(specification["dimension"]),
        "embedding_sha256": sha256_file(args.embedding),
        "identity_sha256": expected_identity,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
