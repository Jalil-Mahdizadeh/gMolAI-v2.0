#!/usr/bin/env python3
"""Validate one embedding and its identity-bound metadata before reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_io import load_json, load_protocol, read_panel_tsv, sha256_file, sha256_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.model == "descriptor_13":
        dimension = int(protocol["diagnostic_control"]["dimension"])
    else:
        dimension = int(protocol["comparators"]["models"][args.model]["dimension"])
    panel_path = args.metadata.parents[2] / "inputs" / "common_panel.tsv"
    rows = read_panel_tsv(panel_path)
    metadata = load_json(args.metadata)
    expected_identity = sha256_lines(row["molecule_hash"] for row in rows)
    if metadata.get("status") != "ok" or metadata.get("model") != args.model:
        raise RuntimeError(f"Invalid metadata status for {args.model}")
    if metadata.get("input_sha256") != sha256_file(panel_path):
        raise RuntimeError(f"{args.model} metadata points to another panel")
    if metadata.get("ordered_identity_sha256") != expected_identity:
        raise RuntimeError(f"{args.model} identity order changed")
    if metadata.get("output_sha256") != sha256_file(args.embedding):
        raise RuntimeError(f"{args.model} embedding checksum changed")
    matrix = np.load(args.embedding, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(rows), dimension) or matrix.dtype != np.float32:
        raise RuntimeError(
            f"{args.model} matrix {matrix.shape}/{matrix.dtype}; "
            f"expected {(len(rows), dimension)}/float32"
        )
    for start in range(0, len(rows), 4096):
        values = np.asarray(matrix[start : start + 4096])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite {args.model} values at {start}")
        if np.any(np.linalg.norm(values.astype(np.float64), axis=1) <= 1.0e-12):
            raise RuntimeError(f"Zero-norm {args.model} values at {start}")
    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                "rows": len(rows),
                "dimension": dimension,
                "embedding_sha256": sha256_file(args.embedding),
                "identity_sha256": expected_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
