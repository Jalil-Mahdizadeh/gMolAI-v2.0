#!/usr/bin/env python3
"""Validate one frozen common-panel representation matrix and sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    load_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    models = protocol["comparators"]["models"]
    if args.model not in models:
        raise KeyError(f"Unknown comparator: {args.model}")
    specification = models[args.model]
    panel_path = BENCHMARK_DIR / "inputs" / "common_panel.tsv"
    rows = read_panel_tsv(panel_path)
    metadata = load_json(args.metadata)
    expected_identity = sha256_lines(row["molecule_hash"] for row in rows)
    expected_execution = (
        "inference_only_frozen_encoder" if args.model == "gmolai" else "inference_only"
    )
    checks = {
        "status": metadata.get("status") == "ok",
        "execution": metadata.get("execution") == expected_execution,
        "model": metadata.get("model") == args.model,
        "input_sha256": metadata.get("input_sha256") == sha256_file(panel_path),
        "ordered_identity_sha256": metadata.get("ordered_identity_sha256")
        == expected_identity,
        "rows": int(metadata.get("rows", -1)) == len(rows),
        "dimension": int(metadata.get("dimension", -1))
        == int(specification["dimension"]),
        "dtype": metadata.get("dtype") == "float32",
        "output_sha256": metadata.get("output_sha256")
        == sha256_file(args.embedding),
        "deterministic": metadata.get("fixed_batch_deterministic_repeat") is True,
        "single_gpu": int(metadata.get("visible_gpu_count", -1))
        == (0 if args.model == "morgan" else 1),
    }
    if args.model not in {"gmolai", "morgan"}:
        checks["batch_size"] = int(metadata.get("batch_size", -1)) == int(
            specification["batch_size"]
        )
    if args.model == "gmolai":
        checks["checkpoint"] = (
            metadata.get("checkpoint_sha256")
            == protocol["sources"]["checkpoint"]["sha256"]
        )
        checks["calibrator"] = (
            metadata.get("calibrator_sha256")
            == protocol["sources"]["calibrator"]["sha256"]
        )
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"{args.model} adapter metadata failed: {failed}")

    matrix = np.load(args.embedding, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(rows), int(specification["dimension"]))
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise RuntimeError(
            f"{args.model} matrix {matrix.shape}/{matrix.dtype}; expected {expected_shape}/float32"
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
                "dimension": int(matrix.shape[1]),
                "embedding_sha256": sha256_file(args.embedding),
                "identity_sha256": expected_identity,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
