#!/usr/bin/env python3
"""Validate one embedding and its identity-bound metadata before reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_io import (
    identity_set_sha256,
    load_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)


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
    nan_row_indices: list[int] = []
    nan_columns: set[int] = set()
    nan_values = 0
    for start in range(0, len(rows), 4096):
        values = np.asarray(matrix[start : start + 4096])
        if args.model == "descriptor_13":
            if np.isinf(values).any():
                raise RuntimeError(f"Infinite descriptor value at {start}")
            mask = np.isnan(values)
            nan_values += int(mask.sum())
            nan_columns.update(np.flatnonzero(mask.any(axis=0)).tolist())
            nan_row_indices.extend(
                (start + int(index)) for index in np.flatnonzero(mask.any(axis=1))
            )
            norm_values = np.nan_to_num(values, nan=0.0)
        else:
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite {args.model} values at {start}")
            norm_values = values
        if np.any(
            np.linalg.norm(norm_values.astype(np.float64), axis=1) <= 1.0e-12
        ):
            raise RuntimeError(f"Zero-norm {args.model} values at {start}")
    if args.model == "descriptor_13":
        amendment = protocol["runtime_amendments"][-1]
        names = protocol["diagnostic_control"]["ordered_features"]
        expected_columns = {
            names.index(name) for name in amendment["allowed_nan_features"]
        }
        observed_identities = {
            rows[index]["molecule_hash"] for index in nan_row_indices
        }
        expected_identities = set(amendment["affected_identities"])
        if identity_set_sha256(expected_identities) != amendment[
            "affected_unique_identity_sha256"
        ]:
            raise RuntimeError("Frozen descriptor amendment identity digest is invalid")
        if (
            observed_identities != expected_identities
            or nan_columns != expected_columns
            or nan_values != int(amendment["expected_nan_values"])
            or metadata.get("nan_values") != nan_values
        ):
            raise RuntimeError("Descriptor NaNs differ from the frozen amendment")
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
