#!/usr/bin/env python3
"""Export the frozen count-Morgan or 13-descriptor diagnostic vectors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_json,
    load_protocol,
    read_panel_tsv,
    require_regular_file,
    sha256_file,
    sha256_lines,
)


GENERATORS = {
    "qed": Descriptors.qed,
    "MolWt": Descriptors.MolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "BalabanJ": Descriptors.BalabanJ,
    "LabuteASA": Descriptors.LabuteASA,
    "TPSA": Descriptors.TPSA,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("morgan_count", "descriptor13"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("Refusing to overwrite a diagnostic representation")
    rows = read_panel_tsv(args.input)
    if not rows:
        raise RuntimeError("Cannot encode an empty panel")
    protocol = load_protocol()
    if args.kind == "morgan_count":
        dimension = int(protocol["models"]["morgan_count"]["dimension"])
        batch_size = int(protocol["models"]["morgan_count"]["batch_size"])
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=dimension)

        def encode(values):
            result = []
            for value in values:
                molecule = Chem.MolFromSmiles(value)
                if molecule is None:
                    raise ValueError("Count Morgan parse failure")
                counts = generator.GetCountFingerprintAsNumPy(molecule)
                result.append(np.log1p(counts.astype(np.float32)))
            return np.asarray(result, dtype=np.float32)

        implementation = {
            "radius": 2, "fp_size": dimension, "count": True,
            "stored_transform": "log1p"
        }
    else:
        descriptor = protocol["descriptor_diagnostic"]
        names = list(descriptor["ordered_features"])
        dimension = len(names)
        batch_size = 1024
        scaler_path = REPOSITORY_ROOT / descriptor["scaler"]["path"]
        require_regular_file(scaler_path, descriptor["scaler"]["sha256"])
        scaler = load_json(scaler_path)
        if scaler["descriptor_names"] != names:
            raise RuntimeError("Frozen descriptor order changed")
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        scale = np.asarray(scaler["scale"], dtype=np.float64)

        def encode(values):
            result = np.empty((len(values), dimension), dtype=np.float32)
            for row_index, value in enumerate(values):
                molecule = Chem.MolFromSmiles(value)
                if molecule is None:
                    raise ValueError("Descriptor parse failure")
                raw = np.asarray([GENERATORS[name](molecule) for name in names], dtype=np.float64)
                raw[~np.isfinite(raw)] = mean[~np.isfinite(raw)]
                result[row_index] = ((raw - mean) / scale).astype(np.float32)
            return result

        implementation = {
            "descriptor_names": names,
            "scaler_sha256": sha256_file(scaler_path),
            "scaler_fit_rows": int(scaler["count"]),
        }
    fixture = [row["canonical_smiles"] for row in rows[:2]]
    first = encode(fixture)
    second = encode(fixture)
    if not np.array_equal(first, second):
        raise RuntimeError("Diagnostic deterministic-repeat check failed")
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(rows), dimension)
    )
    try:
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            batch = encode([row["canonical_smiles"] for row in rows[start:stop]])
            if batch.shape != (stop - start, dimension) or not np.isfinite(batch).all():
                raise RuntimeError(f"Invalid diagnostic batch {start}:{stop}")
            norms = np.linalg.norm(batch.astype(np.float64), axis=1)
            if np.any(norms <= 1.0e-12):
                raise RuntimeError(f"Zero-norm diagnostic vector in {start}:{stop}")
            matrix[start:stop] = batch
        matrix.flush()
        del matrix
        temporary.replace(args.output)
    except Exception:
        del matrix
        temporary.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1, "status": "ok", "execution": "deterministic_diagnostic",
        "model": args.kind, "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "ordered_identity_sha256": sha256_lines(row["molecule_hash"] for row in rows),
        "output": str(args.output), "output_sha256": sha256_file(args.output),
        "rows": len(rows), "dimension": dimension, "dtype": "float32",
        "batch_size": batch_size, "fixed_batch_deterministic_repeat": True,
        "wall_seconds": elapsed, "rows_per_second": len(rows) / elapsed,
        "implementation": implementation, "python": platform.python_version(),
        "host": platform.node(), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.metadata, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

