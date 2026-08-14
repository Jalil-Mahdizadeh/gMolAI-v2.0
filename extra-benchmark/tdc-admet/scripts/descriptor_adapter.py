#!/usr/bin/env python3
"""Export the frozen 13-property RDKit diagnostic panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from rdkit import Chem, __version__ as rdkit_version
from rdkit.Chem import Descriptors

from benchmark_io import (
    atomic_write_json,
    load_protocol,
    read_panel_tsv,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("Refusing to overwrite descriptor outputs")
    protocol = load_protocol()
    names = protocol["diagnostic_control"]["ordered_features"]
    if set(names) - set(GENERATORS):
        raise RuntimeError("Frozen descriptor generator mapping is incomplete")
    rows = read_panel_tsv(args.input)
    started = time.perf_counter()
    values = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise RuntimeError("Descriptor panel contains an unparsable SMILES")
        values.append([float(GENERATORS[name](molecule)) for name in names])
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
        raise RuntimeError("Descriptor calculation produced an invalid matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output exists: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "status": "ok",
        "execution": "deterministic_rdkit_descriptors",
        "model": "descriptor_13",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "ordered_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in rows
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "dimension": int(matrix.shape[1]),
        "dtype": "float32",
        "ordered_features": names,
        "wall_seconds_model_load_warmup_and_export": elapsed,
        "rows_per_second_including_load_warmup_and_export": len(rows) / elapsed,
        "peak_gpu_memory_bytes": None,
        "gpu_name": None,
        "rdkit": rdkit_version,
        "python": platform.python_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.metadata, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
