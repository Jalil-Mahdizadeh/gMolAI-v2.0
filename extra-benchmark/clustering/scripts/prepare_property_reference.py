#!/usr/bin/env python3
"""Freeze QMugs robust-property coordinates, size strata, nulls, and exact neighbors."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math

import numpy as np

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_protocol, read_panel_tsv, sha256_file, write_csv
from metrics_common import atomic_save_npz, exact_property_knn


PROPERTY_NAMES = ("DFT_HOMO_ENERGY", "DFT_HOMO_LUMO_GAP", "log1p_DFT_DIPOLE_TOT")


def decile_null(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Property-null computation requires exactly one visible GPU")
    result = np.empty(len(values), dtype=np.float64)
    device = torch.device("cuda:0")
    with torch.inference_mode():
        for decile in range(10):
            indices = np.flatnonzero(strata == decile)
            block = torch.as_tensor(values[indices], dtype=torch.float64, device=device)
            squared = torch.sum(block * block, dim=1)
            distance2 = squared[:, None] + squared[None, :] - 2.0 * (block @ block.T)
            distance = torch.sqrt(torch.clamp(distance2, min=0.0))
            distance.fill_diagonal_(0.0)
            result[indices] = (torch.sum(distance, dim=1) / (len(indices) - 1)).cpu().numpy()
            del block, squared, distance2, distance
    torch.cuda.empty_cache()
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError("Invalid size-matched QMugs null")
    return result


def main() -> None:
    protocol = load_protocol()
    panel_path = BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv"
    rows = read_panel_tsv(panel_path)
    raw = np.column_stack([
        np.asarray([float(row["DFT_HOMO_ENERGY"]) for row in rows]),
        np.asarray([float(row["DFT_HOMO_LUMO_GAP"]) for row in rows]),
        np.log1p(np.asarray([float(row["DFT_DIPOLE_TOT"]) for row in rows])),
    ]).astype(np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError("Nonfinite QMugs property after frozen transforms")
    q1 = np.quantile(raw, 0.25, axis=0, method="linear")
    median = np.quantile(raw, 0.50, axis=0, method="linear")
    q3 = np.quantile(raw, 0.75, axis=0, method="linear")
    iqr = q3 - q1
    if np.any(iqr <= 0):
        raise RuntimeError("Zero QMugs property IQR")
    scaled = (raw - median) / iqr
    heavy = np.asarray([int(row["heavy_atom_count"]) for row in rows], dtype=np.int16)
    order = np.asarray(sorted(range(len(rows)), key=lambda index: (int(heavy[index]), rows[index]["molecule_hash"])), dtype=np.int32)
    strata = np.empty(len(rows), dtype=np.int8)
    decile_rows = []
    for decile, indices in enumerate(np.array_split(order, 10)):
        strata[indices] = decile
        decile_rows.append({
            "heavy_atom_decile": decile + 1, "rows": len(indices),
            "minimum_heavy_atoms": int(np.min(heavy[indices])),
            "maximum_heavy_atoms": int(np.max(heavy[indices])),
        })
    k = int(protocol["property_evaluation"]["k"])
    neighbors = exact_property_knn(scaled, k)
    null = decile_null(scaled, strata)
    output = BENCHMARK_DIR / "artifacts" / "common" / "qmugs_property_reference.npz"
    atomic_save_npz(
        output, raw_properties=raw, robust_properties=scaled,
        property_neighbors=neighbors, heavy_atom_decile=strata,
        null_mean_distance=null, heavy_atom_count=heavy,
        property_names=np.asarray(PROPERTY_NAMES),
    )
    scaling_rows = [
        {
            "property": PROPERTY_NAMES[index],
            "source_column": ("DFT_HOMO_ENERGY", "DFT_HOMO_LUMO_GAP", "DFT_DIPOLE_TOT")[index],
            "transform": ("identity", "identity", "log1p")[index],
            "q1": q1[index], "median": median[index], "q3": q3[index], "iqr": iqr[index],
            "quantile_method": "linear",
        }
        for index in range(3)
    ]
    scaling_path = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_scaling.csv"
    write_csv(scaling_path, scaling_rows, tuple(scaling_rows[0]))
    decile_path = BENCHMARK_DIR / "outputs" / "tables" / "qmugs_heavy_atom_deciles.csv"
    write_csv(decile_path, decile_rows, tuple(decile_rows[0]))
    report = {
        "schema_version": 1, "status": "ok", "rows": len(rows), "k": k,
        "properties": list(PROPERTY_NAMES), "median": median.tolist(), "iqr": iqr.tolist(),
        "quantile_method": "linear", "reference": str(output),
        "reference_sha256": sha256_file(output), "scaling_table": str(scaling_path),
        "scaling_table_sha256": sha256_file(scaling_path), "decile_table": str(decile_path),
        "decile_table_sha256": sha256_file(decile_path),
        "null_definition": protocol["property_evaluation"]["npd_null"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "property_reference.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

