#!/usr/bin/env python3
"""Profile and tune gMolAI inference without changing scientific artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data

SCRIPT_DIR = Path(__file__).resolve().parent
SPEED_OPT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = SPEED_OPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gmolai_retrain.chem import featurize_molecule
from fast_graph import (
    fast_featurize_molecule,
    initialize_worker,
    pack_smiles_task,
    tasks,
)
from model_core import (
    ManualScatterRawCore,
    ReferenceRawCore,
    ReusedPoolRawCore,
    calibrate_cpu,
    load_frozen_bundle,
    packed_to_device,
    reference_encode,
)


PANEL = REPOSITORY_ROOT / "extra-benchmark/speed/inputs/common_test.tsv"
PANEL_SHA256 = "fac4a8abffd0431b36245c6b6eaa447ce1f1628373cad590e59e7a7e2a0fc18e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_smiles(limit: int | None) -> list[str]:
    if sha256_file(PANEL) != PANEL_SHA256:
        raise RuntimeError("Frozen common locked-test panel changed")
    with PANEL.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 49_844:
        raise RuntimeError("Frozen panel row count changed")
    values = [row["canonical_smiles"] for row in rows]
    return values if limit is None else values[:limit]


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    delta = right - left
    relative = np.linalg.norm(delta, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1), np.finfo(np.float64).tiny
    )
    cosine = np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1),
        np.finfo(np.float64).tiny,
    )
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "maximum_absolute_delta": float(np.max(np.abs(delta))),
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2_p99": float(np.quantile(relative, 0.99)),
        "maximum_relative_l2": float(np.max(relative)),
        "minimum_cosine": float(np.min(cosine)),
    }


def profile_stages(values: list[str], bundle, batch_size: int) -> dict:
    from rdkit import Chem

    totals = {key: 0.0 for key in ("parse_feature", "pyg_collate", "h2d", "forward", "d2h_calibrate")}
    rows = min(len(values), 4096)
    core = ReferenceRawCore(bundle.model).eval()
    for start in range(0, rows, batch_size):
        batch_values = values[start : start + batch_size]
        tick = time.perf_counter()
        features = []
        for value in batch_values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise RuntimeError(value)
            features.append(featurize_molecule(molecule, include_chirality=True, position_dim=0))
        totals["parse_feature"] += time.perf_counter() - tick

        tick = time.perf_counter()
        graphs = [Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei), edge_attr=torch.from_numpy(ea)) for x, ei, ea in features]
        cpu_batch = Batch.from_data_list(graphs)
        totals["pyg_collate"] += time.perf_counter() - tick

        tick = time.perf_counter()
        gpu_batch = cpu_batch.to(bundle.device)
        torch.cuda.synchronize()
        totals["h2d"] += time.perf_counter() - tick

        tick = time.perf_counter()
        with torch.inference_mode():
            raw = core(gpu_batch.x, gpu_batch.edge_index, gpu_batch.edge_attr, gpu_batch.batch)
        torch.cuda.synchronize()
        totals["forward"] += time.perf_counter() - tick

        tick = time.perf_counter()
        calibrate_cpu(raw.float().cpu().numpy(), bundle)
        totals["d2h_calibrate"] += time.perf_counter() - tick
    totals["rows"] = rows
    totals["seconds_total"] = sum(value for key, value in totals.items() if key not in {"rows", "seconds_total"})
    totals["rows_per_second"] = rows / totals["seconds_total"]
    return totals


def validate_featurizer(values: list[str]) -> dict:
    from rdkit import Chem

    started = time.perf_counter()
    graphs = atoms = edges = 0
    for value in values:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(value)
        expected = featurize_molecule(molecule, include_chirality=True, position_dim=0)
        observed = fast_featurize_molecule(molecule)
        for label, left, right in zip(("x", "edge_index", "edge_attr"), expected, observed, strict=True):
            if not np.array_equal(left, right):
                raise RuntimeError(f"Fast featurizer differs at graph {graphs}, array {label}, SMILES {value!r}")
        graphs += 1
        atoms += len(expected[0])
        edges += expected[1].shape[1]
    return {
        "graphs": graphs,
        "atoms": atoms,
        "directed_edges": edges,
        "exact_array_equality": True,
        "seconds": time.perf_counter() - started,
    }


def encode_packed(values: list[str], bundle, batch_size: int, core) -> tuple[np.ndarray, float, int]:
    matrix = np.empty((len(values), 384), dtype=np.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for task in tasks(values, batch_size):
        packed = pack_smiles_task(task)
        tensors = packed_to_device(packed, bundle.device)
        with torch.inference_mode():
            raw = core(*tensors)
        matrix[packed.start : packed.start + packed.graph_count] = calibrate_cpu(
            raw.detach().float().cpu().numpy(), bundle
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return matrix, elapsed, int(torch.cuda.max_memory_allocated())


def encode_parallel(
    values: list[str], bundle, batch_size: int, workers: int, core
) -> tuple[np.ndarray, float, int]:
    """Overlap exact multiprocess graph building with ordered GPU inference."""

    matrix = np.empty((len(values), 384), dtype=np.float32)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
    ) as executor:
        # Process creation and feature-factory initialization are setup costs;
        # no target-panel graph is cached before the sustained timer begins.
        warm_tasks = [(0, values[:8]) for _ in range(2 * workers)]
        list(executor.map(pack_smiles_task, warm_tasks, chunksize=1))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        iterator = executor.map(pack_smiles_task, tasks(values, batch_size), chunksize=1)
        for packed in iterator:
            tensors = packed_to_device(packed, bundle.device)
            with torch.inference_mode():
                raw = core(*tensors)
            matrix[packed.start : packed.start + packed.graph_count] = calibrate_cpu(
                raw.detach().float().cpu().numpy(), bundle
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
    return matrix, elapsed, peak

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8192)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--validate-features", type=int, default=8192, help="0 disables; -1 validates the full panel")
    parser.add_argument("--parallel-workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--parallel-batch-sizes", type=int, nargs="+", default=[64, 256, 512, 1024])
    parser.add_argument("--output", type=Path, default=SPEED_OPT_DIR / "outputs/tuning.json")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one GPU is required; visible={torch.cuda.device_count()}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    values = read_smiles(args.limit)
    result = {
        "panel_sha256": PANEL_SHA256,
        "rows": len(values),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    if args.validate_features:
        feature_values = read_smiles(None if args.validate_features < 0 else args.validate_features)
        result["feature_validation"] = validate_featurizer(feature_values)

    bundle = load_frozen_bundle()
    result["model"] = bundle.metadata
    result["stage_profile_batch64"] = profile_stages(values, bundle, 64)

    reference_rows = min(len(values), 2048)
    reference = np.empty((reference_rows, 384), dtype=np.float32)
    for start in range(0, reference_rows, 64):
        stop = min(start + 64, reference_rows)
        reference[start:stop] = reference_encode(values[start:stop], bundle)

    candidates = []
    for core_name, core in (
        ("reference_pool", ReferenceRawCore(bundle.model).eval()),
        ("reused_pool", ReusedPoolRawCore(bundle.model).eval()),
        ("manual_scatter", ManualScatterRawCore(bundle.model).eval()),
    ):
        for batch_size in args.batch_sizes:
            matrix, elapsed, peak = encode_packed(values, bundle, batch_size, core)
            comparison = compare(reference, matrix[:reference_rows])
            candidates.append(
                {
                    "core": core_name,
                    "batch_size": batch_size,
                    "seconds": elapsed,
                    "rows_per_second": len(values) / elapsed,
                    "peak_gpu_memory_bytes": peak,
                    "comparison_to_reference_batch64_first_rows": comparison,
                }
            )
            print(json.dumps(candidates[-1], sort_keys=True), flush=True)
    result["candidates"] = candidates

    parallel_candidates = []
    parallel_core = ManualScatterRawCore(bundle.model).eval()
    for workers in args.parallel_workers:
        for batch_size in args.parallel_batch_sizes:
            matrix, elapsed, peak = encode_parallel(
                values, bundle, batch_size, workers, parallel_core
            )
            comparison = compare(reference, matrix[:reference_rows])
            parallel_candidates.append(
                {
                    "core": "manual_scatter",
                    "batch_size": batch_size,
                    "workers": workers,
                    "seconds": elapsed,
                    "rows_per_second": len(values) / elapsed,
                    "peak_gpu_memory_bytes": peak,
                    "comparison_to_reference_batch64_first_rows": comparison,
                }
            )
            print(json.dumps(parallel_candidates[-1], sort_keys=True), flush=True)
    result["parallel_candidates"] = parallel_candidates
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

