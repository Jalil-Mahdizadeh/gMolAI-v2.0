#!/usr/bin/env python3
"""Full-panel fail-closed validation and sustained gMolAI speed benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

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
    calibrate_cpu,
    load_frozen_bundle,
    packed_to_device,
)
from tune import PANEL, PANEL_SHA256, compare, read_smiles, sha256_file


BASELINE_CSV = REPOSITORY_ROOT / "extra-benchmark/speed/outputs/speed_results.csv"
BASELINE_CSV_SHA256 = "52a2abe831ae28fadda8454862e33181c752ea1dcdf86628ecc26e2af5fb86d9"


def matrix_sha256(matrix: np.ndarray) -> str:
    if not matrix.flags.c_contiguous:
        raise RuntimeError("Embedding matrix is not contiguous")
    return hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()


def load_packed_batches(
    values: list[str], batch_size: int, workers: int
):
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
    )
    warm_tasks = [(0, values[:8]) for _ in range(2 * workers)]
    list(executor.map(pack_smiles_task, warm_tasks, chunksize=1))
    return executor, executor.map(
        pack_smiles_task, tasks(values, batch_size), chunksize=1
    )


def reference_batch(values: list[str], bundle) -> np.ndarray:
    graphs = []
    for value in values:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(value)
        x, edge_index, edge_attr = featurize_molecule(
            molecule, include_chirality=True, position_dim=0
        )
        graphs.append(
            Data(
                x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
            )
        )
    batch = Batch.from_data_list(graphs).to(bundle.device)
    with torch.inference_mode():
        node_z, graph_z = bundle.model.encode(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        mean_node_z = global_mean_pool(node_z, batch.batch)
    raw = np.concatenate(
        (
            graph_z.detach().float().cpu().numpy(),
            mean_node_z.detach().float().cpu().numpy(),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return calibrate_cpu(raw, bundle)


def encode_optimized(
    values: list[str], bundle, batch_size: int, workers: int
) -> tuple[np.ndarray, dict]:
    matrix = np.empty((len(values), 384), dtype=np.float32)
    core = ManualScatterRawCore(bundle.model).eval()
    executor, iterator = load_packed_batches(values, batch_size, workers)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        batch_count = 0
        for packed in iterator:
            tensors = packed_to_device(packed, bundle.device)
            with torch.inference_mode():
                raw = core(*tensors)
            matrix[packed.start : packed.start + packed.graph_count] = calibrate_cpu(
                raw.detach().float().cpu().numpy(), bundle
            )
            batch_count += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if not np.isfinite(matrix).all() or np.any(np.all(matrix == 0.0, axis=1)):
        raise RuntimeError("Optimized output contains a non-finite or zero vector")
    return matrix, {
        "batch_size": batch_size,
        "workers": workers,
        "batch_count": batch_count,
        "wall_seconds": elapsed,
        "rows_per_second": len(values) / elapsed,
        "milliseconds_per_molecule": 1000.0 * elapsed / len(values),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "matrix_sha256": matrix_sha256(matrix),
    }


def validate_features(values: list[str]) -> dict:
    started = time.perf_counter()
    atoms = directed_edges = 0
    for index, value in enumerate(values):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise RuntimeError(value)
        expected = featurize_molecule(
            molecule, include_chirality=True, position_dim=0
        )
        observed = fast_featurize_molecule(molecule)
        for label, left, right in zip(
            ("x", "edge_index", "edge_attr"), expected, observed, strict=True
        ):
            if not np.array_equal(left, right):
                raise RuntimeError(
                    f"Feature mismatch at row {index}, {label}, {value!r}"
                )
        atoms += len(expected[0])
        directed_edges += expected[1].shape[1]
    return {
        "graphs": len(values),
        "atoms": atoms,
        "directed_edges": directed_edges,
        "exact_array_equality": True,
        "seconds": time.perf_counter() - started,
    }


def reference_for_batches(values: list[str], bundle, batch_size: int) -> np.ndarray:
    matrix = np.empty((len(values), 384), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        matrix[start:stop] = reference_batch(values[start:stop], bundle)
    return matrix


def baseline_row() -> dict:
    if not BASELINE_CSV.is_file():
        raise RuntimeError("Authoritative speed result CSV is missing")
    if sha256_file(BASELINE_CSV) != BASELINE_CSV_SHA256:
        raise RuntimeError("Authoritative speed result CSV changed")
    with BASELINE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    match = [
        row for row in rows
        if row["model"] == "gmolai" and int(row["batch_size"]) == 64
    ]
    if len(match) != 1:
        raise RuntimeError("Could not identify the authoritative gMolAI batch-64 row")
    return {
        "source": str(BASELINE_CSV.relative_to(REPOSITORY_ROOT)),
        "source_sha256": sha256_file(BASELINE_CSV),
        "batch_size": 64,
        "rows": int(match[0]["rows"]),
        "wall_seconds": float(match[0]["wall_seconds"]),
        "rows_per_second": float(match[0]["rows_per_second"]),
        "host": match[0]["host"],
        "slurm_job_id": match[0]["slurm_job_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--feature-validation", choices=("full", "sample"), default="full")
    parser.add_argument("--reference-validation", choices=("full", "sample"), default="full")
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=SPEED_OPT_DIR / "outputs/final_benchmark.json")
    parser.add_argument("--matrix-output", type=Path, default=SPEED_OPT_DIR / "artifacts/optimized_embeddings.npy")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one GPU is required; visible={torch.cuda.device_count()}")
    if not (1 <= args.workers <= 72):
        raise RuntimeError("Worker count must remain within the 72-CPU allocation")
    torch.use_deterministic_algorithms(True)
    if args.matrix_output.exists() or args.output.exists():
        raise RuntimeError("Refusing to overwrite benchmark output; choose new paths")
    torch.backends.cudnn.benchmark = False
    values = read_smiles(None)
    bundle = load_frozen_bundle()

    feature_values = (
        values if args.feature_validation == "full" else values[: args.sample_rows]
    )
    feature_validation = validate_features(feature_values)

    # Warm the selected model path before the one-pass sustained timer.
    warm = pack_smiles_task((0, values[: args.batch_size]))
    warm_tensors = packed_to_device(warm, bundle.device)
    with torch.inference_mode():
        ManualScatterRawCore(bundle.model).eval()(*warm_tensors)
    torch.cuda.synchronize()

    optimized, timing = encode_optimized(
        values, bundle, args.batch_size, args.workers
    )
    reference_values = (
        values if args.reference_validation == "full" else values[: args.sample_rows]
    )
    reference = reference_for_batches(reference_values, bundle, args.batch_size)
    embedding_comparison_same_batch = compare(
        reference, optimized[: len(reference_values)]
    )
    exact_same_batch_required = args.reference_validation == "full"
    if exact_same_batch_required and not embedding_comparison_same_batch["exact"]:
        raise RuntimeError(
            "Optimized embeddings are not bitwise identical to the authoritative "
            "encoder at the same batch boundaries"
        )
    if (
        embedding_comparison_same_batch["minimum_cosine"] < 0.9999
        or embedding_comparison_same_batch["maximum_relative_l2"] > 0.005
    ):
        raise RuntimeError("Optimized sample failed the frozen stability gate")

    reference64_values = values[: args.sample_rows]
    reference64 = reference_for_batches(reference64_values, bundle, 64)
    comparison_to_reference64 = compare(
        reference64, optimized[: len(reference64_values)]
    )
    if (
        comparison_to_reference64["minimum_cosine"] < 0.9999
        or comparison_to_reference64["maximum_relative_l2"] > 0.005
    ):
        raise RuntimeError("Optimized batch-size output failed the frozen stability gate")

    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.matrix_output, optimized, allow_pickle=False)
    loaded = np.load(args.matrix_output, mmap_mode="r", allow_pickle=False)
    if matrix_sha256(loaded) != timing["matrix_sha256"]:
        raise RuntimeError("Persisted validation matrix hash mismatch")

    baseline = baseline_row()
    result = {
        "schema_version": 1,
        "scope": "inference_only_no_training",
        "timing_boundary": "canonical_smiles_in_ram_to_ordered_fp32_vectors_in_host_ram",
        "panel": {
            "path": str(PANEL.relative_to(REPOSITORY_ROOT)),
            "sha256": PANEL_SHA256,
            "rows": len(values),
        },
        "model": bundle.metadata,
        "runtime": {
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cpu_limit": 72,
            "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "optimization": {
            "exact_reduced_RDKit_feature_factory": True,
            "direct_numpy_graph_packing": True,
            "multiprocess_graph_preprocessing": True,
            "manual_equivalent_GINE_scatter": True,
            "checkpoint_or_calibrator_changed": False,
            "new_container_required": False,
        },
        "feature_validation": feature_validation,
        "embedding_validation": {
            "same_batch": embedding_comparison_same_batch,
            "same_batch_rows": len(reference_values),
            "against_authoritative_batch64": comparison_to_reference64,
            "against_authoritative_batch64_rows": len(reference64_values),
        },
        "baseline": baseline,
        "optimized": timing,
        "speedup_vs_authoritative_batch64": timing["rows_per_second"] / baseline["rows_per_second"],
        "matrix_validation_artifact": str(args.matrix_output.resolve().relative_to(REPOSITORY_ROOT)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
