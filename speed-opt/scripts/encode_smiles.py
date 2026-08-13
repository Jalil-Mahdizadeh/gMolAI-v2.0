#!/usr/bin/env python3
"""Fast, fail-closed gMolAI encoder for canonical SMILES in a TSV file."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SPEED_OPT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = SPEED_OPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fast_graph import initialize_worker, pack_smiles_task, tasks
from model_core import (
    ManualScatterRawCore,
    calibrate_cpu,
    load_frozen_bundle,
    packed_to_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    metadata_path = args.metadata or args.output.with_suffix(
        args.output.suffix + ".metadata.json"
    )
    if metadata_path.exists() or metadata_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {metadata_path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one GPU is required; visible={torch.cuda.device_count()}")
    if args.batch_size <= 0 or not (1 <= args.workers <= 72):
        raise ValueError("batch size must be positive and workers must be in 1..72")

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if args.smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing TSV column {args.smiles_column!r}")
        values = [str(row[args.smiles_column]) for row in reader]
    if not values:
        raise ValueError("Input contains no molecules")
    if len(values) < 8:
        raise ValueError("At least eight molecules are required for worker warm-up")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    bundle = load_frozen_bundle()
    core = ManualScatterRawCore(bundle.model).eval()
    matrix = np.empty((len(values), 384), dtype=np.float32)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=initialize_worker,
    ) as executor:
        warm_tasks = [(0, values[:8]) for _ in range(2 * args.workers)]
        list(executor.map(pack_smiles_task, warm_tasks, chunksize=1))
        iterator = executor.map(
            pack_smiles_task, tasks(values, args.batch_size), chunksize=1
        )
        for packed in iterator:
            tensors = packed_to_device(packed, bundle.device)
            with torch.inference_mode():
                raw = core(*tensors)
            matrix[packed.start : packed.start + packed.graph_count] = calibrate_cpu(
                raw.detach().float().cpu().numpy(), bundle
            )
    if not np.isfinite(matrix).all() or np.any(np.all(matrix == 0, axis=1)):
        raise RuntimeError("Encoder produced a non-finite or zero vector")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix, allow_pickle=False)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": str(args.input.resolve()),
                "input_sha256": sha256_file(args.input),
                "smiles_column": args.smiles_column,
                "rows": len(values),
                "dimensions": 384,
                "dtype": "float32",
                "batch_size": args.batch_size,
                "workers": args.workers,
                "checkpoint_sha256": bundle.metadata["checkpoint_sha256"],
                "calibrator_sha256": bundle.metadata["calibrator_sha256"],
                "output_sha256": hashlib.sha256(
                    memoryview(matrix).cast("B")
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"encoded {len(values)} molecules -> {args.output}")


if __name__ == "__main__":
    main()
