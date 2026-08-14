#!/usr/bin/env python3
"""Export the frozen released gMolAI representation for the TDC common panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from rdkit import Chem
import torch

from benchmark_io import (
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gmolai_retrain.config import apply_training_plan, load_config  # noqa: E402
from gmolai_retrain.downstream import (  # noqa: E402
    _encode_molecules,
    _select_representation_embedding,
)
from gmolai_retrain.representations import (  # noqa: E402
    _calibrator_expected_identity,
    _load_embedding_calibrator,
    load_saved_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    return parser.parse_args()


def selected_embedding(
    model: torch.nn.Module,
    molecules: list[Chem.Mol],
    cfg: dict,
    device: torch.device,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    unit_hybrid, blocks = _encode_molecules(model, molecules, cfg, device)
    values = _select_representation_embedding(
        "standardized_raw_hybrid",
        unit_hybrid,
        blocks,
        calibration_mean=mean,
        calibration_scale=scale,
    )
    return np.asarray(values, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("Refusing to overwrite a gMolAI embedding output")
    rows = read_panel_tsv(args.input)
    if not rows:
        raise RuntimeError("Cannot encode an empty panel")
    protocol = load_protocol()
    source = protocol["gmolai"]
    expected = protocol["comparators"]["models"]["gmolai"]

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-GPU contract violated: visible GPUs={torch.cuda.device_count()}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")

    config_path = REPOSITORY_ROOT / source["config"]["path"]
    plan_path = REPOSITORY_ROOT / source["training_plan"]["path"]
    run_dir = REPOSITORY_ROOT / source["checkpoint"]["run_dir"]
    checkpoint_name = source["checkpoint"]["name"]
    checkpoint_path = run_dir / checkpoint_name
    calibrator_path = REPOSITORY_ROOT / source["calibrator"]["path"]

    started = time.perf_counter()
    cfg = load_config(config_path)
    apply_training_plan(cfg, plan_path)
    cfg["paths"]["run_dir"] = str(run_dir)
    cfg["experiment_name"] = run_dir.name
    cfg, manifest, _, model, checkpoint = load_saved_model(
        cfg, checkpoint_name, device
    )
    if int(checkpoint["global_step"]) != 10_000:
        raise RuntimeError("Loaded gMolAI checkpoint is not step 10,000")
    if sha256_file(checkpoint_path) != source["checkpoint"]["sha256"]:
        raise RuntimeError("gMolAI checkpoint hash changed during load")
    mean, scale, calibration_metadata, calibrator_sha256 = _load_embedding_calibrator(
        calibrator_path,
        expected=_calibrator_expected_identity(
            cfg, manifest, checkpoint_path, checkpoint
        ),
        dimensions=int(expected["dimension"]),
    )
    if calibrator_sha256 != source["calibrator"]["sha256"]:
        raise RuntimeError("gMolAI calibrator hash changed")

    molecules = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise RuntimeError("Panel contains an unparsable canonical SMILES")
        molecules.append(molecule)
    mean_array = mean.numpy()
    scale_array = scale.numpy()
    fixture = molecules[: min(2, len(molecules))]
    first = selected_embedding(model, fixture, cfg, device, mean_array, scale_array)
    second = selected_embedding(model, fixture, cfg, device, mean_array, scale_array)
    if not np.array_equal(first, second):
        raise RuntimeError(
            "gMolAI deterministic repeat failed; max delta="
            f"{float(np.max(np.abs(first - second)))}"
        )

    matrix = selected_embedding(model, molecules, cfg, device, mean_array, scale_array)
    expected_shape = (len(rows), int(expected["dimension"]))
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise RuntimeError(f"Unexpected gMolAI matrix {matrix.shape}/{matrix.dtype}")
    if not np.isfinite(matrix).all() or np.any(
        np.linalg.norm(matrix.astype(np.float64), axis=1) <= 1.0e-12
    ):
        raise RuntimeError("gMolAI produced non-finite or zero-norm vectors")

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
        "execution": "inference_only_frozen_encoder",
        "model": "gmolai",
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
        "fixed_batch_deterministic_repeat": True,
        "wall_seconds_model_load_warmup_and_export": elapsed,
        "rows_per_second_including_load_warmup_and_export": len(rows) / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_name": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "calibrator_sha256": calibrator_sha256,
        "calibration_graphs": int(calibration_metadata["graphs"]),
        "representation": expected["representation"],
        "python": platform.python_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(args.metadata, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
