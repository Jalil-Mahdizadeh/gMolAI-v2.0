#!/usr/bin/env python3
"""Measure one frozen encoder at all protocol batch sizes without serialization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Callable

import numpy as np

from benchmark_io import (
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    read_panel_tsv,
    require_hash,
    sha256_file,
    sha256_lines,
)


NEURAL_MODELS = {
    "gmolai",
    "molai",
    "molformer",
    "smi_ted",
    "molclr_gin",
    "kermt_v2",
}

# Set only inside the gMolAI container process so the common harness can drive
# the repository-canonical full-panel pipeline at each protocol batch size.
GMOLAI_ENCODER: Any | None = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--qualify-batch-size",
        type=int,
        help="Load the frozen native adapter, validate one batch, print provenance, and exit.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", required=True, type=int, nargs="+")
    return parser.parse_args()


def require_one_gpu() -> Any:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-GPU contract violated: visible GPUs={torch.cuda.device_count()}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return torch


def materialize_readonly_tensors(model: Any) -> None:
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.detach().clone(memory_format=torch.preserve_format)
        for buffer in model.buffers():
            buffer.data = buffer.detach().clone(memory_format=torch.preserve_format)


def load_gmolai(
    protocol: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    global GMOLAI_ENCODER

    import torch

    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from gmolai_retrain.config import apply_training_plan, load_config
    from gmolai_retrain.fast_inference import OptimizedSmilesEncoder
    from gmolai_retrain.representations import (
        _calibrator_expected_identity,
        _load_embedding_calibrator,
        load_saved_model,
    )

    sources = protocol["gmolai_sources"]
    config_path = REPOSITORY_ROOT / sources["config"]["path"]
    plan_path = REPOSITORY_ROOT / sources["training_plan"]["path"]
    run_dir = REPOSITORY_ROOT / sources["checkpoint"]["run_dir"]
    checkpoint_name = sources["checkpoint"]["name"]
    checkpoint_path = REPOSITORY_ROOT / sources["checkpoint"]["path"]
    calibrator_path = REPOSITORY_ROOT / sources["calibrator"]["path"]

    cfg = load_config(config_path)
    apply_training_plan(cfg, plan_path)
    cfg["paths"]["run_dir"] = str(run_dir)
    cfg["experiment_name"] = run_dir.name
    device = torch.device("cuda:0")
    cfg, manifest, _, model, checkpoint = load_saved_model(
        cfg, checkpoint_name, device
    )
    model.eval().requires_grad_(False)
    if int(checkpoint["global_step"]) != 10_000:
        raise RuntimeError("Loaded gMolAI checkpoint is not step 10,000")
    require_hash(checkpoint_path, sources["checkpoint"]["sha256"])
    mean, scale, calibration_metadata, calibrator_sha256 = _load_embedding_calibrator(
        calibrator_path,
        expected=_calibrator_expected_identity(
            cfg, manifest, checkpoint_path, checkpoint
        ),
        dimensions=int(protocol["models"]["gmolai"]["dimension"]),
    )
    if calibrator_sha256 != sources["calibrator"]["sha256"]:
        raise RuntimeError("gMolAI calibrator hash changed")
    workers = int(protocol["execution"]["cpu_workers"])
    encoder = OptimizedSmilesEncoder(
        model,
        mean,
        scale,
        device=device,
        batch_size=max(int(value) for value in protocol["execution"]["batch_sizes"]),
        node_budget=16384,
        workers=workers,
        mean_node_weight=3.0,
    )
    GMOLAI_ENCODER = encoder

    def encode(values: list[str]) -> np.ndarray:
        return encoder.encode(values)

    metadata = {
        "checkpoint_sha256": sources["checkpoint"]["sha256"],
        "calibrator_sha256": calibrator_sha256,
        "calibration_graphs": int(calibration_metadata["graphs"]),
        "checkpoint_step": int(checkpoint["global_step"]),
        "implementation": encoder.implementation,
        "cpu_workers": encoder.workers,
    }
    return 384, encode, metadata


def load_molai(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    sys.path.insert(0, "/opt/molai")
    from encoder import FrozenMolAI, prepare_smiles

    model = FrozenMolAI(device="cuda")

    def encode(values: list[str]) -> np.ndarray:
        prepared = prepare_smiles(values, allow_unknown_zero=False)
        return model.encode_prepared(prepared).numpy()

    return 512, encode, {"official_preprocessing": "canonical_nonisomeric_ClX_BrY"}


def load_molformer(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = "/opt/molformer/model"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_dir,
        deterministic_eval=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    materialize_readonly_tensors(model)
    model.to("cuda").requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        inputs = tokenizer(
            values,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(inputs["input_ids"].shape[1]) > 202:
            raise RuntimeError("MoLFormer received a pre-screen length violation")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs).pooler_output
        return output.detach().float().cpu().numpy()

    return 768, encode, {
        "model_revision": "7b12d946c181a37f6012b9dc3b002275de070314"
    }


def load_smi_ted(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    import torch

    sys.path.insert(0, "/opt/smi-ted/model")
    from load import load_smi_ted as load_model

    model = load_model(
        folder="/opt/smi-ted/model",
        ckpt_filename="smi-ted-Light_40.pt",
        vocab_filename="bert_vocab_curated.txt",
    )
    model.eval().requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        with torch.inference_mode():
            output = model.encode(
                values,
                batch_size=len(values),
                return_torch=True,
            )
        return output.detach().float().cpu().numpy()

    return 768, encode, {
        "model_revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "official_preprocessing": "canonical_nonisomeric",
    }


def load_molclr(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    sys.path.insert(0, "/opt/molclr")
    from encoder import MolCLRGinEncoder

    model = MolCLRGinEncoder(device="cuda")

    def encode(values: list[str]) -> np.ndarray:
        return model.encode(values, batch_size=len(values)).numpy()

    return 512, encode, {"representation": "preprojection_graph_vector"}


def load_kermt(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    from rdkit import Chem
    from task.extract_embeddings import (
        extract_embeddings_batch,
        load_encoder_from_checkpoint,
        load_projection_from_checkpoint,
    )

    checkpoint = "/opt/kermt/model/kermt_contrastive_v2.0.pt"
    encoder, readout, model_args = load_encoder_from_checkpoint(
        checkpoint, device="cuda"
    )
    if model_args.use_cuikmolmaker_featurization:
        raise RuntimeError("Released KERMT checkpoint requested unavailable cuik_molmaker")
    projection = load_projection_from_checkpoint(
        checkpoint, model_args, device="cuda"
    )
    encoder.eval().requires_grad_(False)
    projection.eval().requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        canonical = []
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"KERMT could not parse {value!r}")
            canonical.append(Chem.MolToSmiles(molecule))
        output, validity = extract_embeddings_batch(
            encoder=encoder,
            readout=readout,
            smiles_batch=canonical,
            args=model_args,
            device="cuda",
            projection_extractor=projection,
        )
        if not all(validity):
            raise RuntimeError("KERMT rejected a pre-screened canonical molecule")
        return np.asarray(output["projected"], dtype=np.float32)

    return 512, encode, {"representation": "cmim_projected_mean_latent"}


def load_morgan(
    _: dict[str, Any],
) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def encode(values: list[str]) -> np.ndarray:
        rows = []
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"Morgan could not parse {value!r}")
            rows.append(generator.GetFingerprintAsNumPy(molecule))
        return np.asarray(rows, dtype=np.float32)

    return 2048, encode, {"radius": 2, "fp_size": 2048}


LOADERS: dict[
    str,
    Callable[
        [dict[str, Any]],
        tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]],
    ],
] = {
    "gmolai": load_gmolai,
    "morgan": load_morgan,
    "molai": load_molai,
    "molformer": load_molformer,
    "smi_ted": load_smi_ted,
    "molclr_gin": load_molclr,
    "kermt_v2": load_kermt,
}


def validate_matrix(matrix: np.ndarray, model: str) -> None:
    for start in range(0, matrix.shape[0], 2048):
        chunk = matrix[start : start + 2048]
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"Non-finite {model} output at rows {start}:{start + len(chunk)}")
        if np.any(np.all(chunk == 0.0, axis=1)):
            raise RuntimeError(f"Zero-vector {model} output at rows {start}:{start + len(chunk)}")


def matrix_sha256(matrix: np.ndarray) -> str:
    if not matrix.flags.c_contiguous:
        raise RuntimeError("Cannot hash a non-contiguous embedding matrix")
    return hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()


def compare_matrices(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, float]:
    exact = True
    close = True
    maximum = 0.0
    for start in range(0, reference.shape[0], 2048):
        left = reference[start : start + 2048]
        right = candidate[start : start + 2048]
        if not np.array_equal(left, right):
            exact = False
        if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
            close = False
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    return exact, close, maximum


def compare_matrices_scale_aware(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    minimum_cosine_similarity: float,
    maximum_relative_l2_delta: float,
) -> dict[str, Any]:
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    difference = right - left
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    difference_norm = np.linalg.norm(difference, axis=1)
    relative = difference_norm / np.maximum(
        left_norm, np.finfo(np.float64).tiny
    )
    cosine = np.clip(
        np.sum(left * right, axis=1)
        / np.maximum(left_norm * right_norm, np.finfo(np.float64).tiny),
        -1.0,
        1.0,
    )
    comparison = {
        "exactly_equal_to_reference": bool(np.array_equal(reference, candidate)),
        "maximum_absolute_delta_from_reference": float(np.max(np.abs(difference))),
        "root_mean_square_delta_from_reference": float(
            np.sqrt(np.mean(difference * difference))
        ),
        "relative_l2_delta_p50_from_reference": float(np.quantile(relative, 0.50)),
        "relative_l2_delta_p95_from_reference": float(np.quantile(relative, 0.95)),
        "relative_l2_delta_p99_from_reference": float(np.quantile(relative, 0.99)),
        "maximum_relative_l2_delta_from_reference": float(np.max(relative)),
        "minimum_cosine_similarity_to_reference": float(np.min(cosine)),
        "cosine_similarity_p01_to_reference": float(np.quantile(cosine, 0.01)),
        "cosine_similarity_p50_to_reference": float(np.quantile(cosine, 0.50)),
    }
    comparison["within_tolerance_of_reference"] = bool(
        comparison["minimum_cosine_similarity_to_reference"]
        >= minimum_cosine_similarity
        and comparison["maximum_relative_l2_delta_from_reference"]
        <= maximum_relative_l2_delta
    )
    return comparison


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    if args.model not in LOADERS or args.model not in protocol["models"]:
        raise ValueError(f"Unsupported model: {args.model}")
    frozen_batches = [int(value) for value in protocol["execution"]["batch_sizes"]]
    if args.batch_sizes != frozen_batches:
        raise RuntimeError(
            f"Batch sizes differ from frozen protocol: {args.batch_sizes} != {frozen_batches}"
        )
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result: {args.output}")
    panel = protocol["panel"]
    require_hash(args.input, panel["tsv_sha256"])
    rows = read_panel_tsv(args.input)
    if len(rows) != int(panel["rows"]):
        raise RuntimeError("Speed panel row count changed")
    identity_digest = sha256_lines(row["molecule_hash"] for row in rows)
    if identity_digest != panel["ordered_identity_sha256"]:
        raise RuntimeError("Speed panel ordered identity digest changed")
    smiles = [row["canonical_smiles"] for row in rows]

    torch = require_one_gpu() if args.model in NEURAL_MODELS else None
    load_started = time.perf_counter()
    dimension, encode_batch, implementation = LOADERS[args.model](protocol)
    model_load_seconds = time.perf_counter() - load_started
    expected_dimension = int(protocol["models"][args.model]["dimension"])
    if dimension != expected_dimension:
        raise RuntimeError(
            f"Adapter dimension differs from protocol: {dimension} != {expected_dimension}"
        )

    if args.qualify_batch_size is not None:
        qualification_size = int(args.qualify_batch_size)
        if qualification_size not in frozen_batches:
            raise RuntimeError(
                f"Qualification size {qualification_size} is not in {frozen_batches}"
            )
        if GMOLAI_ENCODER is not None:
            GMOLAI_ENCODER.batch_size = qualification_size
            GMOLAI_ENCODER.warm_workers(smiles[:qualification_size])
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        qualification_started = time.perf_counter()
        matrix = np.asarray(
            encode_batch(smiles[:qualification_size]), dtype=np.float32
        )
        if torch is not None:
            torch.cuda.synchronize()
        qualification_seconds = time.perf_counter() - qualification_started
        if matrix.shape != (qualification_size, dimension):
            raise RuntimeError(
                f"Qualification shape mismatch for {args.model}: {matrix.shape}"
            )
        validate_matrix(matrix, args.model)
        report = {
            "status": "ok",
            "model": args.model,
            "batch_size": qualification_size,
            "dimension": dimension,
            "seconds": qualification_seconds,
            "rows_per_second": qualification_size / qualification_seconds,
            "model_load_seconds": model_load_seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch is not None else None
            ),
            "implementation": implementation,
        }
        if GMOLAI_ENCODER is not None:
            GMOLAI_ENCODER.close()
        print(json.dumps(report, sort_keys=True))
        return

    worker_startup_seconds = 0.0
    if GMOLAI_ENCODER is not None:
        worker_started = time.perf_counter()
        GMOLAI_ENCODER.warm_workers(smiles[: max(frozen_batches)])
        worker_startup_seconds = time.perf_counter() - worker_started

    fixture_values = smiles[:8]
    first = np.asarray(encode_batch(fixture_values), dtype=np.float32)
    second = np.asarray(encode_batch(fixture_values), dtype=np.float32)
    if first.shape != (8, dimension) or second.shape != (8, dimension):
        raise RuntimeError(
            f"Fixed-batch repeatability shape mismatch: {first.shape}, {second.shape}"
        )
    fixed_batch_repeatability = compare_matrices_scale_aware(
        first,
        second,
        minimum_cosine_similarity=float(
            protocol["execution"]["fixed_batch_minimum_cosine_similarity"]
        ),
        maximum_relative_l2_delta=float(
            protocol["execution"]["fixed_batch_maximum_relative_l2_delta"]
        ),
    )
    if not fixed_batch_repeatability["within_tolerance_of_reference"]:
        raise RuntimeError(
            f"Fixed-batch repeatability qualification failed for {args.model}: "
            f"minimum_cosine="
            f"{fixed_batch_repeatability['minimum_cosine_similarity_to_reference']}, "
            f"maximum_relative_l2="
            f"{fixed_batch_repeatability['maximum_relative_l2_delta_from_reference']}"
        )
    validate_matrix(first, args.model)
    validate_matrix(second, args.model)

    minimum_cosine = float(
        protocol["execution"]["cross_batch_minimum_cosine_similarity"]
    )
    maximum_relative_l2 = float(
        protocol["execution"]["cross_batch_maximum_relative_l2_delta"]
    )
    record_only_nonconformance_models = {
        str(value)
        for value in protocol["execution"][
            "record_only_cross_batch_nonconformance_models"
        ]
    }
    reference: np.ndarray | None = None
    conditions: list[dict[str, Any]] = []
    total_warmup_seconds = 0.0

    for batch_size in frozen_batches:
        if GMOLAI_ENCODER is not None:
            GMOLAI_ENCODER.batch_size = batch_size
        warmup_values = smiles[:batch_size]
        warmup_started = time.perf_counter()
        warmup = np.asarray(encode_batch(warmup_values), dtype=np.float32)
        if torch is not None:
            torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started
        total_warmup_seconds += warmup_seconds
        if warmup.shape != (batch_size, dimension):
            raise RuntimeError(f"Warm-up shape mismatch at batch size {batch_size}")
        validate_matrix(warmup, args.model)
        del warmup

        optimized_full_panel = GMOLAI_ENCODER is not None
        batch_latencies: list[float] = []
        if torch is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_gpu_memory = int(torch.cuda.memory_allocated())
        else:
            baseline_gpu_memory = None

        started = time.perf_counter()
        if optimized_full_panel:
            matrix = np.asarray(encode_batch(smiles), dtype=np.float32)
            if matrix.shape != (len(smiles), dimension):
                raise RuntimeError(
                    f"Unexpected {args.model} full-panel output: {matrix.shape}"
                )
        else:
            matrix = np.empty((len(smiles), dimension), dtype=np.float32)
            for start in range(0, len(smiles), batch_size):
                stop = min(start + batch_size, len(smiles))
                batch_started = time.perf_counter()
                batch = np.asarray(encode_batch(smiles[start:stop]), dtype=np.float32)
                if batch.shape != (stop - start, dimension):
                    raise RuntimeError(
                        f"Unexpected {args.model} output at {start}:{stop}: {batch.shape}"
                    )
                matrix[start:stop] = batch
                batch_latencies.append(time.perf_counter() - batch_started)
        if torch is not None:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        validate_matrix(matrix, args.model)
        digest = matrix_sha256(matrix)
        if reference is None:
            reference = matrix
        comparison = compare_matrices_scale_aware(
            reference,
            matrix,
            minimum_cosine_similarity=minimum_cosine,
            maximum_relative_l2_delta=maximum_relative_l2,
        )
        if (
            not comparison["within_tolerance_of_reference"]
            and args.model not in record_only_nonconformance_models
        ):
            raise RuntimeError(
                f"Batch-size output instability for {args.model} at {batch_size}: "
                f"minimum_cosine={comparison['minimum_cosine_similarity_to_reference']}, "
                f"maximum_relative_l2="
                f"{comparison['maximum_relative_l2_delta_from_reference']}"
            )
        if not comparison["within_tolerance_of_reference"]:
            print(
                json.dumps(
                    {
                        "warning": "recorded_cross_batch_output_nonconformance",
                        "model": args.model,
                        "batch_size": batch_size,
                        "minimum_cosine_similarity_to_reference": comparison[
                            "minimum_cosine_similarity_to_reference"
                        ],
                        "maximum_relative_l2_delta_from_reference": comparison[
                            "maximum_relative_l2_delta_from_reference"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        conditions.append(
            {
                "batch_size": batch_size,
                "measured_passes": 1,
                "warmup_batches": 1,
                "warmup_seconds": warmup_seconds,
                "wall_seconds": elapsed,
                "rows_per_second": len(smiles) / elapsed,
                "milliseconds_per_molecule": elapsed * 1000.0 / len(smiles),
                "batch_count": len(batch_latencies),
                "full_batch_count": len(smiles) // batch_size,
                "last_batch_rows": len(smiles) % batch_size or batch_size,
                "batch_latency_seconds": batch_latencies,
                "batch_latency_scope": (
                    "not_instrumented_to_avoid_perturbing_parallel_pipeline"
                    if optimized_full_panel
                    else "native_batch_calls"
                ),
                "batch_latency_p50_seconds": (
                    None if optimized_full_panel else quantile(batch_latencies, 0.50)
                ),
                "batch_latency_p95_seconds": (
                    None if optimized_full_panel else quantile(batch_latencies, 0.95)
                ),
                "batch_latency_p99_seconds": (
                    None if optimized_full_panel else quantile(batch_latencies, 0.99)
                ),
                "matrix_sha256": digest,
                **comparison,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch is not None else None
                ),
                "baseline_gpu_memory_bytes": baseline_gpu_memory,
            }
        )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "batch_size": batch_size,
                    "rows": len(smiles),
                    "wall_seconds": elapsed,
                    "rows_per_second": len(smiles) / elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if batch_size != frozen_batches[0]:
            del matrix

    del reference
    model_specification = protocol["models"][args.model]
    report = {
        "schema_version": 1,
        "status": "ok",
        "execution": "inference_only_single_pass_speed_benchmark",
        "training_performed": False,
        "model_weights_modified": False,
        "scientific_embedding_artifact_written": False,
        "model": args.model,
        "display_name": model_specification["display_name"],
        "device_class": model_specification["device"],
        "representation": model_specification["representation"],
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "ordered_identity_sha256": identity_digest,
        "dimension": dimension,
        "dtype": "float32",
        "timing_boundary": protocol["timing"]["primary"],
        "condition_order": frozen_batches,
        "model_load_seconds_excluded_from_primary": model_load_seconds,
        "worker_startup_seconds_excluded_from_primary": worker_startup_seconds,
        "warmup_seconds_excluded_from_primary": total_warmup_seconds,
        "fixed_batch_deterministic_qualification": bool(
            fixed_batch_repeatability["exactly_equal_to_reference"]
        ),
        "cross_batch_integrity_passed": bool(
            all(
                condition["within_tolerance_of_reference"]
                for condition in conditions
            )
        ),
        "cross_batch_integrity_policy": (
            "record_only_known_native_nonconformance"
            if args.model in record_only_nonconformance_models
            else "fail_closed"
        ),
        "fixed_batch_repeatability_qualification": {
            "passed": bool(
                fixed_batch_repeatability["within_tolerance_of_reference"]
            ),
            "exactly_equal": bool(
                fixed_batch_repeatability["exactly_equal_to_reference"]
            ),
            "maximum_absolute_delta": fixed_batch_repeatability[
                "maximum_absolute_delta_from_reference"
            ],
            "root_mean_square_delta": fixed_batch_repeatability[
                "root_mean_square_delta_from_reference"
            ],
            "maximum_relative_l2_delta": fixed_batch_repeatability[
                "maximum_relative_l2_delta_from_reference"
            ],
            "minimum_cosine_similarity": fixed_batch_repeatability[
                "minimum_cosine_similarity_to_reference"
            ],
        },
        "conditions": conditions,
        "implementation": implementation,
        "host": platform.node(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__ if torch is not None else None,
        "cuda_runtime": torch.version.cuda if torch is not None else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch is not None else None,
        "visible_gpu_count": torch.cuda.device_count() if torch is not None else 0,
        "maximum_resident_set_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "container": model_specification["container"],
        "container_sha256": model_specification["container_sha256"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if GMOLAI_ENCODER is not None:
        GMOLAI_ENCODER.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "conditions"}, sort_keys=True))


if __name__ == "__main__":
    main()
