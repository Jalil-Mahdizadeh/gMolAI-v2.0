#!/usr/bin/env python3
"""Run the released single-GPU stochastic decoder throughput benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
import torch

from common import (
    atomic_write_json,
    load_json,
    ratio_bootstrap,
    sha256_file,
    utc_now,
)


RDLogger.DisableLog("rdApp.*")


RAW_SCHEMA = pa.schema(
    [
        ("batch_index", pa.int16()),
        ("benchmark_seed_index", pa.int16()),
        ("source_panel_row", pa.int32()),
        ("query_position", pa.int32()),
        ("target_index", pa.int64()),
        ("target_hash", pa.string()),
        ("draw_index", pa.int16()),
        ("sampling_seed", pa.int64()),
        ("raw_smiles", pa.string()),
        ("token_error", pa.string()),
        ("decoder_log_probability", pa.float32()),
        ("generated_length", pa.int16()),
        ("rdkit_valid", pa.bool_()),
        ("rdkit_canonical_smiles", pa.string()),
        ("rdkit_identity_hash", pa.string()),
        ("is_first_rdkit_unique_within_seed", pa.bool_()),
        ("release_policy_accepted", pa.bool_()),
        ("release_canonical_smiles", pa.string()),
        ("release_molecule_hash", pa.string()),
        ("release_rejection_reason", pa.string()),
        ("is_first_policy_unique_within_seed", pa.bool_()),
    ]
)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def release_modules(repo_root: Path) -> tuple[Any, Any, Any]:
    inference_dir = repo_root / "inference"
    source_dir = repo_root / "src"
    for path in (source_dir, inference_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import gmolai as release
    from _decoder import decode_tokens, generate_seeded_sample_pool

    return release, decode_tokens, generate_seeded_sample_pool


def metric_row(
    metric: str,
    value: float,
    unit: str,
    definition: str,
    *,
    numerator: float | None = None,
    denominator_seconds: float | None = None,
    ci: tuple[float, float] | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "ci95_lower": None if ci is None else float(ci[0]),
        "ci95_upper": None if ci is None else float(ci[1]),
        "numerator": numerator,
        "denominator_seconds": denominator_seconds,
        "definition": definition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    config_path = step_root / "config" / "benchmark.json"
    config = load_json(config_path)
    selected_panel_path = step_root / "inputs" / "selected_panel.csv"
    selected_conditions_path = step_root / "inputs" / "selected_conditions.npy"
    selection_metadata_path = step_root / "inputs" / "selection_metadata.json"
    output_dir = step_root / "outputs"
    raw_path = output_dir / "raw" / "proposals.parquet"
    seed_metrics_path = output_dir / "tables" / "per_seed_metrics.csv"
    batch_metrics_path = output_dir / "tables" / "per_batch_timings.csv"
    summary_table_path = output_dir / "tables" / "benchmark_summary.csv"
    summary_path = output_dir / "benchmark_summary.json"
    targets = (
        raw_path,
        seed_metrics_path,
        batch_metrics_path,
        summary_table_path,
        summary_path,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Benchmark outputs already exist: {existing}")
    for path in (selected_panel_path, selected_conditions_path, selection_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared input is missing: {path}")

    release, decode_tokens, generate_seeded_sample_pool = release_modules(repo_root)
    threads = int(config["decoder"]["torch_threads"])
    release.configure_decode_runtime(threads)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Benchmark requires exactly one visible CUDA GPU; observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    panel = pd.read_csv(selected_panel_path)
    conditions = np.load(selected_conditions_path, allow_pickle=False)
    expected_seeds = int(config["selection"]["count"])
    draws = int(config["decoder"]["draws_per_seed"])
    batch_size = int(config["decoder"]["query_batch_size"])
    if len(panel) != expected_seeds or conditions.shape != (expected_seeds, 384):
        raise ValueError("Prepared panel/condition shape differs from frozen protocol")
    if conditions.dtype != np.float32 or not np.isfinite(conditions).all():
        raise ValueError("Prepared condition matrix is not finite float32")
    if draws != 1000 or expected_seeds != 100:
        raise ValueError("Frozen benchmark must remain 100 seeds x 1,000 draws")
    if batch_size <= 0 or expected_seeds % batch_size:
        raise ValueError("Query batch size must divide the selected seed count")

    models_dir = repo_root / config["source"]["models_dir"]
    run_started = time.perf_counter()
    artifact_started = time.perf_counter()
    artifact_hashes = release.validate_artifact_hashes(models_dir)
    artifact_validation_seconds = time.perf_counter() - artifact_started
    load_started = time.perf_counter()
    model, decoder_payload, _ = release.load_decoder(
        models_dir, device, artifact_hashes
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    resolved_config = release.encoder_api.load_json_object(
        models_dir / "resolved_config.json"
    )
    maximum_steps = int(config["decoder"]["maximum_smiles_bytes"])
    temperature = float(config["decoder"]["temperature"])
    top_p = float(config["decoder"]["top_p"])
    sampling_seeds = [
        int(release.sample_seed(str(value))) for value in panel["target_hash"]
    ]

    warm_count = int(config["warmup"]["seed_count"])
    warm_draws = int(config["warmup"]["draws_per_seed"])
    warm_condition = torch.as_tensor(
        np.ascontiguousarray(conditions[:warm_count]),
        dtype=torch.float32,
        device=device,
    )
    warm_seeds = [
        int(release.stable_digest("step03-warmup", value)[:16], 16) % (2**63 - 1)
        for value in panel["target_hash"].iloc[:warm_count]
    ]
    torch.cuda.synchronize()
    warm_started = time.perf_counter()
    warm_tokens, warm_scores, warm_lengths = generate_seeded_sample_pool(
        model,
        warm_condition,
        maximum_steps=maximum_steps,
        draws=warm_draws,
        temperature=temperature,
        top_p=top_p,
        seeds=warm_seeds,
    )
    _ = warm_tokens.cpu(), warm_scores.cpu(), warm_lengths.cpu()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_started
    del warm_condition, warm_tokens, warm_scores, warm_lengths
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temp_raw = raw_path.with_name(f".{raw_path.name}.incomplete")
    temp_raw.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        temp_raw, RAW_SCHEMA, compression="zstd", use_dictionary=True
    )
    batch_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    global_rdkit_hashes: set[str] = set()
    global_policy_hashes: set[str] = set()
    token_error_counts: Counter[str] = Counter()
    policy_rejection_counts: Counter[str] = Counter()
    measured_wall_started = time.perf_counter()
    try:
        for offset in range(0, expected_seeds, batch_size):
            stop = offset + batch_size
            batch_index = offset // batch_size
            batch_panel = panel.iloc[offset:stop]
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            batch_conditions = torch.as_tensor(
                np.ascontiguousarray(conditions[offset:stop]),
                dtype=torch.float32,
                device=device,
            )
            token_tensor, score_tensor, length_tensor = generate_seeded_sample_pool(
                model,
                batch_conditions,
                maximum_steps=maximum_steps,
                draws=draws,
                temperature=temperature,
                top_p=top_p,
                seeds=sampling_seeds[offset:stop],
            )
            token_array = token_tensor.cpu().numpy()
            score_array = score_tensor.cpu().numpy()
            length_array = length_tensor.cpu().numpy()
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - generation_started
            del batch_conditions, token_tensor, score_tensor, length_tensor

            decode_started = time.perf_counter()
            decoded: list[list[dict[str, Any]]] = []
            for local in range(batch_size):
                current: list[dict[str, Any]] = []
                for draw_index in range(draws):
                    raw_smiles, token_error = decode_tokens(
                        token_array[local, draw_index]
                    )
                    current.append(
                        {
                            "raw_smiles": raw_smiles,
                            "token_error": token_error,
                            "decoder_log_probability": float(
                                score_array[local, draw_index]
                            ),
                            "generated_length": int(
                                length_array[local, draw_index]
                            ),
                        }
                    )
                decoded.append(current)
            token_decode_seconds = time.perf_counter() - decode_started

            rdkit_started = time.perf_counter()
            for local in range(batch_size):
                seen: set[str] = set()
                for record in decoded[local]:
                    record["rdkit_valid"] = False
                    record["rdkit_canonical_smiles"] = ""
                    record["rdkit_identity_hash"] = ""
                    record["is_first_rdkit_unique_within_seed"] = False
                    if record["token_error"]:
                        token_error_counts[str(record["token_error"])] += 1
                        continue
                    molecule = Chem.MolFromSmiles(str(record["raw_smiles"]))
                    if molecule is None:
                        continue
                    canonical = Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    )
                    identity_hash = hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest()
                    first = identity_hash not in seen
                    seen.add(identity_hash)
                    global_rdkit_hashes.add(identity_hash)
                    record["rdkit_valid"] = True
                    record["rdkit_canonical_smiles"] = canonical
                    record["rdkit_identity_hash"] = identity_hash
                    record["is_first_rdkit_unique_within_seed"] = first
            rdkit_validation_seconds = time.perf_counter() - rdkit_started

            policy_started = time.perf_counter()
            for local in range(batch_size):
                seen = set()
                for record in decoded[local]:
                    record["release_policy_accepted"] = False
                    record["release_canonical_smiles"] = ""
                    record["release_molecule_hash"] = ""
                    record["release_rejection_reason"] = ""
                    record["is_first_policy_unique_within_seed"] = False
                    if record["token_error"]:
                        record["release_rejection_reason"] = str(
                            record["token_error"]
                        )
                        continue
                    canonical, reason = release.encoder_api.canonicalize_input(
                        str(record["raw_smiles"]), resolved_config
                    )
                    if reason is not None:
                        record["release_rejection_reason"] = str(reason)
                        policy_rejection_counts[str(reason)] += 1
                        continue
                    assert canonical is not None
                    molecule_hash = str(canonical.molecule_hash)
                    first = molecule_hash not in seen
                    seen.add(molecule_hash)
                    global_policy_hashes.add(molecule_hash)
                    record["release_policy_accepted"] = True
                    record["release_canonical_smiles"] = str(canonical.smiles)
                    record["release_molecule_hash"] = molecule_hash
                    record["is_first_policy_unique_within_seed"] = first
            release_policy_seconds = time.perf_counter() - policy_started

            raw_records: list[dict[str, Any]] = []
            batch_seed_records: list[dict[str, Any]] = []
            for local, row in enumerate(batch_panel.itertuples(index=False)):
                records = decoded[local]
                token_decodable = sum(not item["token_error"] for item in records)
                rdkit_valid = sum(bool(item["rdkit_valid"]) for item in records)
                rdkit_unique = sum(
                    bool(item["is_first_rdkit_unique_within_seed"])
                    for item in records
                )
                policy_accepted = sum(
                    bool(item["release_policy_accepted"]) for item in records
                )
                policy_unique = sum(
                    bool(item["is_first_policy_unique_within_seed"])
                    for item in records
                )
                current_seed = {
                    "batch_index": batch_index,
                    "benchmark_seed_index": int(row.benchmark_seed_index),
                    "source_panel_row": int(row.source_panel_row),
                    "query_position": int(row.query_position),
                    "target_index": int(row.target_index),
                    "target_hash": str(row.target_hash),
                    "seed_canonical_smiles": str(row.seed_canonical_smiles),
                    "seed_heavy_atoms": int(row.seed_heavy_atoms),
                    "sampling_seed": int(sampling_seeds[offset + local]),
                    "raw_proposals": draws,
                    "token_decodable_proposals": token_decodable,
                    "token_decodable_fraction": token_decodable / draws,
                    "rdkit_valid_proposals": rdkit_valid,
                    "rdkit_valid_fraction": rdkit_valid / draws,
                    "rdkit_unique_valid_molecules": rdkit_unique,
                    "rdkit_unique_fraction_of_raw": rdkit_unique / draws,
                    "rdkit_unique_fraction_of_valid": (
                        rdkit_unique / rdkit_valid if rdkit_valid else 0.0
                    ),
                    "release_policy_accepted_proposals": policy_accepted,
                    "release_policy_accepted_fraction": policy_accepted / draws,
                    "release_policy_unique_molecules": policy_unique,
                    "release_policy_unique_fraction_of_raw": policy_unique / draws,
                }
                seed_rows.append(current_seed)
                batch_seed_records.append(current_seed)
                for draw_index, record in enumerate(records, start=1):
                    raw_records.append(
                        {
                            "batch_index": batch_index,
                            "benchmark_seed_index": int(row.benchmark_seed_index),
                            "source_panel_row": int(row.source_panel_row),
                            "query_position": int(row.query_position),
                            "target_index": int(row.target_index),
                            "target_hash": str(row.target_hash),
                            "draw_index": draw_index,
                            "sampling_seed": int(sampling_seeds[offset + local]),
                            **record,
                        }
                    )

            serialization_started = time.perf_counter()
            writer.write_table(pa.Table.from_pylist(raw_records, schema=RAW_SCHEMA))
            serialization_seconds = time.perf_counter() - serialization_started
            raw_count = batch_size * draws
            valid_unique_count = sum(
                int(item["rdkit_unique_valid_molecules"])
                for item in batch_seed_records
            )
            policy_unique_count = sum(
                int(item["release_policy_unique_molecules"])
                for item in batch_seed_records
            )
            raw_smiles_seconds = generation_seconds + token_decode_seconds
            valid_unique_seconds = raw_smiles_seconds + rdkit_validation_seconds
            policy_unique_seconds = valid_unique_seconds + release_policy_seconds
            batch_rows.append(
                {
                    "batch_index": batch_index,
                    "first_benchmark_seed_index": offset,
                    "seed_count": batch_size,
                    "raw_proposals": raw_count,
                    "token_decodable_proposals": sum(
                        int(item["token_decodable_proposals"])
                        for item in batch_seed_records
                    ),
                    "rdkit_valid_proposals": sum(
                        int(item["rdkit_valid_proposals"])
                        for item in batch_seed_records
                    ),
                    "rdkit_unique_valid_molecules": valid_unique_count,
                    "release_policy_accepted_proposals": sum(
                        int(item["release_policy_accepted_proposals"])
                        for item in batch_seed_records
                    ),
                    "release_policy_unique_molecules": policy_unique_count,
                    "generation_seconds": generation_seconds,
                    "token_decode_seconds": token_decode_seconds,
                    "rdkit_validation_seconds": rdkit_validation_seconds,
                    "release_policy_seconds": release_policy_seconds,
                    "serialization_seconds": serialization_seconds,
                    "raw_proposals_per_second": raw_count / generation_seconds,
                    "raw_smiles_per_second": raw_count / raw_smiles_seconds,
                    "valid_unique_molecules_per_second": (
                        valid_unique_count / valid_unique_seconds
                    ),
                    "policy_unique_molecules_per_second": (
                        policy_unique_count / policy_unique_seconds
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "of": expected_seeds // batch_size,
                        "raw_proposals_per_second": raw_count / generation_seconds,
                        "valid_unique_molecules": valid_unique_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        writer.close()
        writer = None
        os.replace(temp_raw, raw_path)
    except BaseException:
        if writer is not None:
            writer.close()
        temp_raw.unlink(missing_ok=True)
        raise
    measured_wall_seconds = time.perf_counter() - measured_wall_started

    batch_frame = pd.DataFrame(batch_rows)
    seed_frame = pd.DataFrame(seed_rows)
    total_raw = int(batch_frame["raw_proposals"].sum())
    total_token_decodable = int(batch_frame["token_decodable_proposals"].sum())
    total_rdkit_valid = int(batch_frame["rdkit_valid_proposals"].sum())
    total_rdkit_unique = int(batch_frame["rdkit_unique_valid_molecules"].sum())
    total_policy_accepted = int(
        batch_frame["release_policy_accepted_proposals"].sum()
    )
    total_policy_unique = int(
        batch_frame["release_policy_unique_molecules"].sum()
    )
    generation_seconds = float(batch_frame["generation_seconds"].sum())
    token_decode_seconds = float(batch_frame["token_decode_seconds"].sum())
    rdkit_validation_seconds = float(
        batch_frame["rdkit_validation_seconds"].sum()
    )
    release_policy_seconds = float(batch_frame["release_policy_seconds"].sum())
    serialization_seconds = float(batch_frame["serialization_seconds"].sum())
    raw_smiles_seconds = generation_seconds + token_decode_seconds
    valid_unique_seconds = raw_smiles_seconds + rdkit_validation_seconds
    policy_unique_seconds = valid_unique_seconds + release_policy_seconds
    uncertainty = config["uncertainty"]
    bootstrap_kwargs = {
        "replicates": int(uncertainty["bootstrap_replicates"]),
        "confidence_level": float(uncertainty["confidence_level"]),
    }
    raw_ci = ratio_bootstrap(
        batch_frame["raw_proposals"],
        batch_frame["generation_seconds"],
        seed=int(uncertainty["bootstrap_seed"]),
        **bootstrap_kwargs,
    )
    raw_smiles_ci = ratio_bootstrap(
        batch_frame["raw_proposals"],
        batch_frame["generation_seconds"] + batch_frame["token_decode_seconds"],
        seed=int(uncertainty["bootstrap_seed"]) + 1,
        **bootstrap_kwargs,
    )
    valid_unique_ci = ratio_bootstrap(
        batch_frame["rdkit_unique_valid_molecules"],
        batch_frame["generation_seconds"]
        + batch_frame["token_decode_seconds"]
        + batch_frame["rdkit_validation_seconds"],
        seed=int(uncertainty["bootstrap_seed"]) + 2,
        **bootstrap_kwargs,
    )
    policy_unique_ci = ratio_bootstrap(
        batch_frame["release_policy_unique_molecules"],
        batch_frame["generation_seconds"]
        + batch_frame["token_decode_seconds"]
        + batch_frame["rdkit_validation_seconds"]
        + batch_frame["release_policy_seconds"],
        seed=int(uncertainty["bootstrap_seed"]) + 3,
        **bootstrap_kwargs,
    )
    metrics = [
        metric_row(
            "raw_proposals_per_second",
            total_raw / generation_seconds,
            "proposals/s",
            "Raw proposal slots / condition transfer + GPU generation + token transfer",
            numerator=total_raw,
            denominator_seconds=generation_seconds,
            ci=raw_ci,
        ),
        metric_row(
            "raw_smiles_per_second",
            total_raw / raw_smiles_seconds,
            "SMILES/s",
            "Raw proposal slots / generation + byte-token-to-SMILES conversion",
            numerator=total_raw,
            denominator_seconds=raw_smiles_seconds,
            ci=raw_smiles_ci,
        ),
        metric_row(
            "valid_unique_molecules_per_second",
            total_rdkit_unique / valid_unique_seconds,
            "molecules/s",
            "Sum of per-seed first unique RDKit-valid identities / generation + token decode + RDKit validation",
            numerator=total_rdkit_unique,
            denominator_seconds=valid_unique_seconds,
            ci=valid_unique_ci,
        ),
        metric_row(
            "policy_unique_molecules_per_second",
            total_policy_unique / policy_unique_seconds,
            "molecules/s",
            "Sum of per-seed first unique release-policy identities / generation + token decode + RDKit validation + policy pass",
            numerator=total_policy_unique,
            denominator_seconds=policy_unique_seconds,
            ci=policy_unique_ci,
        ),
        metric_row(
            "token_decodable_fraction",
            total_token_decodable / total_raw,
            "fraction",
            "Token sequences decoded to an ASCII string with EOS / raw slots",
            numerator=total_token_decodable,
        ),
        metric_row(
            "rdkit_valid_fraction",
            total_rdkit_valid / total_raw,
            "fraction",
            "RDKit-parseable proposals / raw slots",
            numerator=total_rdkit_valid,
        ),
        metric_row(
            "rdkit_unique_fraction_of_raw",
            total_rdkit_unique / total_raw,
            "fraction",
            "Sum of per-seed unique RDKit-valid identities / raw slots",
            numerator=total_rdkit_unique,
        ),
        metric_row(
            "release_policy_accepted_fraction",
            total_policy_accepted / total_raw,
            "fraction",
            "Released-policy accepted proposals / raw slots",
            numerator=total_policy_accepted,
        ),
    ]
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "completed_utc": utc_now(),
        "status": "complete",
        "protocol": {
            "embedding_space": config["embedding_space"],
            "selected_seeds": expected_seeds,
            "draws_per_seed": draws,
            "total_raw_proposals": total_raw,
            "query_batch_size": batch_size,
            "measured_batches": int(len(batch_frame)),
            "temperature": temperature,
            "top_p": top_p,
            "maximum_smiles_bytes": maximum_steps,
            "sampling_seed_rule": "released gmolai.sample_seed(target_hash)",
            "warmup_seed_count": warm_count,
            "warmup_draws_per_seed": warm_draws,
        },
        "counts": {
            "raw_proposals": total_raw,
            "token_decodable_proposals": total_token_decodable,
            "rdkit_valid_proposals": total_rdkit_valid,
            "per_seed_unique_rdkit_valid_molecules": total_rdkit_unique,
            "globally_unique_rdkit_valid_molecules": len(global_rdkit_hashes),
            "release_policy_accepted_proposals": total_policy_accepted,
            "per_seed_unique_release_policy_molecules": total_policy_unique,
            "globally_unique_release_policy_molecules": len(global_policy_hashes),
        },
        "timings_seconds": {
            "artifact_validation": artifact_validation_seconds,
            "model_load": model_load_seconds,
            "warmup_excluded": warmup_seconds,
            "generation": generation_seconds,
            "token_decode": token_decode_seconds,
            "rdkit_validation": rdkit_validation_seconds,
            "release_policy": release_policy_seconds,
            "serialization_excluded": serialization_seconds,
            "measured_observed_wall_including_serialization": measured_wall_seconds,
            "whole_script_to_summary": time.perf_counter() - run_started,
        },
        "metrics": {item["metric"]: item for item in metrics},
        "uncertainty": config["uncertainty"],
        "execution": {
            "cuda_visible_device_count": torch.cuda.device_count(),
            "cuda_visible_devices_environment": os.environ.get(
                "CUDA_VISIBLE_DEVICES", ""
            ),
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
            "torch_threads": torch.get_num_threads(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "rdkit": rdBase.rdkitVersion,
            "container_image": os.environ.get("GMOLAI_CONTAINER_IMAGE", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "selected_panel_sha256": sha256_file(selected_panel_path),
            "selected_conditions_sha256": sha256_file(selected_conditions_path),
            "selection_metadata_sha256": sha256_file(selection_metadata_path),
            "release_artifact_sha256": artifact_hashes,
            "decoder_source_checkpoint_sha256": decoder_payload.get(
                "source_training_checkpoint_sha256"
            ),
            "raw_proposals_sha256": sha256_file(raw_path),
        },
        "diagnostics": {
            "token_errors": dict(sorted(token_error_counts.items())),
            "release_policy_rejections": dict(
                sorted(policy_rejection_counts.items())
            ),
        },
    }
    atomic_write_csv(seed_metrics_path, seed_frame)
    atomic_write_csv(batch_metrics_path, batch_frame)
    atomic_write_csv(summary_table_path, pd.DataFrame(metrics))
    atomic_write_json(summary_path, summary)
    print(json.dumps({"status": "complete", "metrics": summary["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
