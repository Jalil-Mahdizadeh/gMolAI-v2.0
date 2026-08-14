#!/usr/bin/env python3
"""Run the scaled MMP and five-space latent-control selection study."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import rdkit
import torch

from analysis_core import (
    METHOD_ORDER,
    RETRIEVAL_METRICS,
    SPACE_ORDER,
    add_observation_ids,
    assign_mismatched_transforms,
    average_replicates,
    by_transformation,
    evaluate_alignment,
    fit_directions,
    hierarchical_bootstrap,
    retrieval_experiment,
    retrieval_summary,
    select_queries,
    space_matrix,
)
from reporting import (
    build_figures,
    select_control_space,
    transfer_assessment,
    write_reports,
)
from scaled_common import (
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    core_sets,
    covariance_eigendecomposition,
    ensure_within,
    hash_ledger,
    load_validate_manifest,
    make_fingerprints,
    payload_array,
    requested_target_sets,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def validate_stage_seal(step_root: Path, name: str) -> dict[str, Any]:
    path = step_root / "state" / name
    if not path.is_file():
        raise RuntimeError(f"Required stage seal is missing: {name}")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal.get("status") != "complete":
        raise RuntimeError(f"Required stage is not complete: {name}")
    for record in seal.get("outputs", {}).values():
        artifact = step_root / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"Sealed artifact changed: {artifact}")
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01b-scaled-space-selection"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    expected = (
        repo_root / "deriv-gen" / "step-01b-scaled-space-selection"
    ).resolve()
    if step_root != expected:
        raise RuntimeError(f"Unexpected study root: {step_root}")
    ensure_within(step_root, repo_root / "deriv-gen")

    config_path = step_root / "config" / "protocol.json"
    manifest_path = step_root / "inputs" / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one visible GPU is required; observed {torch.cuda.device_count()}"
        )
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    intermediate = step_root / "intermediate"
    outputs = step_root / "outputs"
    tables = outputs / "tables"
    raw = outputs / "raw"
    figures = outputs / "figures"
    examples = outputs / "examples"
    state = step_root / "state"
    for directory in (
        intermediate,
        outputs,
        tables,
        raw,
        figures,
        examples,
        state,
    ):
        ensure_within(directory, step_root).mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()
    atomic_write_json(
        state / "ANALYSIS_RUNNING.json",
        {
            "schema_version": 1,
            "status": "running",
            "started_at": started_at,
            "pid": os.getpid(),
        },
        step_root,
    )
    print(f"[{utc_now()}] validating immutable inputs and stage seals", flush=True)
    input_paths, input_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )
    export_seal = validate_stage_seal(step_root, "EXPORT_COMPLETE.json")
    fragmentation_seal = validate_stage_seal(
        step_root, "FRAGMENTATION_COMPLETE.json"
    )
    mining_seal = validate_stage_seal(
        step_root, "MMP_MINING_COMPLETE.json"
    )
    if int(export_seal["rows"]) != int(config["train_rows"]):
        raise RuntimeError("Export row count differs from frozen protocol")

    train_path = step_root / export_seal["export_path"]
    train_payload = torch.load(
        train_path, map_location="cpu", weights_only=False
    )
    validation_payload = torch.load(
        input_paths["validation_embeddings"],
        map_location="cpu",
        weights_only=False,
    )
    calibrator = torch.load(
        input_paths["calibrator"], map_location="cpu", weights_only=False
    )
    if train_payload["metadata"]["split"] != "train":
        raise RuntimeError("Train embedding split identity failed")
    if validation_payload["metadata"]["split"] != "validation":
        raise RuntimeError("Validation embedding split identity failed")
    if (
        train_payload["metadata"]["embedding_definition"]
        != "clean_graph_z_plus_mean_node_z_raw_blocks"
    ):
        raise RuntimeError("Train export is not raw graph-plus-mean blocks")
    if (
        float(
            validation_payload["metadata"]["embedding_parameters"][
                "mean_node_weight"
            ]
        )
        != 3.0
    ):
        raise RuntimeError("Validation bank is not the released weight-3 vector")
    if (
        train_payload["metadata"]["checkpoint_sha256"]
        != input_hashes["checkpoint"]
        or validation_payload["metadata"]["checkpoint_sha256"]
        != input_hashes["checkpoint"]
    ):
        raise RuntimeError("Checkpoint identity differs between embeddings")
    if (
        calibrator["metadata"]["source_embedding_sha256"]
        != "7cc13b3dd1780eafdd59fd26e9a24a20adbae333f82649fbdfa917b0333e7b77"
    ):
        raise RuntimeError("Promoted train-only calibrator binding changed")

    payload_train_hashes = [
        str(value) for value in train_payload["molecule_hashes"]
    ]
    payload_validation_hashes = [
        str(value) for value in validation_payload["molecule_hashes"]
    ]
    train_raw = payload_array(train_payload, "embeddings").astype(
        np.float32, copy=True
    )
    validation_public = payload_array(
        validation_payload, "embeddings"
    ).astype(np.float32, copy=True)
    coordinate_mean = payload_array(
        calibrator, "coordinate_mean"
    ).astype(np.float32)
    coordinate_scale = payload_array(
        calibrator, "coordinate_scale"
    ).astype(np.float32)
    if coordinate_mean.shape != (384,) or coordinate_scale.shape != (384,):
        raise RuntimeError("Calibrator coordinate schema changed")
    if not np.isfinite(coordinate_scale).all() or (
        coordinate_scale <= 0
    ).any():
        raise RuntimeError("Calibrator contains invalid scales")
    train_base = np.ascontiguousarray(
        (train_raw - coordinate_mean[None, :])
        / coordinate_scale[None, :],
        dtype=np.float32,
    )
    validation_base = np.ascontiguousarray(
        validation_public.copy(), dtype=np.float32
    )
    validation_base[:, 256:] /= 3.0
    if train_base.shape != (int(config["train_rows"]), 384):
        raise RuntimeError(f"Unexpected train base shape: {train_base.shape}")
    if validation_base.shape != (
        int(config["validation_rows"]),
        384,
    ):
        raise RuntimeError(
            f"Unexpected validation base shape: {validation_base.shape}"
        )
    if not np.isfinite(train_base).all() or not np.isfinite(
        validation_base
    ).all():
        raise RuntimeError("Non-finite calibrated coordinates")
    released_reconstruction_error = float(
        np.max(
            np.abs(
                space_matrix(validation_base, "released_hybrid_w3")
                - validation_public
            )
        )
    )
    if released_reconstruction_error > 2e-6:
        raise RuntimeError("Released weight-3 validation vector was not preserved")
    train_calibration_diagnostics = {
        "maximum_absolute_coordinate_mean": float(
            np.max(np.abs(train_base.mean(axis=0)))
        ),
        "maximum_absolute_coordinate_std_deviation_from_one": float(
            np.max(np.abs(train_base.std(axis=0) - 1.0))
        ),
        "released_validation_reconstruction_max_abs_error": (
            released_reconstruction_error
        ),
    }
    del train_raw, validation_public, coordinate_mean, coordinate_scale
    del train_payload, validation_payload, calibrator
    gc.collect()

    train_molecules = pd.read_parquet(
        intermediate / "train_molecules.parquet"
    )
    validation_molecules = pd.read_parquet(
        intermediate / "validation_molecules.parquet"
    )
    train_fragments = pd.read_parquet(
        intermediate / "train_fragments.parquet"
    )
    validation_fragments = pd.read_parquet(
        intermediate / "validation_fragments.parquet"
    )
    train_hashes = train_molecules["molecule_hash"].astype(str).tolist()
    validation_hashes = (
        validation_molecules["molecule_hash"].astype(str).tolist()
    )
    if train_hashes != payload_train_hashes:
        raise RuntimeError("Train molecule table is not row-aligned to export")
    if validation_hashes != payload_validation_hashes:
        raise RuntimeError(
            "Validation molecule table is not row-aligned to retrieval bank"
        )
    if set(train_hashes).intersection(validation_hashes):
        raise RuntimeError("Train/validation identity overlap is nonzero")
    validation_smiles = (
        validation_molecules["canonical_smiles"].astype(str).tolist()
    )
    validation_scaffolds = (
        validation_molecules["scaffold"].fillna("").astype(str).tolist()
    )
    validation_heavy = validation_molecules["heavy_atoms"].to_numpy(
        dtype=np.int16
    )

    train_observations = pd.read_parquet(
        intermediate / "train_mmp_observations.parquet"
    )
    eligible = pd.read_parquet(
        intermediate / "eligible_validation_mmp_observations.parquet"
    )
    support = pd.read_parquet(
        intermediate / "train_transformation_support.parquet"
    )
    minimum_cores = int(config["retrieval"]["minimum_train_cores"])
    supported = set(
        support.loc[
            support["train_cores"] >= minimum_cores, "transform"
        ].astype(str)
    )
    train_observations = train_observations.loc[
        train_observations["transform"].astype(str).isin(supported)
    ].reset_index(drop=True)
    eligible = add_observation_ids(eligible, seed)
    support_by_transform = dict(
        zip(
            support["transform"].astype(str),
            support["train_cores"].astype(int),
        )
    )
    if eligible.empty:
        raise RuntimeError("No unseen-core validation observations are eligible")
    print(
        f"[{utc_now()}] fitting directions for {len(supported):,} supported transformations",
        flush=True,
    )

    directions_by_space: dict[str, dict[str, dict[str, Any]]] = {}
    covariance_by_space: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    direction_metadata_frames: list[pd.DataFrame] = []
    for space in SPACE_ORDER:
        print(f"[{utc_now()}] direction fit and covariance: {space}", flush=True)
        train_space = space_matrix(train_base, space)
        directions = fit_directions(
            train_observations,
            train_space,
            minimum_cores=minimum_cores,
        )
        if len(directions) < 2:
            raise RuntimeError(f"Too few fitted directions in {space}")
        eigenvalues, eigenvectors = covariance_eigendecomposition(
            train_space, device
        )
        directions_by_space[space] = directions
        covariance_by_space[space] = (eigenvalues, eigenvectors)
        transforms = np.asarray(sorted(directions), dtype=np.str_)
        atomic_save_npz(
            intermediate / f"mmp_directions_{space}.npz",
            step_root,
            transforms=transforms,
            unit_directions=np.stack(
                [directions[value]["unit"] for value in transforms]
            ).astype(np.float32),
            median_step_norms=np.asarray(
                [directions[value]["median_norm"] for value in transforms],
                dtype=np.float32,
            ),
            train_cores=np.asarray(
                [directions[value]["train_cores"] for value in transforms],
                dtype=np.int32,
            ),
        )
        direction_metadata_frames.append(
            pd.DataFrame(
                [
                    {
                        "space": space,
                        "transform": transform,
                        **{
                            key: value
                            for key, value in record.items()
                            if key != "unit"
                        },
                    }
                    for transform, record in directions.items()
                ]
            )
        )
        del train_space
        gc.collect()
        torch.cuda.empty_cache()

    common_transforms = set.intersection(
        *[
            set(directions_by_space[space])
            for space in SPACE_ORDER
        ]
    )
    if len(common_transforms) < 2:
        raise RuntimeError("Five-space common direction set is too small")
    eligible = assign_mismatched_transforms(
        eligible,
        common_transforms,
        support_by_transform,
        seed,
    )
    queries = select_queries(
        eligible,
        common_transforms,
        maximum=int(config["retrieval"]["maximum_queries"]),
        per_transform=int(
            config["retrieval"]["maximum_queries_per_transform"]
        ),
        primary_support=int(
            config["retrieval"]["primary_support_threshold"]
        ),
        seed=seed,
    )
    queries["seed_hash"] = [
        validation_hashes[int(value)] for value in queries["lhs_index"]
    ]
    queries["target_hash"] = [
        validation_hashes[int(value)] for value in queries["rhs_index"]
    ]
    queries["seed_smiles"] = [
        validation_smiles[int(value)] for value in queries["lhs_index"]
    ]
    queries["target_smiles"] = [
        validation_smiles[int(value)] for value in queries["rhs_index"]
    ]
    atomic_write_csv(
        intermediate / "retrieval_queries.csv", queries, step_root
    )
    atomic_write_csv(
        tables / "mmp_direction_fit_by_transformation.csv",
        pd.concat(direction_metadata_frames, ignore_index=True),
        step_root,
    )
    print(
        f"[{utc_now()}] common directions={len(common_transforms):,}; retrieval queries={len(queries):,}",
        flush=True,
    )

    print(f"[{utc_now()}] building validation chemistry indices", flush=True)
    fingerprints = make_fingerprints(validation_smiles)
    validation_core_sets = core_sets(
        validation_fragments, len(validation_molecules)
    )
    requested_targets = requested_target_sets(validation_fragments)

    alignment_frames: list[pd.DataFrame] = []
    retrieval_frames: list[pd.DataFrame] = []
    for space in SPACE_ORDER:
        print(f"[{utc_now()}] alignment and retrieval: {space}", flush=True)
        train_space = space_matrix(train_base, space)
        validation_space = space_matrix(validation_base, space)
        alignment_frames.append(
            evaluate_alignment(
                eligible,
                validation_space,
                directions_by_space[space],
                space=space,
            )
        )
        eigenvalues, eigenvectors = covariance_by_space[space]
        retrieval_frames.append(
            retrieval_experiment(
                space=space,
                train_space=train_space,
                validation_space=validation_space,
                queries=queries,
                directions=directions_by_space[space],
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                fingerprints=fingerprints,
                hashes=validation_hashes,
                smiles=validation_smiles,
                scaffolds=validation_scaffolds,
                heavy_atoms=validation_heavy,
                validation_core_sets=validation_core_sets,
                requested_targets=requested_targets,
                device=device,
                local_neighbors=int(
                    config["retrieval"]["local_train_neighbors"]
                ),
                top_k=int(config["retrieval"]["top_k"]),
                summary_top_k=int(
                    config["retrieval"]["summary_top_k"]
                ),
                random_replicates=int(
                    config["retrieval"]["random_replicates"]
                ),
                batch_size=int(
                    config["retrieval"]["gpu_query_batch_size"]
                ),
                seed=seed,
            )
        )
        del train_space, validation_space
        gc.collect()
        torch.cuda.empty_cache()

    alignment = pd.concat(alignment_frames, ignore_index=True)
    retrieval = pd.concat(retrieval_frames, ignore_index=True)
    retrieval_average = average_replicates(retrieval)
    atomic_write_parquet(
        raw / "mmp_alignment_per_observation.parquet",
        alignment,
        step_root,
    )
    atomic_write_parquet(
        raw / "retrieval_per_query_and_replicate.parquet",
        retrieval,
        step_root,
    )
    atomic_write_parquet(
        raw / "retrieval_per_query.parquet",
        retrieval_average,
        step_root,
    )

    alignment_transform = by_transformation(
        alignment,
        ("alignment", "null_alignment", "alignment_gain"),
        analysis="alignment_all",
    )
    retrieval_transform_frames = [
        by_transformation(
            retrieval_average,
            RETRIEVAL_METRICS,
            analysis="retrieval",
            method=method,
        )
        for method in METHOD_ORDER
    ]
    retrieval_transform = pd.concat(
        retrieval_transform_frames, ignore_index=True
    )
    retrieval_table = retrieval_summary(retrieval_average)
    atomic_write_csv(
        tables / "alignment_by_transformation.csv",
        alignment_transform,
        step_root,
    )
    atomic_write_csv(
        tables / "retrieval_by_transformation.csv",
        retrieval_transform,
        step_root,
    )
    atomic_write_csv(
        tables / "retrieval_summary.csv", retrieval_table, step_root
    )

    print(
        f"[{utc_now()}] hierarchical paired bootstrap "
        f"({config['statistics']['hierarchical_bootstrap_resamples']} replicates)",
        flush=True,
    )
    cohorts = [
        int(value) for value in config["statistics"]["support_cohorts"]
    ]
    resamples = int(
        config["statistics"]["hierarchical_bootstrap_resamples"]
    )
    alpha = float(config["statistics"]["ci_alpha"])
    alignment_boot, alignment_diff = hierarchical_bootstrap(
        alignment,
        metrics=("alignment", "null_alignment", "alignment_gain"),
        analysis="alignment_all",
        cohorts=cohorts,
        resamples=resamples,
        alpha=alpha,
        seed=seed,
    )
    mmp_retrieval = retrieval_average.loc[
        retrieval_average["method"] == "mmp_direction"
    ].copy()
    retrieval_boot, retrieval_diff = hierarchical_bootstrap(
        mmp_retrieval,
        metrics=RETRIEVAL_METRICS,
        analysis="retrieval_mmp_direction",
        cohorts=cohorts,
        resamples=resamples,
        alpha=alpha,
        seed=seed + 1,
    )
    bootstrap_summary = pd.concat(
        [alignment_boot, retrieval_boot], ignore_index=True
    )
    paired_differences = pd.concat(
        [alignment_diff, retrieval_diff], ignore_index=True
    )
    atomic_write_csv(
        tables / "hierarchical_bootstrap_summary.csv",
        bootstrap_summary,
        step_root,
    )
    atomic_write_csv(
        tables / "paired_differences_vs_released_w3.csv",
        paired_differences,
        step_root,
    )

    decision, decision_table = select_control_space(
        config, bootstrap_summary, retrieval_average, paired_differences
    )
    support_thresholds = pd.read_csv(
        tables / "mmp_support_thresholds.csv"
    )
    write_reports(
        step_root=step_root,
        decision=decision,
        decision_table=decision_table,
        support_thresholds=support_thresholds,
        bootstrap_summary=bootstrap_summary,
        retrieval_summary=retrieval_table,
        train_rows=len(train_base),
        validation_rows=len(validation_base),
        query_count=len(queries),
        mining_summary=mining_seal["summary"],
    )
    atomic_write_csv(
        tables / "space_selection.csv", decision_table, step_root
    )
    atomic_write_json(
        outputs / "space_decision.json", decision, step_root
    )
    build_figures(
        bootstrap_summary,
        retrieval_table,
        figures,
        step_root,
        int(config["retrieval"]["primary_support_threshold"]),
    )
    atomic_write_csv(
        examples / "mmp_direction_top1_examples.csv",
        retrieval.loc[
            (retrieval["method"] == "mmp_direction")
            & (retrieval["replicate"] == 0),
            [
                "space",
                "query_id",
                "transform",
                "train_cores",
                "seed_hash",
                "true_target_hash",
                "top1_candidate_hash",
                "seed_smiles",
                "true_target_smiles",
                "top1_candidate_smiles",
                "target_rank_within_50",
                "exact_requested_transform",
                "mmp_consistency",
                "seed_retrieved_tanimoto",
            ],
        ],
        step_root,
    )

    study_summary = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "data": {
            "train_rows": len(train_base),
            "validation_rows": len(validation_base),
            "test_rows": 0,
            "identity_overlap": 0,
            "sampling_seed": int(config["train_sampling_seed"]),
            "checkpoint_sha256": input_hashes["checkpoint"],
            "calibrator_sha256": input_hashes["calibrator"],
            "validation_embedding_sha256": input_hashes[
                "validation_embeddings"
            ],
        },
        "calibration_diagnostics": train_calibration_diagnostics,
        "fragmentation": {
            "train": fragmentation_seal["train"],
            "validation": fragmentation_seal["validation"],
        },
        "mmp_mining": mining_seal["summary"],
        "common_fitted_transformations": len(common_transforms),
        "retrieval_queries": len(queries),
        "spaces": list(SPACE_ORDER),
        "methods": list(METHOD_ORDER),
        "statistics": config["statistics"],
        "decision": decision,
        "claim_boundary": (
            "Retrieval geometry only; no decoder or novel-molecule "
            "generation was tested."
        ),
    }
    atomic_write_json(
        outputs / "study_summary.json", study_summary, step_root
    )

    finished_at = utc_now()
    wall_time = time.perf_counter() - started
    runtime = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": wall_time,
        "command": sys.argv,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "rdkit": rdkit.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "gpu": {
            "visible_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
        },
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "git_commit": git_output(repo_root, "rev-parse", "HEAD"),
        "git_status_at_completion": git_output(
            repo_root, "status", "--short"
        ),
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_sha256": input_hashes,
        "stage_seal_sha256": {
            "export": sha256_file(
                state / "EXPORT_COMPLETE.json"
            ),
            "fragmentation": sha256_file(
                state / "FRAGMENTATION_COMPLETE.json"
            ),
            "mmp_mining": sha256_file(
                state / "MMP_MINING_COMPLETE.json"
            ),
        },
    }
    atomic_write_json(state / "run_metadata.json", runtime, step_root)
    ledger_text = hash_ledger(
        outputs, exclude={"SHA256SUMS"}
    )
    atomic_write_text(outputs / "SHA256SUMS", ledger_text, step_root)
    complete = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "complete",
        "finished_at": finished_at,
        "wall_time_seconds": wall_time,
        "single_gpu": True,
        "gpu": torch.cuda.get_device_name(0),
        "train_rows": len(train_base),
        "validation_rows": len(validation_base),
        "test_rows": 0,
        "retrieval_queries": len(queries),
        "common_fitted_transformations": len(common_transforms),
        "selected_edit_control_space": decision[
            "selected_edit_control_space"
        ],
        "decoder_conditioning_representation": decision[
            "decoder_conditioning_representation"
        ],
        "results_sha256": sha256_file(step_root / "RESULTS.md"),
        "decision_sha256": sha256_file(step_root / "DECISION.md"),
        "output_ledger_sha256": sha256_file(
            outputs / "SHA256SUMS"
        ),
    }
    atomic_write_json(state / "COMPLETE.json", complete, step_root)
    (state / "ANALYSIS_RUNNING.json").unlink(missing_ok=True)
    print(
        f"[{utc_now()}] analysis complete in {wall_time:.1f}s; "
        f"selected={decision['selected_edit_control_space']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
