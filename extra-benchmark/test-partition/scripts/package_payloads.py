#!/usr/bin/env python3
"""Bind native comparator matrices to exact gMolAI panel identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_json,
    load_protocol,
    protocol_digest,
    read_panel_tsv,
    require_hash,
    sha256_file,
    sha256_lines,
)


MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument(
        "--repair-derived-metadata",
        action="store_true",
        help=(
            "Atomically replace an existing derived gMolAI payload only when "
            "its tensors and all metadata except embedding_parameters already "
            "match the frozen reconstruction."
        ),
    )
    return parser.parse_args()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite packaged payload: {path}")
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".partial", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_replace(path: Path, value: Any) -> None:
    """Atomically replace one explicitly validated derived payload."""
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".partial", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def payloads_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if set(first) != set(second):
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                return False
        elif left != right:
            return False
    return True


def payloads_differ_only_in_embedding_parameters(
    existing: dict[str, Any], desired: dict[str, Any]
) -> bool:
    """Confirm that a repair cannot alter vectors, identities, or other metadata."""
    if set(existing) != set(desired):
        return False
    for key in existing:
        left, right = existing[key], desired[key]
        if key == "metadata":
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False
            left = dict(left)
            right = dict(right)
            left.pop("embedding_parameters", None)
            right.pop("embedding_parameters", None)
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
                return False
        elif left != right:
            return False
    return True


def source_indices(
    source: dict[str, Any], common_rows: list[dict[str, str]]
) -> np.ndarray:
    positions: dict[str, int] = {}
    for index, molecule_hash in enumerate(source["molecule_hashes"]):
        key = str(molecule_hash)
        if key in positions:
            raise RuntimeError("Authoritative source contains duplicate molecule hashes")
        positions[key] = index
    missing = [row["molecule_hash"] for row in common_rows if row["molecule_hash"] not in positions]
    if missing:
        raise RuntimeError(f"Common panel has {len(missing)} identities absent from source")
    return np.asarray([positions[row["molecule_hash"]] for row in common_rows], dtype=np.int64)


def comparator_metadata(
    model: str,
    specification: dict[str, Any],
    source_metadata: dict[str, Any],
    split: str,
    rows: int,
    identity_sha256: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if model == "gmolai":
        checkpoint = source_metadata["checkpoint"]
        checkpoint_sha256 = source_metadata["checkpoint_sha256"]
        global_step = int(source_metadata["global_step"])
    else:
        checkpoint = specification.get("model_revision", specification["representation"])
        checkpoint_sha256 = specification["container_sha256"]
        global_step = 0
    digest = protocol_digest(protocol)
    if model == "gmolai":
        embedding_parameters = dict(source_metadata["embedding_parameters"])
    else:
        embedding_parameters = {
            "native_dimension": int(specification["dimension"]),
            "precision": "float32",
            "frozen_protocol_digest": digest,
        }
    return {
        "schema_version": 1,
        "architecture": model,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "global_step": global_step,
        "dimensions": int(specification["dimension"]),
        "embedding_definition": specification["representation"],
        "embedding_parameters": embedding_parameters,
        "config_hash": digest,
        "training_plan_hash": "not_applicable_frozen_inference_" + digest,
        "graph_manifest_hash": source_metadata["graph_manifest_hash"],
        "descriptor_schema_hash": source_metadata["descriptor_schema_hash"],
        "sampling": "authoritative_panel_filtered_to_all_comparator_common_coverage",
        "sampling_seed": 20260810,
        "split": split,
        "graphs": rows,
        "common_identity_sha256": identity_sha256,
        "execution": "inference_only",
        "uses_3d": False,
    }


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    source_key = "train" if args.split == "train" else "test"
    common_name = "train" if args.split == "train" else "test"
    panel_path = BENCHMARK_DIR / "inputs" / f"common_{common_name}.tsv"
    common_rows = read_panel_tsv(panel_path)
    identity_sha256 = sha256_lines(row["molecule_hash"] for row in common_rows)
    common_state = load_json(BENCHMARK_DIR / "state" / f"common_{common_name}.json")
    if common_state.get("common_identity_sha256") != identity_sha256:
        raise RuntimeError("Common-panel identity digest differs from assembly state")

    source_specification = protocol["authoritative_panels"][source_key]
    source_path = REPOSITORY_ROOT / source_specification["path"]
    require_hash(source_path, source_specification["sha256"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    indices = source_indices(source, common_rows)
    index_tensor = torch.from_numpy(indices)
    packaged: dict[str, dict[str, Any]] = {}

    for model in MODELS:
        specification = protocol["comparators"][model]
        dimension = int(specification["dimension"])
        if model == "gmolai":
            embeddings = source["embeddings"].index_select(0, index_tensor).float()
            source_matrix_sha256 = source_specification["sha256"]
        else:
            matrix_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}-{common_name}.npy"
            metadata_path = BENCHMARK_DIR / "outputs" / "embeddings" / f"{model}-{common_name}.json"
            adapter_metadata = load_json(metadata_path)
            if adapter_metadata.get("ordered_identity_sha256") != identity_sha256:
                raise RuntimeError(f"{model} adapter identity differs from common {common_name}")
            if adapter_metadata.get("output_sha256") != sha256_file(matrix_path):
                raise RuntimeError(f"{model} adapter output hash differs from its metadata")
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            if matrix.shape != (len(common_rows), dimension) or matrix.dtype != np.float32:
                raise RuntimeError(f"Invalid native matrix for {model} {common_name}")
            embeddings = torch.from_numpy(np.asarray(matrix).copy())
            source_matrix_sha256 = sha256_file(matrix_path)
        if embeddings.shape != (len(common_rows), dimension) or not torch.isfinite(embeddings).all():
            raise RuntimeError(f"Invalid packaged embedding for {model} {common_name}")

        payload = {
            "metadata": comparator_metadata(
                model,
                specification,
                source["metadata"],
                args.split,
                len(common_rows),
                identity_sha256,
                protocol,
            ),
            "embeddings": embeddings.contiguous(),
            "standardized_descriptor_targets": source[
                "standardized_descriptor_targets"
            ].index_select(0, index_tensor),
            "graph_ids": source["graph_ids"].index_select(0, index_tensor),
            "source_buckets": source["source_buckets"].index_select(0, index_tensor),
            "molecule_hashes": [row["molecule_hash"] for row in common_rows],
        }
        metadata_repaired = False
        destination = BENCHMARK_DIR / "outputs" / "payloads" / f"{model}-{common_name}.pt"
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            if isinstance(existing, dict) and payloads_equal(existing, payload):
                pass
            elif (
                args.repair_derived_metadata
                and model == "gmolai"
                and isinstance(existing, dict)
                and payloads_differ_only_in_embedding_parameters(existing, payload)
            ):
                atomic_torch_replace(destination, payload)
                metadata_repaired = True
            else:
                raise RuntimeError(
                    "Existing packaged payload differs from the frozen result: "
                    f"{destination}"
                )
        else:
            atomic_torch_save(destination, payload)
        packaged[model] = {
            "path": str(destination.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(destination),
            "source_matrix_sha256": source_matrix_sha256,
            "rows": len(common_rows),
            "dimension": dimension,
            "identity_sha256": identity_sha256,
            "metadata_repaired": metadata_repaired,
        }

    manifest_path = BENCHMARK_DIR / "state" / f"packaged_{common_name}.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "ok",
            "split": args.split,
            "common_panel_sha256": sha256_file(panel_path),
            "common_identity_sha256": identity_sha256,
            "models": packaged,
        },
    )
    print(json.dumps({"status": "ok", "split": args.split, "models": packaged}, sort_keys=True))


if __name__ == "__main__":
    main()
