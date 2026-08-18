#!/usr/bin/env python3
"""Fail-closed verification of completed benchmark bindings and deliverables."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_io import (
    BENCHMARK_DIR, atomic_write_json, load_json, load_protocol, protocol_digest,
    read_panel_tsv, sha256_file,
)


MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2", "morgan_count", "descriptor13")


def main() -> None:
    protocol = load_protocol()
    failures = []
    verified = []
    preflight_path = BENCHMARK_DIR / "audit" / "preflight.json"
    preflight = load_json(preflight_path)
    if preflight.get("status") != "ok" or preflight.get("protocol_sha256") != protocol_digest(protocol):
        failures.append("preflight/protocol digest binding failed")
    verified.append(str(preflight_path))
    partials = sorted(str(path) for path in BENCHMARK_DIR.rglob("*.partial"))
    if partials:
        failures.append(f"stale partial outputs: {partials[:10]}")
    for benchmark in ("classyfire", "qmugs"):
        panel_path = BENCHMARK_DIR / "inputs" / "prepared" / f"{benchmark}_common.tsv"
        rows = read_panel_tsv(panel_path)
        identities = [row["molecule_hash"] for row in rows]
        if len(identities) != len(set(identities)):
            failures.append(f"duplicate final identities: {benchmark}")
        common_state_path = BENCHMARK_DIR / "state" / f"{benchmark}_common.json"
        common_state = load_json(common_state_path)
        if common_state.get("status") != "ok" or int(common_state.get("final_rows", -1)) != len(rows):
            failures.append(f"common-support state failed: {benchmark}")
        if common_state.get("final_panel_sha256") != sha256_file(panel_path):
            failures.append(f"common-support panel hash failed: {benchmark}")
        if benchmark == "classyfire":
            counts = Counter(row["subclass"] for row in rows)
            expected = int(common_state["balance_per_subclass"])
            if len(counts) != 25 or set(counts.values()) != {expected}:
                failures.append("ClassyFire balance contract failed")
        elif not common_state.get("coverage_failure") and len(rows) != int(common_state["target_rows"]):
            failures.append("QMugs target-size contract failed")
        verified.extend([str(panel_path), str(common_state_path)])
        for model in MODELS:
            audit = BENCHMARK_DIR / "audit" / f"embedding-{model}-{benchmark}.json"
            record = load_json(audit)
            embedding = BENCHMARK_DIR / "artifacts" / "embeddings" / benchmark / f"{model}.npy"
            matrix = np.load(embedding, mmap_mode="r", allow_pickle=False)
            if record.get("status") != "ok" or matrix.shape[0] != len(rows):
                failures.append(f"embedding binding failed: {model}/{benchmark}")
            if record.get("embedding_sha256") != sha256_file(embedding):
                failures.append(f"embedding hash failed: {model}/{benchmark}")
            verified.extend([str(audit), str(embedding)])
    property_reference = BENCHMARK_DIR / "artifacts" / "common" / "qmugs_property_reference.npz"
    property_reference_hash = sha256_file(property_reference)
    for model in MODELS:
        structural_state_path = BENCHMARK_DIR / "state" / "structural" / f"{model}.json"
        property_state_path = BENCHMARK_DIR / "state" / "property" / f"{model}.json"
        structural_state = load_json(structural_state_path)
        property_state = load_json(property_state_path)
        structural_embedding = BENCHMARK_DIR / "artifacts" / "embeddings" / "classyfire" / f"{model}.npy"
        property_embedding = BENCHMARK_DIR / "artifacts" / "embeddings" / "qmugs" / f"{model}.npy"
        if structural_state.get("status") != "ok" or structural_state.get("kmeans") != protocol["structural_evaluation"]["kmeans"]:
            failures.append(f"structural state/specification failed: {model}")
        if structural_state.get("input_embedding_sha256") != sha256_file(structural_embedding):
            failures.append(f"structural embedding binding failed: {model}")
        if property_state.get("status") != "ok" or int(property_state.get("k", -1)) != int(protocol["property_evaluation"]["k"]):
            failures.append(f"property state/specification failed: {model}")
        if property_state.get("embedding_sha256") != sha256_file(property_embedding):
            failures.append(f"property embedding binding failed: {model}")
        if property_state.get("property_reference_sha256") != property_reference_hash:
            failures.append(f"property-reference binding failed: {model}")
        for key in ("cluster_labels", "neighbors", "query_source_data"):
            path = Path(structural_state[key])
            if not path.is_file() or sha256_file(path) != structural_state[f"{key}_sha256"]:
                failures.append(f"structural artifact hash failed: {model}/{key}")
            else:
                verified.append(str(path))
        for key in ("neighbors", "query_source_data"):
            path = Path(property_state[key])
            if not path.is_file() or sha256_file(path) != property_state[f"{key}_sha256"]:
                failures.append(f"property artifact hash failed: {model}/{key}")
            else:
                verified.append(str(path))
        verified.extend([str(structural_state_path), str(property_state_path)])
    verified.append(str(property_reference))
    structural = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "classyfire_structural_metrics.csv")
    prop = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_metrics.csv")
    for name, frame, expected_metrics in (
        ("structural", structural, {"ARI", "AMI", "NMI", "macro_same_subclass_at_100"}),
        ("property", prop, {"NPD_at_100", "property_neighbor_recall_at_100"}),
    ):
        if set(frame["model"]) != set(MODELS) or set(frame["metric"]) != expected_metrics:
            failures.append(f"{name} metric coverage differs")
        numeric = frame[["estimate", "ci95_lower", "ci95_upper"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric[:, 1] > numeric[:, 0]) or np.any(numeric[:, 0] > numeric[:, 2]):
            failures.append(f"{name} estimates/intervals invalid")
    figure_manifest = load_json(BENCHMARK_DIR / "audit" / "figure_manifest.json")
    for figure in figure_manifest["figures"]:
        for item in figure["files"] + figure["source_data"]:
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                failures.append(f"figure/source hash failed: {path}")
            verified.append(str(path))
    for required in (
        BENCHMARK_DIR / "PROTOCOL.md", BENCHMARK_DIR / "RESULTS.md",
        BENCHMARK_DIR / "audit" / "container_packages.json",
        BENCHMARK_DIR / "audit" / "pretraining_exposure.json",
        BENCHMARK_DIR / "state" / "structural_summary.json",
        BENCHMARK_DIR / "state" / "property_summary.json",
        BENCHMARK_DIR / "state" / "pca_visualization.json",
    ):
        if not required.is_file():
            failures.append(f"required deliverable missing: {required}")
        else:
            verified.append(str(required))
    container_packages = load_json(BENCHMARK_DIR / "audit" / "container_packages.json")
    if container_packages.get("status") != "ok":
        failures.append("container package inventory failed")
    else:
        for inventory in container_packages.get("inventories", {}).values():
            path = Path(inventory["inventory"])
            if not path.is_file() or sha256_file(path) != inventory["inventory_sha256"]:
                failures.append(f"container package inventory hash failed: {path}")
            else:
                verified.append(str(path))
    result = {
        "schema_version": 1, "status": "failed" if failures else "ok",
        "failures": failures, "verified_paths": sorted(set(verified)),
        "verified_path_count": len(set(verified)),
        "protocol_status": protocol["protocol_status"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = BENCHMARK_DIR / "audit" / "verification.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
