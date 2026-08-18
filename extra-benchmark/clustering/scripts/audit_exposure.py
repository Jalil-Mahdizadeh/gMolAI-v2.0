#!/usr/bin/env python3
"""Audit exact gMolAI corpus overlap and presentation before frozen step 10,000."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import duckdb
import pyarrow as pa

from benchmark_io import (
    BENCHMARK_DIR, REPOSITORY_ROOT, atomic_write_json, load_protocol,
    read_panel_tsv, sha256_file, write_csv,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from gmolai_retrain.config import apply_training_plan, load_config  # noqa: E402
from gmolai_retrain.downstream_audit import _identity_digest, _parquet_files_by_bucket  # noqa: E402
from gmolai_retrain.downstream_exposure import (  # noqa: E402
    _checkpoint_records, _scan_target_locations, _seen_at_cycle_zero_cursor,
)
from gmolai_retrain.util import runtime_versions, stable_u64  # noqa: E402


def join_corpus(cfg, manifest, panels):
    bucket_count = int(cfg["data"]["hash_buckets"])
    parquet_by_bucket = _parquet_files_by_bucket(manifest, bucket_count)
    wanted = defaultdict(lambda: {"benchmark": [], "panel_index": [], "molecule_hash": [], "canonical_smiles": []})
    matches = {name: [None] * len(rows) for name, rows in panels.items()}
    for name, rows in panels.items():
        for index, row in enumerate(rows):
            bucket = int(stable_u64(row["canonical_smiles"]) % bucket_count)
            target = wanted[bucket]
            target["benchmark"].append(name)
            target["panel_index"].append(index)
            target["molecule_hash"].append(row["molecule_hash"])
            target["canonical_smiles"].append(row["canonical_smiles"])
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("PRAGMA threads=8")
        for bucket, columns in sorted(wanted.items()):
            connection.register("wanted", pa.table(columns))
            joined = connection.execute(
                """
                SELECT w.benchmark, w.panel_index, w.molecule_hash,
                       w.canonical_smiles, d.canonical_smiles, d.split
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet_by_bucket[bucket])],
            ).fetchall()
            connection.unregister("wanted")
            for row in joined:
                name, index = str(row[0]), int(row[1])
                if matches[name][index] is not None:
                    raise RuntimeError("Pretraining corpus returned a duplicate identity")
                if str(row[3]) != str(row[4]):
                    raise RuntimeError("SHA-256 collision/canonical mismatch in corpus join")
                matches[name][index] = {"molecule_hash": str(row[2]), "split": str(row[5])}
    finally:
        connection.close()
    return matches


def main() -> None:
    protocol = load_protocol()
    cfg = load_config(REPOSITORY_ROOT / protocol["gmolai"]["config"]["path"])
    apply_training_plan(cfg, REPOSITORY_ROOT / protocol["gmolai"]["training_plan"]["path"])
    checkpoint_path = REPOSITORY_ROOT / protocol["gmolai"]["checkpoint"]["path"]
    cfg["paths"]["run_dir"] = str(checkpoint_path.parent.parent)
    panels = {
        "classyfire": read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv"),
        "qmugs": read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv"),
    }
    dataset_manifest_path = Path(cfg["paths"]["work_dir"]) / "dataset_manifest.json"
    graph_manifest_path = Path(cfg["paths"]["work_dir"]) / "graph_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError("Dataset manifest configuration differs")
    if graph_manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError("Graph manifest configuration differs")
    if graph_manifest["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Graph/dataset manifest identity differs")
    matches = join_corpus(cfg, dataset_manifest, panels)
    split_hashes = {}
    train_union = set()
    for name, rows in panels.items():
        current = {"train": set(), "validation": set(), "test": set()}
        for row, match in zip(rows, matches[name]):
            if match is not None:
                current[match["split"]].add(row["molecule_hash"])
        split_hashes[name] = current
        train_union.update(current["train"])
    train_shards = [entry for entry in graph_manifest["shards"] if entry["split"] == "train"]
    checkpoints = _checkpoint_records(
        cfg, graph_manifest=graph_manifest, train_shards=train_shards,
        checkpoint_names=[str(checkpoint_path)],
    )
    checkpoint = checkpoints[0]
    training_seed = int(cfg.get("training", {}).get("seed", cfg["seed"]))
    locations, scan = _scan_target_locations(
        train_shards, target_hashes=train_union, seed=training_seed,
        world_size=int(checkpoint["world_size"]), workers=min(8, os.cpu_count() or 1),
    )
    seen = {
        molecule_hash for molecule_hash, location in locations.items()
        if _seen_at_cycle_zero_cursor(location, checkpoint["data_states"][int(location["rank"])])
    }
    summary_rows = []
    ledger_rows = []
    datasets = {}
    for name, rows in panels.items():
        split = split_hashes[name]
        corpus = set().union(*split.values())
        dataset_seen = split["train"] & seen
        summary = {
            "benchmark": name, "panel_rows": len(rows),
            "pretraining_corpus_overlap": len(corpus),
            "pretraining_corpus_overlap_percent": 100.0 * len(corpus) / len(rows),
            "pretraining_train_overlap": len(split["train"]),
            "pretraining_validation_overlap": len(split["validation"]),
            "pretraining_test_overlap": len(split["test"]),
            "seen_before_step_10000": len(dataset_seen),
            "seen_before_step_10000_percent_panel": 100.0 * len(dataset_seen) / len(rows),
            "seen_before_step_10000_percent_train_overlap": (
                100.0 * len(dataset_seen) / len(split["train"]) if split["train"] else 0.0
            ),
        }
        summary_rows.append(summary)
        datasets[name] = {
            **summary,
            "accepted_identity_set_sha256": _identity_digest([row["molecule_hash"] for row in rows]),
            "corpus_overlap_identity_set_sha256": _identity_digest(sorted(corpus)),
            "train_overlap_identity_set_sha256": _identity_digest(sorted(split["train"])),
            "seen_identity_set_sha256": _identity_digest(sorted(dataset_seen)),
        }
        for index, (row, match) in enumerate(zip(rows, matches[name])):
            location = locations.get(row["molecule_hash"])
            ledger_rows.append({
                "benchmark": name, "panel_index": index,
                "molecule_hash": row["molecule_hash"],
                "canonical_smiles": row["canonical_smiles"],
                "pretraining_corpus_overlap": match is not None,
                "pretraining_split": "" if match is None else match["split"],
                "seen_before_step_10000": row["molecule_hash"] in dataset_seen,
                "rank": "" if location is None else location["rank"],
                "manifest_train_shard_index": "" if location is None else location["manifest_train_shard_index"],
                "graph_index_in_shard": "" if location is None else location["graph_index_in_shard"],
            })
    summary_path = BENCHMARK_DIR / "outputs" / "tables" / "gmolai_pretraining_exposure.csv"
    write_csv(summary_path, summary_rows, tuple(summary_rows[0]))
    ledger_path = BENCHMARK_DIR / "audit" / "gmolai_pretraining_exposure_ledger.csv"
    write_csv(ledger_path, ledger_rows, tuple(ledger_rows[0]))
    competitor_rows = []
    for model in protocol["models"]["primary_order"]:
        for benchmark in panels:
            if model == "gmolai":
                record = next(row for row in summary_rows if row["benchmark"] == benchmark)
                status = "exact identity and presentation audit complete"
                exact = True
                overlap = record["pretraining_corpus_overlap"]
            else:
                status = "published checkpoint/corpus provenance only; exact identity manifest unavailable"
                exact = False
                overlap = ""
            competitor_rows.append({
                "model": model, "benchmark": benchmark,
                "exact_identity_exposure_known": exact,
                "exact_corpus_overlap": overlap, "status": status,
                "interpretation": "no unseen/OOD claim",
            })
    competitor_path = BENCHMARK_DIR / "outputs" / "tables" / "model_pretraining_exposure_status.csv"
    write_csv(competitor_path, competitor_rows, tuple(competitor_rows[0]))
    result = {
        "schema_version": 1, "status": "ok",
        "audit": "exact canonical corpus overlap and graph presentation before step 10000",
        "pretrained_model_executed": False, "training_permitted": False,
        "checkpoint": checkpoint, "training_stream_seed": training_seed,
        "corpus": {
            "dataset_manifest": str(dataset_manifest_path), "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
            "graph_manifest": str(graph_manifest_path), "graph_manifest_sha256": sha256_file(graph_manifest_path),
            "graph_manifest_hash": graph_manifest["graph_manifest_hash"],
            "graphs_total": int(graph_manifest["counts"]["graphs_total"]),
            "graphs_train": int(graph_manifest["counts"]["graphs_train"]),
        },
        "datasets": datasets, "scan": scan,
        "summary_table": str(summary_path), "summary_table_sha256": sha256_file(summary_path),
        "identity_ledger": str(ledger_path), "identity_ledger_sha256": sha256_file(ledger_path),
        "competitor_status_table": str(competitor_path),
        "competitor_status_table_sha256": sha256_file(competitor_path),
        "no_unseen_or_ood_claim": True, "runtime": runtime_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audit" / "pretraining_exposure.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

