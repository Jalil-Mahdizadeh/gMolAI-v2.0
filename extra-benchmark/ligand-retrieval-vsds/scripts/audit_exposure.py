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
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    read_panel_tsv,
    read_tsv,
    sha256_file,
    write_csv,
)

sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from gmolai_retrain.config import apply_training_plan, load_config  # noqa: E402
from gmolai_retrain.downstream_audit import (  # noqa: E402
    _identity_digest,
    _parquet_files_by_bucket,
)
from gmolai_retrain.downstream_exposure import (  # noqa: E402
    _checkpoint_records,
    _scan_target_locations,
    _seen_at_cycle_zero_cursor,
)
from gmolai_retrain.util import runtime_versions, stable_u64  # noqa: E402


def join_corpus(cfg, manifest, panel):
    bucket_count = int(cfg["data"]["hash_buckets"])
    parquet_by_bucket = _parquet_files_by_bucket(manifest, bucket_count)
    wanted = defaultdict(lambda: {"panel_index": [], "molecule_hash": [], "canonical_smiles": []})
    matches = [None] * len(panel)
    for index, row in enumerate(panel):
        bucket = int(stable_u64(row["canonical_smiles"]) % bucket_count)
        target = wanted[bucket]
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
                SELECT w.panel_index, w.molecule_hash, w.canonical_smiles,
                       d.canonical_smiles, d.split
                FROM read_parquet(?) AS d
                INNER JOIN wanted AS w USING (molecule_hash)
                """,
                [str(parquet_by_bucket[bucket])],
            ).fetchall()
            connection.unregister("wanted")
            for row in joined:
                index = int(row[0])
                if matches[index] is not None:
                    raise RuntimeError("Pretraining corpus returned a duplicate identity")
                if str(row[2]) != str(row[3]):
                    raise RuntimeError("SHA-256 collision/canonical mismatch in corpus join")
                matches[index] = {
                    "molecule_hash": str(row[1]),
                    "split": str(row[4]),
                }
    finally:
        connection.close()
    return matches


def summary_row(scope, target_id, label, rows, match_by_hash, seen):
    identities = [row["molecule_hash"] for row in rows]
    corpus = [identity for identity in identities if match_by_hash.get(identity) is not None]
    split_counts = {
        split: sum(
            match_by_hash.get(identity) is not None
            and match_by_hash[identity]["split"] == split
            for identity in identities
        )
        for split in ("train", "validation", "test")
    }
    seen_count = sum(identity in seen for identity in identities)
    return {
        "scope": scope,
        "target_id": target_id,
        "label": label,
        "memberships": len(rows),
        "unique_molecules": len(set(identities)),
        "pretraining_corpus_overlap": len(corpus),
        "pretraining_corpus_overlap_percent": 100.0 * len(corpus) / max(1, len(rows)),
        "pretraining_train_overlap": split_counts["train"],
        "pretraining_validation_overlap": split_counts["validation"],
        "pretraining_test_overlap": split_counts["test"],
        "seen_before_step_10000": seen_count,
        "seen_before_step_10000_percent": 100.0 * seen_count / max(1, len(rows)),
    }


def main() -> None:
    protocol = load_protocol()
    population = json.loads(
        (BENCHMARK_DIR / "state/POPULATION_FROZEN.json").read_text(encoding="utf-8")
    )
    panel_path = BENCHMARK_DIR / "inputs/prepared/common_panel.tsv"
    membership_path = BENCHMARK_DIR / "inputs/prepared/common_memberships.tsv"
    if population.get("common_panel_sha256") != sha256_file(panel_path):
        raise RuntimeError("Frozen exposure panel changed")
    if population.get("common_memberships_sha256") != sha256_file(membership_path):
        raise RuntimeError("Frozen exposure memberships changed")
    panel = read_panel_tsv(panel_path)
    memberships = read_tsv(membership_path)
    cfg = load_config(REPOSITORY_ROOT / protocol["gmolai"]["config"]["path"])
    apply_training_plan(
        cfg, REPOSITORY_ROOT / protocol["gmolai"]["training_plan"]["path"]
    )
    checkpoint_path = REPOSITORY_ROOT / protocol["gmolai"]["checkpoint"]["path"]
    cfg["paths"]["run_dir"] = str(checkpoint_path.parent.parent)
    artifacts = protocol["exposure_audit"]["corpus_artifacts"]
    dataset_manifest_path = REPOSITORY_ROOT / artifacts["dataset_manifest"]["path"]
    graph_manifest_path = REPOSITORY_ROOT / artifacts["graph_manifest"]["path"]
    if sha256_file(dataset_manifest_path) != artifacts["dataset_manifest"]["sha256"]:
        raise RuntimeError("Frozen dataset manifest changed")
    if sha256_file(graph_manifest_path) != artifacts["graph_manifest"]["sha256"]:
        raise RuntimeError("Frozen graph manifest changed")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    graph_manifest = json.loads(graph_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError("Dataset manifest configuration differs")
    if graph_manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError("Graph manifest configuration differs")
    if graph_manifest["dataset_manifest_hash"] != dataset_manifest["manifest_hash"]:
        raise RuntimeError("Graph/dataset manifest identity differs")
    matches = join_corpus(cfg, dataset_manifest, panel)
    match_by_hash = {
        row["molecule_hash"]: match for row, match in zip(panel, matches)
    }
    train_hashes = {
        identity
        for identity, match in match_by_hash.items()
        if match is not None and match["split"] == "train"
    }
    train_shards = [
        entry for entry in graph_manifest["shards"] if entry["split"] == "train"
    ]
    checkpoints = _checkpoint_records(
        cfg,
        graph_manifest=graph_manifest,
        train_shards=train_shards,
        checkpoint_names=[str(checkpoint_path)],
    )
    checkpoint = checkpoints[0]
    training_seed = int(cfg.get("training", {}).get("seed", cfg["seed"]))
    locations, scan = _scan_target_locations(
        train_shards,
        target_hashes=train_hashes,
        seed=training_seed,
        world_size=int(checkpoint["world_size"]),
        workers=min(8, os.cpu_count() or 1),
    )
    seen = {
        molecule_hash
        for molecule_hash, location in locations.items()
        if _seen_at_cycle_zero_cursor(
            location, checkpoint["data_states"][int(location["rank"])]
        )
    }
    ledger_rows = []
    for row, match in zip(panel, matches):
        identity = row["molecule_hash"]
        location = locations.get(identity)
        ledger_rows.append(
            {
                "panel_index": row["panel_index"],
                "molecule_hash": identity,
                "canonical_smiles": row["canonical_smiles"],
                "pretraining_corpus_overlap": match is not None,
                "pretraining_split": "" if match is None else match["split"],
                "seen_before_step_10000": identity in seen,
                "rank": "" if location is None else location["rank"],
                "manifest_train_shard_index": ""
                if location is None
                else location["manifest_train_shard_index"],
                "graph_index_in_shard": ""
                if location is None
                else location["graph_index_in_shard"],
            }
        )
    ledger_path = BENCHMARK_DIR / "audits/gmolai_pretraining_exposure_ledger.csv"
    write_csv(ledger_path, ledger_rows, tuple(ledger_rows[0]))

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in memberships:
        groups[(row["target_id"], row["label"])].append(row)
    summary_rows = []
    for label in ("active", "inactive_or_lower_affinity"):
        label_rows = [row for row in memberships if row["label"] == label]
        summary_rows.append(
            summary_row("overall", "ALL", label, label_rows, match_by_hash, seen)
        )
    for (target_id, label), rows in sorted(groups.items()):
        summary_rows.append(
            summary_row("target", target_id, label, rows, match_by_hash, seen)
        )
    summary_path = BENCHMARK_DIR / "results/tables/pretraining_exposure.csv"
    write_csv(summary_path, summary_rows, tuple(summary_rows[0]))

    competitor_rows = []
    for model in protocol["models"]["primary_order"]:
        competitor_rows.append(
            {
                "model": model,
                "exact_molecule_level_exposure_known": model == "gmolai",
                "status": (
                    "exact canonical corpus, split, and step-10000 presentation audit complete"
                    if model == "gmolai"
                    else "published checkpoint/corpus provenance only; exact molecule manifest unavailable"
                ),
                "interpretation": "descriptive only; no unseen/OOD performance claim",
            }
        )
    competitor_path = BENCHMARK_DIR / "results/tables/competitor_exposure_status.csv"
    write_csv(competitor_path, competitor_rows, tuple(competitor_rows[0]))
    result = {
        "schema_version": 1,
        "status": "ok",
        "audit": "exact canonical corpus overlap, assigned split, and graph presentation before step 10000",
        "pretrained_model_executed": False,
        "training_permitted": False,
        "checkpoint": checkpoint,
        "training_stream_seed": training_seed,
        "corpus": {
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
            "graph_manifest": str(graph_manifest_path),
            "graph_manifest_sha256": sha256_file(graph_manifest_path),
            "graph_manifest_hash": graph_manifest["graph_manifest_hash"],
            "graphs_total": int(graph_manifest["counts"]["graphs_total"]),
            "graphs_train": int(graph_manifest["counts"]["graphs_train"]),
        },
        "unique_panel_molecules": len(panel),
        "target_memberships": len(memberships),
        "unique_corpus_overlap": sum(match is not None for match in matches),
        "unique_train_overlap": len(train_hashes),
        "unique_seen_before_step_10000": len(seen),
        "scan": scan,
        "summary_table": str(summary_path),
        "summary_table_sha256": sha256_file(summary_path),
        "identity_ledger": str(ledger_path),
        "identity_ledger_sha256": sha256_file(ledger_path),
        "competitor_status_table": str(competitor_path),
        "competitor_status_table_sha256": sha256_file(competitor_path),
        "accepted_identity_set_sha256": _identity_digest(
            [row["molecule_hash"] for row in panel]
        ),
        "corpus_overlap_identity_set_sha256": _identity_digest(
            sorted(identity for identity, match in match_by_hash.items() if match is not None)
        ),
        "seen_identity_set_sha256": _identity_digest(sorted(seen)),
        "no_unseen_or_ood_claim": True,
        "runtime": runtime_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "audits/pretraining_exposure.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

