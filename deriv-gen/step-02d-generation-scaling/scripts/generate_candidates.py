#!/usr/bin/env python3
"""Generate one deterministic Step-2d GPU shard without chemistry-aware ordering."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import (
    STEP_ROOT,
    atomic_write_json,
    configure_determinism,
    decode_tokens,
    load_decoder,
    load_json,
    protocol,
    require_one_gpu,
    resolve_manifest_inputs,
    sha256_file,
    stable_digest,
    utc_now,
)
from generation_core import (
    beam_order,
    generate_beam_pool,
    generate_seeded_sample_pool,
    proportional_merge,
)


SCHEMA = pa.schema(
    [
        ("phase", pa.string()),
        ("strategy", pa.string()),
        ("query_position", pa.int32()),
        ("target_index", pa.int64()),
        ("target_hash", pa.string()),
        ("proposal_rank", pa.int16()),
        ("source_kind", pa.string()),
        ("source_rank", pa.int16()),
        ("raw_smiles", pa.string()),
        ("token_error", pa.string()),
        ("cumulative_decoder_log_probability", pa.float32()),
        ("generated_length", pa.int16()),
    ]
)


class AtomicParquetWriters:
    def __init__(self, targets: dict[str, Path]) -> None:
        self.targets = targets
        self.temporaries: dict[str, Path] = {}
        self.writers: dict[str, pq.ParquetWriter] = {}
        for name, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(raw)
            self.temporaries[name] = temporary
            self.writers[name] = pq.ParquetWriter(
                temporary, SCHEMA, compression="zstd", use_dictionary=True
            )

    def write(self, name: str, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.writers[name].write_table(pa.Table.from_pylist(rows, schema=SCHEMA))

    def commit(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        for name, temporary in self.temporaries.items():
            os.replace(temporary, self.targets[name])
        self.temporaries.clear()

    def abort(self) -> None:
        for writer in self.writers.values():
            writer.close()
        for temporary in self.temporaries.values():
            temporary.unlink(missing_ok=True)
        self.writers.clear()
        self.temporaries.clear()


def decode_record(
    tokens: np.ndarray, score: float, length: int, source_kind: str, source_rank: int
) -> dict[str, Any]:
    raw, error = decode_tokens(tokens)
    return {
        "source_kind": source_kind,
        "source_rank": int(source_rank),
        "raw_smiles": raw,
        "token_error": error,
        "cumulative_decoder_log_probability": float(score),
        "generated_length": int(length),
    }


def strategies_for_phase(cfg: dict[str, Any], phase: str, root: Path) -> list[dict[str, Any]]:
    if phase == "development":
        if (root / "state" / "STRATEGY_FROZEN.json").exists():
            raise RuntimeError("Development generation cannot start after strategy freeze")
        return list(cfg["generation"]["development_strategies"])
    frozen_path = root / "state" / "STRATEGY_FROZEN.json"
    if not frozen_path.is_file():
        raise RuntimeError("Final generation is forbidden before strategy freeze")
    frozen = load_json(frozen_path)
    if frozen.get("status") != "frozen_before_final_generation":
        raise RuntimeError("Invalid frozen-strategy seal")
    return [dict(frozen["selected_strategy"])]


def sample_seed(global_seed: int, phase: str, pool: str, target_hash: str) -> int:
    return int(stable_digest(global_seed, phase, pool, target_hash)[:16], 16) % (2**63 - 1)


def make_rows(
    phase: str,
    strategy: str,
    query_position: int,
    target_index: int,
    target_hash: str,
    stream: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(stream) != 1000:
        raise RuntimeError(f"Strategy {strategy} yielded {len(stream)} rather than 1000 slots")
    return [
        {
            "phase": phase,
            "strategy": strategy,
            "query_position": int(query_position),
            "target_index": int(target_index),
            "target_hash": str(target_hash),
            "proposal_rank": rank,
            **record,
        }
        for rank, record in enumerate(stream, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    phase = args.phase
    shard = int(args.shard_id)
    shards = int(args.num_shards)
    if not 0 <= shard < shards:
        raise ValueError("Invalid shard identity")
    state_path = root / "state" / f"{phase.upper()}_SHARD_{shard:02d}_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    if not (root / "state" / "PANELS_PREPARED.json").is_file():
        raise RuntimeError("Prepare panels before generation")
    cfg = protocol(root)
    if shards != int(cfg["execution"]["gpu_shards"]):
        raise RuntimeError("Shard count differs from preregistered execution")
    strategies = strategies_for_phase(cfg, phase, root)
    paths, input_hashes = resolve_manifest_inputs(repo_root, root)
    device = require_one_gpu()
    configure_determinism(int(cfg["seed"]) + shard)
    panel_name = "development_panel.csv" if phase == "development" else "fresh_validation_panel.csv"
    condition_name = "development_conditions.npy" if phase == "development" else "final_conditions.npy"
    panel = pd.read_csv(root / "prepared" / panel_name)
    conditions = np.load(root / "prepared" / condition_name, mmap_mode="r")
    if conditions.shape != (len(panel), 384):
        raise RuntimeError("Panel/condition dimensions differ")
    positions = np.arange(shard, len(panel), shards, dtype=np.int64)
    if not len(positions):
        raise RuntimeError("Empty GPU shard")
    selected = panel.iloc[positions].reset_index(drop=True)
    selected_conditions = np.asarray(conditions[positions], dtype=np.float32)
    model, checkpoint = load_decoder(paths["decoder_checkpoint"], device)
    if checkpoint["model_config"]["condition_dimensions"] != 384:
        raise RuntimeError("Frozen decoder condition dimension changed")
    maximum_steps = int(cfg["generation"]["maximum_smiles_bytes"])

    needed_beam = any(value["kind"] in {"beam", "hybrid"} for value in strategies)
    needed_pools = sorted(
        {value["sample_pool"] for value in strategies if value["kind"] in {"sample", "hybrid"}}
    )
    pool_configs = {
        value["name"]: value for value in cfg["generation"]["development_base_sample_pools"]
    }
    targets = {
        value["name"]: root
        / "outputs"
        / "raw"
        / phase
        / f"proposals-{value['name']}-shard-{shard:02d}-of-{shards:02d}.parquet"
        for value in strategies
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise RuntimeError(f"Unsealed output files already exist: {existing[:3]}")
    writers = AtomicParquetWriters(targets)
    started = time.monotonic()
    try:
        sample_batch = int(cfg["generation"]["sample_query_batch_size"])
        loop_batch = (
            min(sample_batch, int(cfg["generation"]["beam_query_batch_size"]))
            if needed_beam
            else sample_batch
        )
        for offset in range(0, len(selected), loop_batch):
            stop = min(offset + loop_batch, len(selected))
            batch_panel = selected.iloc[offset:stop]
            batch_conditions = torch.as_tensor(
                selected_conditions[offset:stop], dtype=torch.float32, device=device
            )
            greedy_tokens = model.generate(
                batch_conditions, maximum_steps=maximum_steps
            ).cpu().numpy()
            greedy = [
                decode_record(
                    greedy_tokens[local], math.nan, len(greedy_tokens[local]), "greedy", 0
                )
                for local in range(stop - offset)
            ]

            beam_by_penalty: dict[float, list[list[dict[str, Any]]]] = {}
            if needed_beam:
                beam_tokens, beam_scores, beam_lengths = generate_beam_pool(
                    model,
                    batch_conditions,
                    maximum_steps=maximum_steps,
                    beam_width=int(cfg["generation"]["beam_width"]),
                )
                beam_tokens_np = beam_tokens.cpu().numpy()
                beam_scores_np = beam_scores.cpu().numpy()
                beam_lengths_np = beam_lengths.cpu().numpy()
                penalties = sorted(
                    {float(value.get("length_penalty", 0.0)) for value in strategies if value["kind"] in {"beam", "hybrid"}}
                )
                for penalty in penalties:
                    order = beam_order(beam_scores, beam_lengths, penalty).cpu().numpy()
                    current_batch = []
                    for local in range(stop - offset):
                        current = []
                        for source_index in order[local]:
                            source_index = int(source_index)
                            current.append(
                                decode_record(
                                    beam_tokens_np[local, source_index],
                                    float(beam_scores_np[local, source_index]),
                                    int(beam_lengths_np[local, source_index]),
                                    "beam",
                                    source_index + 1,
                                )
                            )
                        current_batch.append(current)
                    beam_by_penalty[penalty] = current_batch
                del beam_tokens, beam_scores, beam_lengths

            samples: dict[str, list[list[dict[str, Any]]]] = {}
            hashes = batch_panel["target_hash"].astype(str).tolist()
            for pool_name in needed_pools:
                pool = pool_configs[pool_name]
                seeds = [sample_seed(int(cfg["seed"]), phase, pool_name, value) for value in hashes]
                sample_tokens, sample_scores, sample_lengths = generate_seeded_sample_pool(
                    model,
                    batch_conditions,
                    maximum_steps=maximum_steps,
                    draws=int(pool["draws"]),
                    temperature=float(pool["temperature"]),
                    top_p=float(pool["top_p"]),
                    seeds=seeds,
                )
                token_np = sample_tokens.cpu().numpy()
                score_np = sample_scores.cpu().numpy()
                length_np = sample_lengths.cpu().numpy()
                current_batch = []
                for local in range(stop - offset):
                    current_batch.append(
                        [
                            decode_record(
                                token_np[local, draw],
                                float(score_np[local, draw]),
                                int(length_np[local, draw]),
                                "sample",
                                draw + 1,
                            )
                            for draw in range(int(pool["draws"]))
                        ]
                    )
                samples[pool_name] = current_batch
                del sample_tokens, sample_scores, sample_lengths

            for local, row in enumerate(batch_panel.itertuples(index=False)):
                for strategy in strategies:
                    if strategy["kind"] == "beam":
                        stream = beam_by_penalty[float(strategy["length_penalty"])][local]
                    elif strategy["kind"] == "sample":
                        stream = [greedy[local], *samples[strategy["sample_pool"]][local]]
                    elif strategy["kind"] == "hybrid":
                        beam_stream = beam_by_penalty[float(strategy["length_penalty"])][local]
                        sample_stream = [greedy[local], *samples[strategy["sample_pool"]][local]]
                        sources = proportional_merge(
                            [("beam", index) for index in range(int(strategy["beam_hypotheses"]))],
                            [("sample", index) for index in range(int(strategy["sample_hypotheses"]))],
                        )
                        stream = [
                            beam_stream[index] if kind == "beam" else sample_stream[index]
                            for kind, index in sources
                        ]
                    else:
                        raise RuntimeError(f"Unknown strategy kind: {strategy['kind']}")
                    writers.write(
                        strategy["name"],
                        make_rows(
                            phase,
                            strategy["name"],
                            int(row.query_position),
                            int(row.target_index),
                            str(row.target_hash),
                            stream,
                        ),
                    )
            del batch_conditions, greedy_tokens, beam_by_penalty, samples
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"{phase} shard {shard}: {stop:,}/{len(selected):,} seeds "
                f"({stop / elapsed:.3f} seeds/s)",
                flush=True,
            )
        writers.commit()
    except BaseException:
        writers.abort()
        raise

    output_hashes = {name: sha256_file(path) for name, path in targets.items()}
    state = {
        "schema_version": 1,
        "status": "complete",
        "phase": phase,
        "shard_id": shard,
        "num_shards": shards,
        "seed_rows": len(selected),
        "strategies": [value["name"] for value in strategies],
        "proposals_per_strategy_seed": 1000,
        "output_sha256": output_hashes,
        "decoder_checkpoint_sha256": input_hashes["decoder_checkpoint"],
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "visible_gpu": torch.cuda.get_device_name(0),
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
