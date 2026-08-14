#!/usr/bin/env python3
"""Small frozen-artifact smoke tests for Step-2b generation primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from candidate_model import generate_beam_pool, generate_sample_pool
from common import (
    atomic_write_json,
    configure_determinism,
    load_decoder,
    load_json,
    released_train_rows,
    require_one_gpu,
    sha256_file,
    utc_now,
    validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    output = root / "state" / "COMPONENT_TESTS.json"
    if output.exists():
        print(output.read_text(encoding="utf-8"))
        return
    paths, hashes = validate_manifest(repo_root, root)
    panel = pd.read_csv(root / "prepared" / "development_panel.csv").iloc[:2]
    indices = panel["correct_source_index"].to_numpy(dtype=np.int64)
    conditions = released_train_rows(
        paths["train_raw_embeddings"], paths["gmolai_calibrator"], indices
    )
    device = require_one_gpu()
    configure_determinism(20260815)
    before = sha256_file(paths["decoder_checkpoint"])
    model, _ = load_decoder(paths["decoder_checkpoint"], device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen decoder still has trainable parameters")
    condition = torch.as_tensor(conditions, dtype=torch.float32, device=device)
    beam_tokens, beam_scores, beam_lengths = generate_beam_pool(
        model, condition, maximum_steps=128, beam_width=4
    )
    first = generate_sample_pool(
        model,
        condition,
        maximum_steps=128,
        draws=4,
        temperature=0.85,
        top_p=0.95,
        seed=712345,
    )
    second = generate_sample_pool(
        model,
        condition,
        maximum_steps=128,
        draws=4,
        temperature=0.85,
        top_p=0.95,
        seed=712345,
    )
    deterministic_sampling = all(
        torch.equal(left, right) for left, right in zip(first, second)
    )
    if not deterministic_sampling:
        raise RuntimeError("Fixed-seed sampling is not deterministic")
    if beam_tokens.shape != (2, 4, 128):
        raise RuntimeError("Unexpected beam output shape")
    if beam_scores.shape != (2, 4) or beam_lengths.shape != (2, 4):
        raise RuntimeError("Unexpected beam metadata shape")
    del model, condition
    torch.cuda.empty_cache()
    after = sha256_file(paths["decoder_checkpoint"])
    if before != after or before != hashes["decoder_checkpoint"]:
        raise RuntimeError("Decoder checkpoint changed during smoke test")
    result = {
        "schema_version": 1,
        "status": "pass",
        "tested_at": utc_now(),
        "beam_shape": list(beam_tokens.shape),
        "sample_shape": list(first[0].shape),
        "fixed_seed_sampling_deterministic": deterministic_sampling,
        "decoder_parameters_trainable": 0,
        "decoder_checkpoint_sha256_before": before,
        "decoder_checkpoint_sha256_after": after,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(output, result, root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
