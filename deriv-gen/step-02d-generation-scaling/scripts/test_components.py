#!/usr/bin/env python3
"""Small integrity and chemistry tests for the preregistered Step-2d pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from common import (
    STEP1B_ROOT,
    STEP2C_ROOT,
    STEP_ROOT,
    atomic_write_json,
    load_decoder,
    protocol,
    resolve_manifest_inputs,
    utc_now,
)
from generation_core import beam_order, proportional_merge


for source in (STEP1B_ROOT / "scripts", STEP2C_ROOT / "scripts"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from audit_core import mmp_explanations  # noqa: E402
from scaled_common import fragment_molecules  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    root = args.step_root.resolve()
    state_path = root / "state" / "COMPONENT_TESTS.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    cfg = protocol(root)
    paths, _ = resolve_manifest_inputs(args.repo_root.resolve(), root)
    model, checkpoint = load_decoder(paths["decoder_checkpoint"], torch.device("cpu"))
    if any(value.requires_grad for value in model.parameters()):
        raise RuntimeError("Decoder was not frozen")
    if int(checkpoint["model_config"]["condition_dimensions"]) != 384:
        raise RuntimeError("Decoder condition dimension changed")

    scores = torch.tensor([[-2.0, -3.0, -4.0]])
    lengths = torch.tensor([[20, 5, 40]])
    if beam_order(scores, lengths, 0.0).tolist() != [[0, 1, 2]]:
        raise RuntimeError("Beam score ordering failed")
    merged = proportional_merge(
        [("beam", 0), ("beam", 1)], [("sample", 0), ("sample", 1)]
    )
    if merged != [("beam", 0), ("sample", 0), ("beam", 1), ("sample", 1)]:
        raise RuntimeError("Hybrid stream merge failed")

    smiles = ["Cc1ccccc1", "CCc1ccccc1"]
    fragments, heavy, stats = fragment_molecules(
        smiles, settings=cfg["mmp"], workers=1, progress_every=10
    )
    pairs = pd.DataFrame(
        {
            "candidate_row_id": [0],
            "query_position": [0],
            "seed_structure_index": [0],
            "candidate_structure_index": [1],
        }
    )
    explanations = mmp_explanations(
        pairs,
        fragments,
        settings=cfg["mmp"],
        threads=1,
        temporary_dir=root / "state" / "component_duckdb_tmp",
    )
    if explanations.empty:
        raise RuntimeError("Known one-cut MMP pair was not detected")
    state = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "frozen_decoder": True,
        "decoder_condition_dimensions": 384,
        "hybrid_merge_test": "passed",
        "beam_order_test": "passed",
        "known_mmp_test": "passed",
        "known_mmp_fragment_rows": len(fragments),
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
