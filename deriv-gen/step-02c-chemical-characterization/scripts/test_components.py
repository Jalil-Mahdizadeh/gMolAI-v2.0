#!/usr/bin/env python3
"""Small deterministic checks for Step 2c chemistry primitives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit_core import audit_raw_smiles, mmp_explanations
from common import STEP_ROOT, atomic_write_json, load_json, protocol, resolve_manifest_inputs


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
STEP1B_SCRIPTS = REPO_ROOT / "deriv-gen" / "step-01b-scaled-space-selection" / "scripts"
if str(STEP1B_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP1B_SCRIPTS))

from scaled_common import fragment_molecules  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    cfg = protocol(step_root)
    paths, _ = resolve_manifest_inputs(repo_root, step_root)

    resolved = load_json(paths["external_resolved_config"])
    policy = audit_raw_smiles(
        ["CC", "CCc1ccccc1", "Cc1ccccc1"],
        resolved_config=resolved,
        workers=1,
    )
    if not policy["rdkit_valid"].all() or not policy["policy_accepted"].all():
        raise RuntimeError("Synthetic valid molecules failed the frozen policy")
    if not policy.loc[policy["raw_smiles"] == "CC", "raw_equals_canonical"].item():
        raise RuntimeError("Canonical spelling test failed")

    smiles = ["Cc1ccccc1", "CCc1ccccc1"]
    fragments, heavy, stats = fragment_molecules(
        smiles, settings=cfg["mmp"], workers=1, progress_every=10
    )
    if heavy.tolist() != [7, 8] or stats["parse_failures"] != 0:
        raise RuntimeError("Step-1b fragmentation descriptor test failed")
    pairs = pd.DataFrame(
        {
            "candidate_row_id": np.asarray([0], dtype=np.int64),
            "query_position": np.asarray([0], dtype=np.int64),
            "seed_structure_index": np.asarray([0], dtype=np.int64),
            "candidate_structure_index": np.asarray([1], dtype=np.int64),
        }
    )
    explanations = mmp_explanations(
        pairs,
        fragments,
        settings=cfg["mmp"],
        threads=1,
        temporary_dir=step_root / "state" / "component_test_duckdb_tmp",
    )
    primary = explanations.loc[explanations["is_primary_explanation"].astype(bool)]
    if len(primary) != 1:
        raise RuntimeError("Known toluene-to-ethylbenzene MMP was not recovered")
    row = primary.iloc[0]
    if int(row["variable_heavy_atom_delta"]) != 1:
        raise RuntimeError("Known MMP variable-size delta is incorrect")
    if row["mmp_edit_class"] != "substituent_growth":
        raise RuntimeError("Known MMP edit class is incorrect")

    seal = {
        "schema_version": 1,
        "status": "passed",
        "tests": {
            "frozen_policy_accepts_known_valid_molecules": True,
            "canonical_spelling_identified": True,
            "step1b_one_cut_fragmentation_imported": True,
            "known_one_cut_mmp_recovered": True,
            "directional_variable_delta_recovered": True,
        },
        "known_mmp_transform": str(row["seed_to_candidate_transform"]),
    }
    atomic_write_json(step_root / "state" / "COMPONENT_TESTS.json", seal, step_root)
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
