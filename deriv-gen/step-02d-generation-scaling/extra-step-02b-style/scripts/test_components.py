#!/usr/bin/env python3
"""Small deterministic tests for ranking, bootstrap, and path containment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import MetricSpec, paired_bootstrap, select_reranked
from common import atomic_write_json, ensure_inside, require_analysis_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root = require_analysis_root(args.analysis_root)

    candidates = pd.DataFrame(
        {
            "query_position": [0, 0, 0, 1, 1],
            "first_proposal_rank": [10, 20, 20, 25, 60],
            "latent_relative_l2_to_seed_condition": [0.3, 0.2, 0.2, 0.4, 0.1],
            "latent_cosine_to_seed_condition": [0.7, 0.8, 0.9, 0.6, 0.99],
            "canonical_smiles": ["A", "B", "C", "D", "E"],
        }
    )
    selected50 = select_reranked(candidates, 50).set_index("query_position")
    assert selected50.loc[0, "canonical_smiles"] == "C"
    assert selected50.loc[1, "canonical_smiles"] == "D"
    selected100 = select_reranked(candidates, 100).set_index("query_position")
    assert selected100.loc[1, "canonical_smiles"] == "E"

    frame = pd.DataFrame(
        {
            "value": [0.0, 1.0, 1.0, 0.0],
            "numerator": [0.0, 1.0, 1.0, 0.0],
            "denominator": [1.0, 1.0, 1.0, 0.0],
        }
    )
    specs = [
        MetricSpec("mean_value", "mean", column="value"),
        MetricSpec(
            "ratio_value",
            "ratio",
            numerator="numerator",
            denominator="denominator",
        ),
    ]
    first = paired_bootstrap(
        frame, specs, resamples=200, confidence_level=0.95, seed=123
    )
    second = paired_bootstrap(
        frame, specs, resamples=200, confidence_level=0.95, seed=123
    )
    pd.testing.assert_frame_equal(first, second)
    assert np.isclose(first.set_index("metric").loc["mean_value", "estimate"], 0.5)
    assert np.isclose(
        first.set_index("metric").loc["ratio_value", "estimate"], 2.0 / 3.0
    )

    try:
        ensure_inside(root.parent / "forbidden", root)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Path containment accepted an outside path")

    state = {
        "schema_version": 1,
        "status": "passed",
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "tests": [
            "reranking_primary_and_tie_break_order",
            "nested_budget_selection",
            "paired_bootstrap_determinism",
            "ratio_recomputed_within_bootstrap",
            "output_path_containment",
        ],
    }
    atomic_write_json(root / "state" / "COMPONENT_TESTS.json", state, root)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
