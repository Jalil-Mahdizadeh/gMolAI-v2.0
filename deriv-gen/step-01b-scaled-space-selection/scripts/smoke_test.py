#!/usr/bin/env python3
"""Small deterministic smoke test for scaled-study analysis primitives."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from analysis_core import (
    SPACE_ORDER,
    add_observation_ids,
    assign_mismatched_transforms,
    evaluate_alignment,
    fit_directions,
    hierarchical_bootstrap,
    select_queries,
    space_matrix,
)
from scaled_common import topk_l2


def main() -> None:
    rng = np.random.default_rng(17)
    train = rng.normal(size=(96, 384)).astype(np.float32)
    validation = rng.normal(size=(48, 384)).astype(np.float32)
    train_rows = []
    validation_rows = []
    transforms = ["a>>b", "c>>d", "e>>f"]
    for transform_position, transform in enumerate(transforms):
        direction = rng.normal(size=384).astype(np.float32)
        direction /= np.linalg.norm(direction)
        for core_position in range(8):
            lhs = transform_position * 16 + 2 * core_position
            rhs = lhs + 1
            train[rhs] = train[lhs] + (1.0 + 0.05 * core_position) * direction
            train_rows.append(
                {
                    "core": f"train-core-{transform}-{core_position}",
                    "transform": transform,
                    "lhs_index": lhs,
                    "rhs_index": rhs,
                }
            )
        for core_position in range(4):
            lhs = transform_position * 8 + 2 * core_position
            rhs = lhs + 1
            validation[rhs] = validation[lhs] + (
                1.1 + 0.03 * core_position
            ) * direction
            validation_rows.append(
                {
                    "core": f"validation-core-{transform}-{core_position}",
                    "transform": transform,
                    "lhs_substituent": transform.split(">>")[0],
                    "rhs_substituent": transform.split(">>")[1],
                    "lhs_index": lhs,
                    "rhs_index": rhs,
                    "train_cores": 8,
                    "support_tier": "5-9",
                }
            )
    train_frame = pd.DataFrame(train_rows)
    eligible = add_observation_ids(pd.DataFrame(validation_rows), 17)
    support = {value: 8 for value in transforms}
    all_directions = {}
    alignments = []
    for space in SPACE_ORDER:
        directions = fit_directions(
            train_frame, space_matrix(train, space), minimum_cores=2
        )
        if set(directions) != set(transforms):
            raise RuntimeError(f"Direction smoke test failed in {space}")
        all_directions[space] = directions
    eligible = assign_mismatched_transforms(
        eligible, set(transforms), support, 17
    )
    queries = select_queries(
        eligible,
        set(transforms),
        maximum=12,
        per_transform=4,
        primary_support=5,
        seed=17,
    )
    if len(queries) != 12:
        raise RuntimeError("Query-selection smoke test failed")
    for space in SPACE_ORDER:
        alignments.append(
            evaluate_alignment(
                eligible,
                space_matrix(validation, space),
                all_directions[space],
                space=space,
            )
        )
    alignment = pd.concat(alignments, ignore_index=True)
    summary, differences = hierarchical_bootstrap(
        alignment,
        metrics=("alignment", "null_alignment", "alignment_gain"),
        analysis="alignment_all",
        cohorts=(2, 5),
        resamples=20,
        alpha=0.05,
        seed=17,
    )
    if summary.empty or differences.empty:
        raise RuntimeError("Hierarchical-bootstrap smoke test failed")

    device = torch.device("cuda:0")
    indices, distances = topk_l2(
        validation[:4],
        validation,
        k=3,
        device=device,
        batch_size=2,
    )
    if indices.shape != (4, 3) or distances.shape != (4, 3):
        raise RuntimeError("GPU top-k smoke test failed")

    connection = duckdb.connect(":memory:")
    temporary = Path("/tmp/gmolai-scaled-duckdb-smoke")
    temporary.mkdir(parents=True, exist_ok=True)
    connection.execute(
        "SET temp_directory='"
        + str(temporary).replace("'", "''")
        + "'"
    )
    connection.execute("SELECT 1").fetchone()
    connection.close()
    print(
        {
            "status": "smoke-ok",
            "spaces": len(SPACE_ORDER),
            "queries": len(queries),
            "bootstrap_rows": len(summary),
            "gpu": torch.cuda.get_device_name(0),
        }
    )


if __name__ == "__main__":
    main()
