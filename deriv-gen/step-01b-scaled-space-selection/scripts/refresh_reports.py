#!/usr/bin/env python3
"""Refresh reports and seals from completed machine-readable study tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reporting import select_control_space, write_reports
from scaled_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    hash_ledger,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01b-scaled-space-selection"),
    )
    args = parser.parse_args()
    step_root = args.step_root.resolve()
    if step_root.name != "step-01b-scaled-space-selection":
        raise RuntimeError(f"Unexpected study root: {step_root}")
    config = json.loads(
        (step_root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    outputs = step_root / "outputs"
    tables = outputs / "tables"
    bootstrap = pd.read_csv(
        tables / "hierarchical_bootstrap_summary.csv"
    )
    paired = pd.read_csv(
        tables / "paired_differences_vs_released_w3.csv"
    )
    retrieval_average = pd.read_parquet(
        outputs / "raw" / "retrieval_per_query.parquet"
    )
    retrieval_summary = pd.read_csv(tables / "retrieval_summary.csv")
    support_thresholds = pd.read_csv(
        tables / "mmp_support_thresholds.csv"
    )
    mining = json.loads(
        (step_root / "state" / "MMP_MINING_COMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    complete_path = step_root / "state" / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete":
        raise RuntimeError("Cannot refresh reports from an incomplete study")

    decision, selection = select_control_space(
        config, bootstrap, retrieval_average, paired
    )
    write_reports(
        step_root=step_root,
        decision=decision,
        decision_table=selection,
        support_thresholds=support_thresholds,
        bootstrap_summary=bootstrap,
        retrieval_summary=retrieval_summary,
        train_rows=int(complete["train_rows"]),
        validation_rows=int(complete["validation_rows"]),
        query_count=int(complete["retrieval_queries"]),
        mining_summary=mining["summary"],
    )
    atomic_write_csv(tables / "space_selection.csv", selection, step_root)
    atomic_write_json(outputs / "space_decision.json", decision, step_root)

    summary_path = outputs / "study_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision"] = decision
    summary["reporting_note"] = (
        "Weight-3 versus weight-1 language uses paired hierarchical confidence "
        "intervals; directional leadership is separated from the final "
        "compatibility-aware selection."
    )
    atomic_write_json(summary_path, summary, step_root)

    ledger_path = outputs / "SHA256SUMS"
    atomic_write_text(
        ledger_path,
        hash_ledger(outputs, exclude={"SHA256SUMS"}),
        step_root,
    )
    complete["selected_edit_control_space"] = decision[
        "selected_edit_control_space"
    ]
    complete["decoder_conditioning_representation"] = decision[
        "decoder_conditioning_representation"
    ]
    complete["results_sha256"] = sha256_file(step_root / "RESULTS.md")
    complete["decision_sha256"] = sha256_file(step_root / "DECISION.md")
    complete["output_ledger_sha256"] = sha256_file(ledger_path)
    atomic_write_json(complete_path, complete, step_root)
    atomic_write_json(
        step_root / "state" / "REPORT_REFRESH.json",
        {
            "schema_version": 1,
            "status": "complete",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "Statistically qualify weight comparisons and distinguish "
                "directional from overall leadership."
            ),
            "results_sha256": complete["results_sha256"],
            "decision_sha256": complete["decision_sha256"],
            "output_ledger_sha256": complete["output_ledger_sha256"],
        },
        step_root,
    )
    print(
        json.dumps(
            {
                "status": "reports-refreshed",
                "selected_edit_control_space": decision[
                    "selected_edit_control_space"
                ],
                "results_sha256": complete["results_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
