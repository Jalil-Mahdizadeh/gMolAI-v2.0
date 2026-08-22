#!/usr/bin/env python3
"""Materialize model-specific panels from label-blind adapter screens."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    columns_of,
    load_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
    write_csv,
    write_tsv,
)


def main() -> None:
    protocol = load_protocol()
    models = tuple(protocol["models"]["primary_order"])
    source = BENCHMARK_DIR / "inputs/prepared/molecule_candidates.tsv"
    rows = read_panel_tsv(source)
    source_hash = sha256_file(source)
    columns = columns_of(source)
    panel_dir = BENCHMARK_DIR / "inputs/prepared/model_panels"
    coverage_rows = []
    rejection_rows = []
    panels = {}
    for model in models:
        screen_path = BENCHMARK_DIR / "artifacts/screens" / f"{model}.json"
        report = load_json(screen_path)
        if report.get("status") != "ok" or report.get("model") != model:
            raise RuntimeError(f"Invalid label-blind screen report: {screen_path}")
        if report.get("input_sha256") != source_hash or int(report.get("rows", -1)) != len(rows):
            raise RuntimeError(f"Screen input binding differs: {screen_path}")
        accepted = [int(value) for value in report.get("accepted_indices", [])]
        accepted_set = set(accepted)
        if (
            len(accepted) != len(accepted_set)
            or any(index < 0 or index >= len(rows) for index in accepted)
            or len(accepted) != int(report.get("accepted", -1))
        ):
            raise RuntimeError(f"Invalid accepted index set in {screen_path}")
        selected = [
            dict(rows[index], panel_index=output_index)
            for output_index, index in enumerate(accepted)
        ]
        panel_path = panel_dir / f"{model}.tsv"
        write_tsv(panel_path, selected, columns)
        if report.get("accepted_identity_sha256") != sha256_lines(
            row["molecule_hash"] for row in selected
        ):
            raise RuntimeError(f"Accepted identity digest differs for {model}")
        coverage_rows.append(
            {
                "model": model,
                "attempted_unique_molecules": len(rows),
                "screen_accepted": len(selected),
                "screen_rejected": len(rows) - len(selected),
                "screen_coverage_fraction": len(selected) / max(1, len(rows)),
                "accepted_identity_sha256": report["accepted_identity_sha256"],
                "screen_report_sha256": sha256_file(screen_path),
                "model_panel_sha256": sha256_file(panel_path),
            }
        )
        counts = Counter(str(item.get("reason", "unspecified")) for item in report.get("rejections", []))
        for reason, count in sorted(counts.items()):
            rejection_rows.append({"model": model, "reason": reason, "count": count})
        panels[model] = {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "rows": len(selected),
            "identity_sha256": report["accepted_identity_sha256"],
        }
    coverage_path = BENCHMARK_DIR / "audits" / "label_blind_screen_coverage.csv"
    rejection_path = BENCHMARK_DIR / "audits" / "label_blind_screen_rejections.csv"
    write_csv(coverage_path, coverage_rows, tuple(coverage_rows[0]))
    write_csv(rejection_path, rejection_rows, ("model", "reason", "count"))
    result = {
        "schema_version": 1,
        "status": "ok",
        "execution": "label_blind_before_representation_forward_pass",
        "source_panel": str(source),
        "source_panel_sha256": source_hash,
        "source_rows": len(rows),
        "panels": panels,
        "coverage_table": str(coverage_path),
        "coverage_table_sha256": sha256_file(coverage_path),
        "rejection_table": str(rejection_path),
        "rejection_table_sha256": sha256_file(rejection_path),
        "labels_read": False,
        "performance_inspected": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "SCREENING_COMPLETE.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

