#!/usr/bin/env python3
"""Create an identical all-comparator panel from frozen coverage screens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
    write_panel_tsv,
)


SCREENED_MODELS = ("morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--coverage-dir", default=BENCHMARK_DIR / "state", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    rows = read_panel_tsv(args.panel)
    input_hash = sha256_file(args.panel)
    accepted_by_model: dict[str, set[int]] = {
        "gmolai": set(range(len(rows)))
    }
    reports: dict[str, dict] = {}
    for model in SCREENED_MODELS:
        path = args.coverage_dir / f"{model}-{args.name}-screen.json"
        report = load_json(path)
        if report.get("status") != "ok" or report.get("model") != model:
            raise RuntimeError(f"Invalid coverage report: {path}")
        if int(report.get("rows", -1)) != len(rows):
            raise RuntimeError(f"Coverage row count differs in {path}")
        if report.get("input_sha256") != input_hash:
            raise RuntimeError(f"Coverage input hash differs in {path}")
        indices = [int(value) for value in report.get("accepted_indices", [])]
        accepted = set(indices)
        if len(indices) != len(accepted) or any(
            index < 0 or index >= len(rows) for index in accepted
        ):
            raise RuntimeError(f"Invalid accepted indices in {path}")
        if len(accepted) != int(report.get("accepted", -1)):
            raise RuntimeError(f"Accepted count differs in {path}")
        accepted_by_model[model] = accepted
        reports[model] = report

    common = set.intersection(*accepted_by_model.values())
    common_indices = sorted(common)
    if len(common_indices) < 2:
        raise RuntimeError("All-comparator common panel has fewer than two molecules")
    common_rows = [
        {**rows[source_index], "panel_index": common_index}
        for common_index, source_index in enumerate(common_indices)
    ]
    common_path = BENCHMARK_DIR / "inputs" / f"common_{args.name}.tsv"
    common_smiles_path = BENCHMARK_DIR / "inputs" / f"common_{args.name}.smi"
    write_panel_tsv(common_path, common_rows)
    atomic_write_text(
        common_smiles_path,
        "".join(f"{row['canonical_smiles']}\n" for row in common_rows),
    )

    coverage_rows = []
    for model in ("gmolai", *SCREENED_MODELS):
        accepted_count = len(accepted_by_model[model])
        coverage_rows.append(
            {
                "model": model,
                "attempted": len(rows),
                "accepted": accepted_count,
                "rejected": len(rows) - accepted_count,
                "coverage_fraction": accepted_count / len(rows),
                "common_rows": len(common_indices),
            }
        )
    csv_path = BENCHMARK_DIR / "outputs" / f"coverage_{args.name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)

    result = {
        "schema_version": 1,
        "status": "ok",
        "name": args.name,
        "source_panel": str(args.panel.resolve().relative_to(REPOSITORY_ROOT)),
        "source_panel_sha256": input_hash,
        "attempted_rows": len(rows),
        "common_rows": len(common_indices),
        "common_fraction": len(common_indices) / len(rows),
        "common_source_indices": common_indices,
        "common_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in common_rows
        ),
        "common_panel": str(common_path.relative_to(REPOSITORY_ROOT)),
        "common_panel_sha256": sha256_file(common_path),
        "common_smiles_sha256": sha256_file(common_smiles_path),
        "coverage_csv": str(csv_path.relative_to(REPOSITORY_ROOT)),
        "coverage_csv_sha256": sha256_file(csv_path),
        "models": {
            row["model"]: {
                "attempted": row["attempted"],
                "accepted": row["accepted"],
                "rejected": row["rejected"],
                "coverage_fraction": row["coverage_fraction"],
            }
            for row in coverage_rows
        },
        "protocol_comparators": sorted(protocol["comparators"]),
    }
    output = BENCHMARK_DIR / "state" / f"common_{args.name}.json"
    atomic_write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "common_source_indices"}, sort_keys=True))


if __name__ == "__main__":
    main()
