#!/usr/bin/env python3
"""Render a compact, manuscript-facing Markdown summary from frozen tables."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    read_csv,
    sha256_file,
)


METRIC_LABELS = {
    "ef1": "EF1%",
    "bedroc20": "BEDROC (alpha=20)",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
}


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def interval(row: dict[str, str]) -> str:
    return (
        f"{number(row['target_level_mean'])} "
        f"[{number(row['target_level_mean_ci95_lower'])}, "
        f"{number(row['target_level_mean_ci95_upper'])}]"
    )


def require_bound(state: dict, key: str, path) -> None:
    if state.get(key) != sha256_file(path):
        raise RuntimeError(f"Frozen report input changed: {path}")


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    protocol = load_protocol()
    population = load_json(BENCHMARK_DIR / "state/POPULATION_FROZEN.json")
    anchors = load_json(BENCHMARK_DIR / "state/ANCHORS_FROZEN.json")
    summary = load_json(BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json")
    exposure = load_json(BENCHMARK_DIR / "audits/pretraining_exposure.json")
    figure_manifest = load_json(BENCHMARK_DIR / "audits/figure_manifest.json")
    if any(value.get("status") not in {"ok", "frozen"} for value in (population, anchors, summary, exposure, figure_manifest)):
        raise RuntimeError("All frozen analyses must finish before result rendering")

    summary_path = BENCHMARK_DIR / "results/tables/model_summary.csv"
    paired_path = BENCHMARK_DIR / "results/tables/paired_comparisons.csv"
    coverage_path = BENCHMARK_DIR / "results/tables/model_coverage.csv"
    exposure_path = BENCHMARK_DIR / "results/tables/pretraining_exposure.csv"
    require_bound(summary, "model_summary_sha256", summary_path)
    require_bound(summary, "paired_comparisons_sha256", paired_path)
    require_bound(population, "model_coverage_sha256", coverage_path)
    require_bound(exposure, "summary_table_sha256", exposure_path)

    models = list(protocol["models"]["primary_order"])
    display = {model: protocol["models"][model]["display_name"] for model in models}
    primary_shots = int(protocol["retrieval"]["primary_shots"])
    secondary_shots = int(protocol["retrieval"]["secondary_shots"])
    summaries = read_csv(summary_path)

    def selected(shots: int, condition: str, metric: str) -> dict[str, dict[str, str]]:
        rows = {
            row["model"]: row
            for row in summaries
            if int(row["shots"]) == shots
            and row["condition"] == condition
            and row["metric"] == metric
            and row["model"] in models
        }
        if set(rows) != set(models):
            raise RuntimeError(f"Incomplete model summary for {shots}/{condition}/{metric}")
        return rows

    primary_by_metric = {
        metric: selected(primary_shots, "standard", metric)
        for metric in METRIC_LABELS
    }
    scaffold = selected(primary_shots, "scaffold_excluded", "ef1")
    one_shot = selected(secondary_shots, "standard", "ef1")
    primary_table = []
    for model in models:
        primary_table.append(
            [
                display[model],
                *(interval(primary_by_metric[metric][model]) for metric in METRIC_LABELS),
            ]
        )

    paired = [
        row
        for row in read_csv(paired_path)
        if int(row["shots"]) == primary_shots
        and row["condition"] == "standard"
        and row["metric"] == "ef1"
    ]
    paired_by_model = {row["comparator_model"]: row for row in paired}
    if set(paired_by_model) != set(models) - {"gmolai"}:
        raise RuntimeError("Incomplete primary paired comparison table")
    paired_table = []
    for model in models:
        if model == "gmolai":
            continue
        row = paired_by_model[model]
        paired_table.append(
            [
                display[model],
                f"{number(row['paired_mean_difference'])} "
                f"[{number(row['paired_mean_difference_ci95_lower'])}, "
                f"{number(row['paired_mean_difference_ci95_upper'])}]",
                f"{row['strict_wins']}/{row['strict_losses']}/{row['exact_ties']}",
            ]
        )

    sensitivity_table = [
        [
            display[model],
            interval(one_shot[model]),
            interval(scaffold[model]),
            scaffold[model]["targets"],
        ]
        for model in models
    ]
    coverage = {row["model"]: row for row in read_csv(coverage_path)}
    coverage_table = [
        [
            display[model],
            coverage[model]["validated_representation_rows"],
            coverage[model]["rejected_before_common_support"],
            number(100.0 * float(coverage[model]["validated_coverage_fraction"]), 2),
        ]
        for model in models
    ]
    exposure_overall = {
        row["label"]: row
        for row in read_csv(exposure_path)
        if row["scope"] == "overall"
    }
    exposure_table = []
    for label in ("active", "inactive_or_lower_affinity"):
        row = exposure_overall[label]
        exposure_table.append(
            [
                label.replace("_", " "),
                row["memberships"],
                row["pretraining_corpus_overlap"],
                number(row["pretraining_corpus_overlap_percent"], 2),
                row["seen_before_step_10000"],
                number(row["seen_before_step_10000_percent"], 2),
            ]
        )

    lines = [
        "# VSDS-vd TrueDecoy_gap ligand-retrieval results",
        "",
        "This report is generated directly from frozen result tables. Values are target-level means across deterministic anchor draws, followed by target-stratified bootstrap 95% confidence intervals. Anchor draws are not treated as independent inferential replicates; no formal p-values were performed.",
        "",
        "## Frozen study population",
        "",
        f"The all-seven intersection retained **{population['all_seven_common_before_target_eligibility']:,}** unique molecules before target eligibility and **{population['final_unique_molecules']:,}** unique molecules across **{population['eligible_targets']}** eligible protein targets. The primary analysis used {primary_shots} active anchors and {protocol['retrieval']['draws_per_target']} deterministic draws per target.",
        "",
        *markdown_table(
            ["Model", "Validated", "Rejected", "Coverage (%)"], coverage_table
        ),
        "",
        "## Primary five-shot retrieval",
        "",
        *markdown_table(
            ["Model", *METRIC_LABELS.values()], primary_table
        ),
        "",
        "## Paired primary EF1%: gMolAI minus comparator",
        "",
        *markdown_table(
            ["Comparator", "Mean difference [95% CI]", "Wins/losses/ties"],
            paired_table,
        ),
        "",
        "## Prespecified sensitivity analyses",
        "",
        *markdown_table(
            ["Model", "One-shot EF1% [95% CI]", "Scaffold-excluded EF1% [95% CI]", "Scaffold targets"],
            sensitivity_table,
        ),
        "",
        "## gMolAI pretraining exposure (descriptive only)",
        "",
        *markdown_table(
            ["Label", "Memberships", "Corpus overlap", "Overlap (%)", "Seen by step 10,000", "Seen (%)"],
            exposure_table,
        ),
        "",
        "Exact molecule-level exposure was auditable only for gMolAI. These counts are descriptive and do not support an unseen-molecule or out-of-distribution performance claim for any model.",
        "",
        "## Artifacts",
        "",
        "The manuscript figure is `figures/main_lbvs_figure.{pdf,svg,png}`; the compact supplementary figure is `figures/si_lbvs_secondary_metrics.{pdf,svg,png}`; and the seven-model target-balanced ROC plot is `figures/five_shot_macro_roc_curves.{pdf,svg,png}`. Every plotted value is retained in `figures/source-data/`.",
        "",
    ]
    report_path = BENCHMARK_DIR / "RESULTS.md"
    atomic_write_text(report_path, "\n".join(lines))
    state = {
        "schema_version": 1,
        "status": "ok",
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
        "population_state_sha256": sha256_file(BENCHMARK_DIR / "state/POPULATION_FROZEN.json"),
        "summary_state_sha256": sha256_file(BENCHMARK_DIR / "state/SUMMARY_COMPLETE.json"),
        "exposure_audit_sha256": sha256_file(BENCHMARK_DIR / "audits/pretraining_exposure.json"),
        "figure_manifest_sha256": sha256_file(BENCHMARK_DIR / "audits/figure_manifest.json"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(BENCHMARK_DIR / "state/REPORT_COMPLETE.json", state)
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
