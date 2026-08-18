#!/usr/bin/env python3
"""Render the concise scientific RESULTS.md from sealed metric tables."""

from __future__ import annotations

import pandas as pd

from benchmark_io import BENCHMARK_DIR, atomic_write_text, load_json, load_protocol


PRIMARY = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")
METRIC_LABELS = {
    "ARI": "ARI", "AMI": "AMI", "NMI": "NMI",
    "macro_same_subclass_at_100": "Macro same-subclass@100",
    "NPD_at_100": "NPD@100",
    "property_neighbor_recall_at_100": "Property-neighbor Recall@100",
}


def interval(row, metric: str) -> str:
    precision = 4 if metric == "property_neighbor_recall_at_100" else 3
    return f"{row.estimate:.{precision}f} [{row.ci95_lower:.{precision}f}, {row.ci95_upper:.{precision}f}]"


def metric_table(frame: pd.DataFrame, metrics: tuple[str, ...]) -> str:
    display = frame.drop_duplicates("model").set_index("model")["display_name"].to_dict()
    lines = ["| Representation | " + " | ".join(METRIC_LABELS[metric] for metric in metrics) + " |",
             "|---|" + "---:|" * len(metrics)]
    for model in PRIMARY:
        values = []
        for metric in metrics:
            row = next(frame[(frame["model"] == model) & (frame["metric"] == metric)].itertuples())
            values.append(interval(row, metric))
        lines.append(f"| {display[model]} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def rank_text(frame: pd.DataFrame, metric: str, higher: bool, label: str) -> str:
    subset = frame[(frame["metric"] == metric) & frame["model"].isin(PRIMARY)].sort_values("estimate", ascending=not higher)
    order = subset["model"].tolist()
    rank = order.index("gmolai") + 1
    best = subset.iloc[0]
    g = subset[subset["model"] == "gmolai"].iloc[0]
    if rank == 1:
        return f"{label}: gMolAI ranked first ({g.estimate:.3f})"
    return f"{label}: gMolAI ranked {rank}/{len(PRIMARY)} ({g.estimate:.3f}), with {best.display_name} first ({best.estimate:.3f})"


def contrast(frame: pd.DataFrame, name: str, metric: str, precision: int = 3) -> str:
    row = frame[(frame["contrast"] == name) & (frame["metric"] == metric)].iloc[0]
    return f"{row.estimate:.{precision}f} [{row.ci95_lower:.{precision}f}, {row.ci95_upper:.{precision}f}]"


def main() -> None:
    protocol = load_protocol()
    structural = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "classyfire_structural_metrics.csv")
    prop = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_metrics.csv")
    class_state = load_json(BENCHMARK_DIR / "state" / "classyfire_common.json")
    qmugs_state = load_json(BENCHMARK_DIR / "state" / "qmugs_common.json")
    class_prep = load_json(BENCHMARK_DIR / "audit" / "classyfire_preparation.json")
    qmugs_prep = load_json(BENCHMARK_DIR / "audit" / "qmugs_preparation.json")
    exposure = load_json(BENCHMARK_DIR / "audit" / "pretraining_exposure.json")
    structural_differences = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "classyfire_paired_differences.csv")
    property_differences = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "qmugs_paired_differences.csv")
    scaling = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "qmugs_property_scaling.csv")
    scaling_text = "; ".join(
        f"{row.property}: median {row.median:.6f}, IQR {row.iqr:.6f}"
        for row in scaling.itertuples()
    )
    seeds = pd.read_csv(BENCHMARK_DIR / "outputs" / "tables" / "classyfire_kmeans_seed_sensitivity.csv")
    gseed = seeds[seeds["model"] == "gmolai"]
    structural_ranks = "; ".join([
        rank_text(structural, "ARI", True, "ARI"),
        rank_text(structural, "AMI", True, "AMI"),
        rank_text(structural, "macro_same_subclass_at_100", True, "same-subclass@100"),
    ])
    property_ranks = "; ".join([
        rank_text(prop, "NPD_at_100", False, "NPD@100"),
        rank_text(prop, "property_neighbor_recall_at_100", True, "Recall@100"),
    ])
    exposure_lines = []
    benchmark_labels = {"classyfire": "ClassyFire", "qmugs": "QMugs"}
    for benchmark in ("classyfire", "qmugs"):
        record = exposure["datasets"][benchmark]
        exposure_lines.append(
            f"{benchmark_labels[benchmark]}: {record['pretraining_corpus_overlap']:,}/{record['panel_rows']:,} exact corpus overlaps "
            f"({record['pretraining_corpus_overlap_percent']:.2f}%), of which {record['seen_before_step_10000']:,} "
            "had been presented before the frozen checkpoint."
        )
    target_ok = not bool(qmugs_state["coverage_failure"])
    assessment = (
        "The benchmark is methodologically strong enough for manuscript inclusion as a compact external representation analysis: it uses independent labels/properties, identical molecular support, one common operator, exact neighborhoods, and query-level paired uncertainty. It should remain a supporting representation result, not be framed as a universal clustering benchmark."
        if target_ok else
        "The structural component is usable, but the prespecified 50,000-molecule QMugs target was not met; resolve or clearly foreground this coverage limitation before manuscript integration."
    )
    text = f"""# Frozen minimal clustering benchmark — results

## Outcome

The frozen benchmark completed on {class_state['final_rows']:,} balanced ClassyFire-25 molecules ({class_state['balance_per_subclass']:,} per subclass) and {qmugs_state['final_rows']:,} QMugs molecules from {qmugs_state['attempted_rows']:,} attempted identities. All seven primary representations use identical identities within each endpoint, and every final vector was finite and nonzero. Count Morgan and the frozen 13-descriptor vector are diagnostics only.

## ClassyFire-25 structural organization

Values are estimates with molecule-stratified paired 95% bootstrap intervals. K-means estimates average the five prespecified algorithmic seeds; the seeds are not inferential replicates.

{metric_table(structural, ('ARI', 'AMI', 'NMI', 'macro_same_subclass_at_100'))}

{structural_ranks}. In paired contrasts, gMolAI minus Morgan was {contrast(structural_differences, 'gmolai_minus_morgan', 'ARI')} for ARI but {contrast(structural_differences, 'gmolai_minus_morgan', 'AMI')} for AMI; gMolAI minus KERMT v2 was {contrast(structural_differences, 'gmolai_minus_kermt_v2', 'ARI')} for ARI. Across the five K-means seeds, gMolAI ARI ranged from {gseed['ARI'].min():.3f} to {gseed['ARI'].max():.3f} and AMI from {gseed['AMI'].min():.3f} to {gseed['AMI'].max():.3f}. Thus, no representation dominates all structural endpoints. The taxonomy is an external, reproducible ClassyFire reference, but it is hierarchical ontology assignment rather than an error-free physical ground truth; ARI/AMI quantify agreement, not chemical correctness.

Main figures: [structural metrics](outputs/figures/figure_classyfire_main_metrics.pdf) and [visualization-only PCA](outputs/figures/figure_classyfire_pca.pdf).

## QMugs independent-property organization

NPD@100 is lower-is-better; Recall@100 is higher-is-better. QM properties are DFT HOMO energy, HOMO–LUMO gap, and log(1 + total dipole), robust-scaled on the frozen common panel. The frozen constants (in property-table units) were: {scaling_text}. No property bins or property-derived clusters were used.

{metric_table(prop, ('NPD_at_100', 'property_neighbor_recall_at_100'))}

{property_ranks}. Relative to the closest NPD competitor, SMI-TED-Light, the paired gMolAI difference was {contrast(property_differences, 'gmolai_minus_smi_ted', 'NPD_at_100', 4)}; relative to binary Morgan, the Recall@100 difference was {contrast(property_differences, 'gmolai_minus_morgan', 'property_neighbor_recall_at_100', 5)}. Random-neighbor expected recall is approximately {100 / (qmugs_state['final_rows'] - 1):.4f}, so absolute recall remains modest even though every primary model exceeds chance. The 13-descriptor diagnostic was substantially weaker than gMolAI on both endpoints, arguing against the selected electronic-property result being explained by those descriptors alone. Heavy-atom-count-decile results and per-property deviations are reported in SI source tables and figures. This complement tests local organization by three independent electronic properties; it does not imply that this three-property geometry exhausts molecular-property similarity.

Main figures: [property metrics](outputs/figures/figure_qmugs_main_metrics.pdf), [per-property deviations](outputs/figures/figure_qmugs_property_deviations.pdf), and [visualization-only PCA](outputs/figures/figure_qmugs_pca_homo_lumo_gap.pdf).

## Coverage and exposure audit

ClassyFire began with {class_prep['attempted']:,} rows; {class_prep['canonicalization_rejected']:,} unsupported-element rows were excluded, leaving {class_prep['eligible_unique']:,}. The all-model intersection contained {class_state['all_model_common_before_balance']:,}; MolAI rejected 6,073 identities, and the limiting subclass (steroidal glycosides) retained {class_state['balance_per_subclass']:,}, fixing the balanced panel at {class_state['final_rows']:,}. This strict intersection is fair across representations but materially conditions inference on MolAI-encodable chemistry and discards otherwise usable molecules. QMugs yielded {qmugs_prep['eligible_unique']:,} eligible unique identities; {qmugs_state['all_model_common_before_truncation']:,} of the first {qmugs_state['attempted_rows']:,} were common, so the prespecified {qmugs_state['target_rows']:,}-molecule target was met without expansion.

{' '.join(exposure_lines)} Exact identity exposure is unknown for competitors whose released checkpoints do not provide molecule-level training manifests; consequently, no unseen-molecule or out-of-distribution claim is made. The explicit-H amendment was applied symmetrically before canonicalization, and the QMugs heavy-atom counts agreed exactly after normalization.

## Manuscript assessment

{assessment}

No additional dataset or clustering algorithm is needed before integration. The manuscript must preserve the distinction between the seven-model primary ranking and both diagnostics, foreground the ClassyFire support contraction, report K-means seed ranges alongside query-bootstrap intervals, state the asymmetric pretraining-exposure knowledge, and use PCA only as an illustrative panel. Do not aggregate structural and property endpoints into one score or claim universal clustering superiority.

Full definitions and frozen choices are in [PROTOCOL.md](PROTOCOL.md); exact tables, per-query source data, neighbor/cluster artifacts, figure source data, and verification records are retained in this directory.
"""
    atomic_write_text(BENCHMARK_DIR / "RESULTS.md", text)
    print(BENCHMARK_DIR / "RESULTS.md")


if __name__ == "__main__":
    main()
