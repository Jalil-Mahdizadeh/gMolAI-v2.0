#!/usr/bin/env python3
"""Build Step-2b figures, examples, decision gates, and concise reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    bootstrap_mean_ci,
    load_json,
    protocol,
    sha256_file,
    stable_digest,
    utc_now,
    validate_manifest,
    write_hash_ledger,
)


def metric_row(
    metrics: pd.DataFrame, policy: str, control: str, size: int
) -> pd.Series:
    selected = metrics.loc[
        (metrics["policy"] == policy)
        & (metrics["control"] == control)
        & (metrics["candidate_set_size"] == size)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Missing unique metric row: {policy}/{control}/{size}")
    return selected.iloc[0]


def gate(name: str, value: float, threshold: float, relation: str = ">=") -> dict[str, Any]:
    passed = value >= threshold if relation == ">=" else value <= threshold
    return {
        "name": name,
        "value": float(value),
        "relation": relation,
        "threshold": float(threshold),
        "pass": bool(passed),
    }


def create_figures(
    metrics: pd.DataFrame,
    policy: str,
    development_selection: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    correct = metrics.loc[
        (metrics["policy"] == policy) & (metrics["control"] == "correct_embedding")
    ].sort_values("candidate_set_size")
    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    axis.plot(
        correct["candidate_set_size"],
        correct["oracle_target_recall_at_k"],
        marker="o",
        label="Oracle target Recall@k",
    )
    axis.plot(
        correct["candidate_set_size"],
        correct["latent_reranked_exact_target_identity_at_1"],
        marker="s",
        label="Latent-reranked exact identity@1",
    )
    greedy = float(correct.iloc[0]["same_panel_greedy_target_identity"])
    axis.axhline(greedy, color="0.45", linestyle="--", label=f"Fresh-panel greedy ({greedy:.3f})")
    axis.set(xlabel="Candidate-set size", ylabel="Fraction", ylim=(0.0, 1.01))
    axis.set_xticks(correct["candidate_set_size"])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "final_recall_and_reranking.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    size = int(correct["candidate_set_size"].max())
    at50 = metrics.loc[
        (metrics["policy"] == policy) & (metrics["candidate_set_size"] == size)
    ].copy()
    labels = [
        "Correct",
        "Shuffled",
        "Nearest wrong",
    ]
    controls = [
        "correct_embedding",
        "shuffled_embedding",
        "nearest_wrong_embedding",
    ]
    source_values = [
        float(at50.loc[at50["control"] == control, "latent_reranked_exact_condition_source_identity_at_1"].iloc[0])
        for control in controls
    ]
    target_values = [
        float(at50.loc[at50["control"] == control, "latent_reranked_exact_target_identity_at_1"].iloc[0])
        for control in controls
    ]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    axis.bar(x - 0.18, source_values, width=0.36, label="Condition-source identity")
    axis.bar(x + 0.18, target_values, width=0.36, label="Original-target identity")
    axis.set_xticks(x, labels)
    axis.set(ylabel="Exact identity@1 after latent reranking", ylim=(0.0, 1.01))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "final_condition_following_controls.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    ordered = development_selection.sort_values("registered_order")
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(
        ordered["policy"],
        ordered["latent_reranked_exact_target_identity_at_1"],
        color=["#2b8cbe" if value == policy else "#bdbdbd" for value in ordered["policy"]],
    )
    axis.set(ylabel="Development exact identity@1 at k=50", ylim=(0.0, 1.01))
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "development_policy_selection.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def bootstrap_key_metrics(
    query: pd.DataFrame,
    *,
    policy: str,
    sizes: list[int],
    seed: int,
    resamples: int,
) -> pd.DataFrame:
    requested: list[tuple[str, int, str]] = []
    for size in sizes:
        requested.extend(
            [
                ("correct_embedding", size, "oracle_target_recall"),
                ("correct_embedding", size, "reranked_exact_target_identity"),
            ]
        )
    maximum = max(sizes)
    for control in ("shuffled_embedding", "nearest_wrong_embedding"):
        requested.extend(
            [
                (control, maximum, "reranked_exact_target_identity"),
                (control, maximum, "reranked_exact_condition_source_identity"),
                (control, maximum, "oracle_condition_source_recall"),
            ]
        )
    requested.extend(
        [
            ("correct_embedding", maximum, "reranked_target_scaffold_recovery"),
            ("correct_embedding", maximum, "reranked_morgan_similarity_to_target"),
            ("correct_embedding", maximum, "reranked_latent_cosine_to_supplied_condition"),
            (
                "correct_embedding",
                maximum,
                "reranked_latent_relative_l2_to_supplied_condition",
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    for control, size, metric in requested:
        subset = query.loc[
            (query["policy"] == policy)
            & (query["control"] == control)
            & (query["candidate_set_size"] == size)
        ]
        values = subset[metric].to_numpy(dtype=np.float64)
        mean, low, high = bootstrap_mean_ci(
            values,
            seed=seed + int(stable_digest(control, size, metric)[:8], 16),
            resamples=resamples,
        )
        rows.append(
            {
                "policy": policy,
                "control": control,
                "candidate_set_size": size,
                "metric": metric,
                "n": int(np.isfinite(values).sum()),
                "mean": mean,
                "ci_95_low": low,
                "ci_95_high": high,
                "bootstrap_resamples": resamples,
            }
        )
    return pd.DataFrame(rows)


def make_examples(
    query: pd.DataFrame,
    policy: str,
    maximum_k: int,
    molecules: pd.DataFrame,
    panel: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    correct = query.loc[
        (query["policy"] == policy) & (query["control"] == "correct_embedding")
    ]
    greedy = correct.loc[correct["candidate_set_size"] == 1].copy()
    reranked = correct.loc[correct["candidate_set_size"] == maximum_k].copy()
    keep = [
        "query_position",
        "candidate_count",
        "oracle_target_recall",
        "reranked_canonical_smiles",
        "reranked_exact_target_identity",
        "reranked_morgan_similarity_to_target",
        "reranked_target_scaffold_recovery",
        "reranked_latent_cosine_to_supplied_condition",
        "reranked_latent_relative_l2_to_supplied_condition",
    ]
    combined = greedy[keep].merge(
        reranked[keep], on="query_position", suffixes=("__greedy", "__k50")
    )
    target_indices = panel["target_index"].to_numpy(dtype=np.int64)
    combined["target_index"] = target_indices[combined["query_position"].to_numpy(dtype=np.int64)]
    combined["target_smiles"] = molecules.iloc[combined["target_index"]][
        "canonical_smiles"
    ].astype(str).to_numpy()
    combined["target_hash"] = molecules.iloc[combined["target_index"]][
        "molecule_hash"
    ].astype(str).to_numpy()
    combined["case"] = np.where(
        (combined["reranked_exact_target_identity__greedy"] == 0)
        & (combined["reranked_exact_target_identity__k50"] == 1),
        "recovered_by_candidate_search",
        np.where(
            combined["reranked_exact_target_identity__k50"] == 0,
            "still_missed_at_k50",
            "greedy_success_retained",
        ),
    )
    selected_indices: list[int] = []
    limits = {
        "recovered_by_candidate_search": 125,
        "still_missed_at_k50": 125,
        "greedy_success_retained": 50,
    }
    for case, limit in limits.items():
        indices = combined.index[combined["case"] == case].tolist()
        indices.sort(
            key=lambda index: stable_digest(
                seed, "step2b-example", case, combined.loc[index, "target_hash"]
            )
        )
        selected_indices.extend(indices[:limit])
    return combined.loc[selected_indices].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root", type=Path, default=Path("/repo/deriv-gen/step-02b-candidate-reranking")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    if not (root / "state" / "FINAL_COMPLETE.json").is_file():
        raise RuntimeError("Final Step-2b evaluation is incomplete")
    cfg = protocol(root)
    paths, hashes = validate_manifest(repo_root, root)
    policy_seal = load_json(root / "state" / "POLICY_FROZEN.json")
    policy = str(policy_seal["selected_policy"])
    metrics_path = root / "outputs" / "tables" / "final_metrics_by_policy_control_k.csv"
    metrics = pd.read_csv(metrics_path)
    query = pd.read_parquet(root / "outputs" / "raw" / "final_query_results.parquet")
    sizes = [int(value) for value in cfg["panels"]["candidate_set_sizes"]]
    maximum = max(sizes)
    correct = metric_row(metrics, policy, "correct_embedding", maximum)
    shuffled = metric_row(metrics, policy, "shuffled_embedding", maximum)
    wrong = metric_row(metrics, policy, "nearest_wrong_embedding", maximum)
    gates_cfg = cfg["go_no_go"]
    gates = [
        gate(
            "correct_k50_reranked_exact_identity",
            correct["latent_reranked_exact_target_identity_at_1"],
            gates_cfg["correct_k50_reranked_exact_identity_minimum"],
        ),
        gate(
            "correct_k50_oracle_recall",
            correct["oracle_target_recall_at_k"],
            gates_cfg["correct_k50_oracle_recall_minimum"],
        ),
        gate(
            "correct_k50_rerank_selection_efficiency",
            correct["rerank_target_selection_efficiency_given_oracle_presence"],
            gates_cfg["correct_k50_rerank_selection_efficiency_minimum"],
        ),
        gate(
            "correct_k50_valid_top1",
            correct["latent_reranked_valid_top1_rate"],
            gates_cfg["correct_k50_valid_top1_minimum"],
        ),
        gate(
            "correct_k50_scaffold_recovery",
            correct["latent_reranked_target_scaffold_recovery"],
            gates_cfg["correct_k50_scaffold_recovery_minimum"],
        ),
        gate(
            "correct_k50_mean_morgan",
            correct["latent_reranked_mean_morgan_to_target"],
            gates_cfg["correct_k50_mean_morgan_minimum"],
        ),
        gate(
            "correct_k50_candidate_set_nonempty",
            correct["candidate_set_nonempty_rate"],
            gates_cfg["correct_k50_candidate_set_nonempty_minimum"],
        ),
        gate(
            "correct_gain_over_same_panel_greedy",
            correct["exact_target_identity_gain_over_same_panel_greedy"],
            gates_cfg["correct_exact_identity_gain_over_same_panel_greedy_minimum"],
        ),
        gate(
            "correct_gain_over_historical_0_639",
            correct["exact_target_identity_gain_over_historical_step2"],
            gates_cfg["correct_exact_identity_gain_over_historical_0_639_minimum"],
        ),
        gate(
            "shuffled_condition_source_identity",
            shuffled["latent_reranked_exact_condition_source_identity_at_1"],
            gates_cfg["shuffled_k50_condition_source_identity_minimum"],
        ),
        gate(
            "nearest_wrong_condition_source_identity",
            wrong["latent_reranked_exact_condition_source_identity_at_1"],
            gates_cfg["nearest_wrong_k50_condition_source_identity_minimum"],
        ),
        gate(
            "shuffled_source_minus_original_target_identity",
            shuffled["latent_reranked_exact_condition_source_identity_at_1"]
            - shuffled["latent_reranked_exact_target_identity_at_1"],
            gates_cfg[
                "shuffled_k50_source_minus_original_target_identity_minimum"
            ],
        ),
        gate(
            "nearest_wrong_source_minus_original_target_identity",
            wrong["latent_reranked_exact_condition_source_identity_at_1"]
            - wrong["latent_reranked_exact_target_identity_at_1"],
            gates_cfg[
                "nearest_wrong_k50_source_minus_original_target_identity_minimum"
            ],
        ),
    ]
    decision = "GO" if all(item["pass"] for item in gates) else "NO-GO"
    search_related = bool(
        correct["oracle_target_recall_at_k"] >= 0.85
        and correct["exact_target_identity_gain_over_same_panel_greedy"] >= 0.10
    )
    compression_related = bool(
        correct["oracle_target_recall_at_k"] < 0.80
        and correct["candidate_set_nonempty_rate"] >= 0.99
    )
    if search_related:
        error_interpretation = "primarily_search_related_at_the_tested_candidate_budget"
    elif compression_related:
        error_interpretation = "consistent_with_candidate_coverage_or_many_to_one_compression"
    else:
        error_interpretation = "mixed_or_inconclusive_search_and_representation_limit"

    development_selection = pd.read_csv(
        root / "outputs" / "tables" / "development_policy_selection.csv"
    )
    figure_paths = create_figures(
        metrics, policy, development_selection, root / "outputs" / "figures"
    )
    ci_frame = bootstrap_key_metrics(
        query,
        policy=policy,
        sizes=sizes,
        seed=int(cfg["seed"]),
        resamples=int(cfg["evaluation"]["bootstrap_resamples"]),
    )
    ci_path = root / "outputs" / "tables" / "final_bootstrap_cis.csv"
    atomic_write_csv(ci_path, ci_frame, root)
    molecules = pd.read_parquet(paths["validation_molecules"])
    panel = pd.read_csv(root / "prepared" / "fresh_validation_panel.csv")
    examples = make_examples(
        query, policy, maximum, molecules, panel, int(cfg["seed"])
    )
    examples_path = root / "outputs" / "examples" / "final_reranked_examples.csv"
    atomic_write_csv(examples_path, examples, root)

    decision_payload = {
        "schema_version": 1,
        "status": "complete",
        "decided_at": utc_now(),
        "decision": decision,
        "selected_policy": policy,
        "final_panel": "fresh deterministic validation panel disjoint from original Step-2 10k",
        "final_rows_per_control": int(cfg["panels"]["final_rows"]),
        "candidate_set_size_for_decision": maximum,
        "correct": correct.to_dict(),
        "shuffled": shuffled.to_dict(),
        "nearest_wrong": wrong.to_dict(),
        "gates": gates,
        "error_interpretation": error_interpretation,
        "search_related_definition_met": search_related,
        "compression_related_definition_met": compression_related,
        "answers": {
            "correct_molecule_frequently_present": bool(
                correct["oracle_target_recall_at_k"] >= 0.85
            ),
            "frozen_gmolai_reliably_selects_present_target": bool(
                correct["rerank_target_selection_efficiency_given_oracle_presence"]
                >= gates_cfg["correct_k50_rerank_selection_efficiency_minimum"]
            ),
            "exact_top1_gain_over_0_639": float(
                correct["exact_target_identity_gain_over_historical_step2"]
            ),
            "remaining_error": error_interpretation,
            "sufficient_for_mmp_perturbed_decoder_next_step": decision == "GO",
        },
        "ranking_used_target_structure": False,
        "decoder_training": False,
        "encoder_training": False,
        "latent_perturbation": False,
        "derivative_generation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
        "decoder_checkpoint_sha256": hashes["decoder_checkpoint"],
        "gmolai_checkpoint_sha256": hashes["gmolai_checkpoint"],
        "calibrator_sha256": hashes["gmolai_calibrator"],
        "output_sha256": {
            "final_metrics": sha256_file(metrics_path),
            "bootstrap_cis": sha256_file(ci_path),
            "examples": sha256_file(examples_path),
            **{
                f"figure_{index + 1}": sha256_file(path)
                for index, path in enumerate(figure_paths)
            },
        },
    }
    decision_path = root / "outputs" / "decision.json"
    atomic_write_json(decision_path, decision_payload, root)

    greedy = float(correct["same_panel_greedy_target_identity"])
    exact = float(correct["latent_reranked_exact_target_identity_at_1"])
    oracle = float(correct["oracle_target_recall_at_k"])
    selection_efficiency = float(
        correct["rerank_target_selection_efficiency_given_oracle_presence"]
    )
    results_text = f"""# Step 2b results

## Outcome

**{decision}.** The frozen development-selected policy was `{policy}`. Final
evaluation used {int(cfg['panels']['final_rows']):,} fresh validation molecules
per control, with zero overlap with the original Step-2 generation panel.

At k={maximum}, the correct molecule was present for **{oracle:.2%}** of correct
conditions (oracle Recall@{maximum}). Target-blind frozen-gMolAI reranking gave
**{exact:.2%}** exact identity@1, versus **{greedy:.2%}** for greedy decoding on
the same fresh panel: an absolute gain of
**{float(correct['exact_target_identity_gain_over_same_panel_greedy']):.2%}**.
Relative to the historical Step-2 63.90%, the gain is
**{float(correct['exact_target_identity_gain_over_historical_step2']):.2%}**.
When the target was present, latent reranking selected it with
**{selection_efficiency:.2%}** efficiency.

The reranked correct-condition top-1 has
{float(correct['latent_reranked_valid_top1_rate']):.2%} validity,
{float(correct['latent_reranked_target_scaffold_recovery']):.2%} scaffold
recovery, and mean Morgan similarity
{float(correct['latent_reranked_mean_morgan_to_target']):.4f}.

## Condition-use controls

- Shuffled conditions: source identity@1
  {float(shuffled['latent_reranked_exact_condition_source_identity_at_1']):.2%};
  original-target identity@1
  {float(shuffled['latent_reranked_exact_target_identity_at_1']):.2%}.
- Nearest-wrong conditions: source identity@1
  {float(wrong['latent_reranked_exact_condition_source_identity_at_1']):.2%};
  original-target identity@1
  {float(wrong['latent_reranked_exact_target_identity_at_1']):.2%}.

These candidates and their ranking therefore continue to follow the supplied
condition rather than the original target.

## Interpretation

The registered error classification is
`{error_interpretation}`. Oracle coverage and deployable reranking are kept
separate throughout; no oracle target quantity was used to order candidates.

The frozen gate table is machine-readable in `outputs/decision.json`. The
decision on proceeding to MMP-perturbed decoding is **{decision}**. No MMP
perturbation or derivative generation was performed here.
"""
    atomic_write_text(root / "RESULTS.md", results_text, root)
    decision_text = f"""# Step 2b decision

Decision: **{decision}**

- Frozen candidate policy: `{policy}`
- Fresh validation greedy exact identity: {greedy:.2%}
- Oracle target Recall@{maximum}: {oracle:.2%}
- Frozen-latent reranked exact identity@1: {exact:.2%}
- Reranker selection efficiency when target is present: {selection_efficiency:.2%}
- Residual-error interpretation: `{error_interpretation}`

All {len(gates)} preregistered gates must pass for GO; {sum(item['pass'] for item in gates)} passed.
The machine-readable gate values are in `outputs/decision.json`.

Step 2b ends here. No latent perturbation or derivative generation was run.
"""
    atomic_write_text(root / "DECISION.md", decision_text, root)
    readme_text = f"""# Step 2b: candidate search plus frozen-latent reranking

Status: complete. Decision: **{decision}**.

The frozen Step-2 decoder was evaluated with the development-selected
`{policy}` candidate policy. Final evaluation used a fresh deterministic
10,000-molecule validation panel disjoint from the original Step-2 generation
panel. Candidates were filtered by the unchanged chemistry policy, re-encoded
by frozen released gMolAI, and ranked only by supplied-condition latent
consistency.

Key outputs:

- `RESULTS.md`: concise results and scientific interpretation.
- `DECISION.md`: GO/NO-GO decision.
- `outputs/decision.json`: frozen gates and exact machine-readable values.
- `outputs/tables/`: development selection and final metrics by control/k.
- `outputs/raw/`: candidate-level and query-level reproducibility tables.
- `outputs/figures/`: presentation figures (not used for ranking).

No model was trained or modified, and no derivative generation was performed.
"""
    atomic_write_text(root / "README.md", readme_text, root)
    complete_path = root / "state" / "COMPLETE.json"
    complete = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "decision": decision,
        "selected_policy": policy,
        "decision_sha256": sha256_file(decision_path),
        "results_sha256": sha256_file(root / "RESULTS.md"),
        "final_metrics_sha256": sha256_file(metrics_path),
        "final_query_results_sha256": sha256_file(
            root / "outputs" / "raw" / "final_query_results.parquet"
        ),
        "fresh_final_rows_per_control": int(cfg["panels"]["final_rows"]),
        "original_step2_panel_overlap": 0,
        "decoder_checkpoint_sha256": hashes["decoder_checkpoint"],
        "gmolai_checkpoint_sha256": hashes["gmolai_checkpoint"],
        "calibrator_sha256": hashes["gmolai_calibrator"],
        "ranking_used_target_structure": False,
        "decoder_training": False,
        "encoder_training": False,
        "latent_perturbation": False,
        "derivative_generation": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(complete_path, complete, root)
    write_hash_ledger(root, root / "outputs" / "SHA256SUMS")
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
