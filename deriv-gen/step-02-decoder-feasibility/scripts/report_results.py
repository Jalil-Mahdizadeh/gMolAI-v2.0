#!/usr/bin/env python3
"""Create Step 2 figures, summaries, and the frozen GO/NO-GO decision."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
    hash_ledger,
    load_validate_manifest,
    sha256_file,
)

CONTROLS = [
    "correct_embedding",
    "shuffled_embedding",
    "zero_embedding",
    "wrong_molecule_embedding",
]
LABEL = {
    "correct_embedding": "Correct",
    "shuffled_embedding": "Shuffled",
    "zero_embedding": "Zero",
    "wrong_molecule_embedding": "Nearest wrong",
}
COLOR = {
    "correct_embedding": "#2563eb",
    "shuffled_embedding": "#d97706",
    "zero_embedding": "#64748b",
    "wrong_molecule_embedding": "#dc2626",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_figure(fig: plt.Figure, path: Path, root: Path) -> None:
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        fig.savefig(
            temporary,
            format=target.suffix.lstrip("."),
            dpi=220 if target.suffix == ".png" else None,
            bbox_inches="tight",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def row(metrics: pd.DataFrame, control: str) -> pd.Series:
    found = metrics.loc[metrics["control"] == control]
    if len(found) != 1:
        raise RuntimeError(f"Expected one metrics row for {control}")
    return found.iloc[0]


def metric(
    metrics: pd.DataFrame,
    control: str,
    name: str,
    statistic: str = "mean",
) -> float:
    return float(row(metrics, control)[f"{name}_{statistic}"])


def interval(metrics: pd.DataFrame, control: str, name: str) -> str:
    current = row(metrics, control)
    values = [
        float(current[f"{name}_mean"]),
        float(current[f"{name}_ci_low"]),
        float(current[f"{name}_ci_high"]),
    ]
    if not all(math.isfinite(item) for item in values):
        return "NA"
    return f"{values[0]:.4f} [{values[1]:.4f}, {values[2]:.4f}]"


def figures(curve: pd.DataFrame, metrics: pd.DataFrame, root: Path) -> None:
    epochs = curve["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes[0, 0].plot(epochs, curve["train_cross_entropy"], marker="o", label="train CE")
    axes[0, 0].plot(epochs, curve["dev_teacher_nll_correct"], marker="o", label="dev NLL")
    axes[0, 0].set_title("Teacher-forced loss")
    axes[0, 0].set_ylabel("nats / token")
    axes[0, 0].legend(frameon=False)
    for name, label, color in (
        ("correct", "correct", "#2563eb"),
        ("shuffled", "shuffled", "#d97706"),
        ("zero", "zero", "#64748b"),
    ):
        axes[0, 1].plot(
            epochs,
            curve[f"dev_generation_identity_{name}"],
            marker="o",
            label=label,
            color=color,
        )
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Train-partition dev generation")
    axes[0, 1].set_ylabel("identity recovery")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(
        epochs, curve["dev_condition_identity_gap"], marker="o", color="#7c3aed"
    )
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Condition-use gap")
    axes[1, 0].set_ylabel("correct - best control")
    axes[1, 1].plot(epochs, curve["learning_rate"], marker="o", color="#059669")
    axes[1, 1].set_title("Optimization schedule")
    axes[1, 1].set_ylabel("learning rate")
    for axis in axes.ravel():
        axis.set_xlabel("epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.2)
    fig.suptitle("Conditional decoder training and validation curves")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(fig, root / "outputs" / "figures" / f"training_curves.{suffix}", root)
    plt.close(fig)

    names = [
        ("rdkit_valid_smiles", "Valid"),
        ("exact_canonical_reconstruction", "Exact canonical"),
        ("molecular_identity_recovery", "Target identity"),
        ("scaffold_recovery", "Scaffold"),
        ("morgan_similarity_to_target", "Morgan"),
    ]
    x = np.arange(len(names))
    width = 0.19
    fig, axis = plt.subplots(figsize=(11.2, 4.8))
    for offset, control in enumerate(CONTROLS):
        axis.bar(
            x + (offset - 1.5) * width,
            [metric(metrics, control, name) for name, _ in names],
            width,
            color=COLOR[control],
            label=LABEL[control],
        )
    axis.set_xticks(x, [label for _, label in names])
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("fraction or mean similarity")
    axis.set_title("Held-out reconstruction and condition-use controls")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(fig, root / "outputs" / "figures" / f"condition_controls.{suffix}", root)
    plt.close(fig)

    x = np.arange(len(CONTROLS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for axis, target_name, supplied_name, title, ylabel in (
        (
            axes[0],
            "reencoded_cosine_to_target",
            "reencoded_cosine_to_supplied_condition",
            "Frozen gMolAI re-encoding",
            "cosine similarity",
        ),
        (
            axes[1],
            "reencoded_relative_l2_to_target",
            "reencoded_relative_l2_to_supplied_condition",
            "Latent reconstruction error",
            "relative L2 error",
        ),
    ):
        axis.bar(
            x - width / 2,
            [metric(metrics, control, target_name) for control in CONTROLS],
            width,
            label="to target",
            color="#2563eb",
        )
        axis.bar(
            x + width / 2,
            [metric(metrics, control, supplied_name) for control in CONTROLS],
            width,
            label="to supplied condition",
            color="#7c3aed",
        )
        axis.set_xticks(x, [LABEL[item] for item in CONTROLS], rotation=18)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylim(-1.0, 1.03)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        save_figure(fig, root / "outputs" / "figures" / f"latent_consistency.{suffix}", root)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-02-decoder-feasibility"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = args.step_root.resolve()
    config_path = root / "config" / "protocol.json"
    manifest_path = root / "inputs" / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, input_hashes = load_validate_manifest(repo_root, root, manifest)
    training_path = root / "state" / "TRAINING_COMPLETE.json"
    evaluation_path = root / "state" / "EVALUATION_COMPLETE.json"
    if not training_path.is_file() or not evaluation_path.is_file():
        raise RuntimeError("Training and evaluation must be sealed first")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    decoder_export = json.loads(
        (root / "state" / "DECODER_EXPORT.json").read_text(
            encoding="utf-8"
        )
    )
    exported_decoder_path = root / decoder_export["artifact"]
    if (
        decoder_export.get("status") != "complete"
        or not exported_decoder_path.is_file()
        or sha256_file(exported_decoder_path)
        != decoder_export["sha256"]
    ):
        raise RuntimeError("Inference-only decoder export is not sealed")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary = json.loads(
        (root / "outputs" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    extension_path = (
        root / "state" / "DEVELOPMENT_EXTENSION_DECISION.json"
    )
    if not extension_path.is_file():
        raise RuntimeError("Development-duration decision is not sealed")
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    curve = pd.read_csv(root / "outputs" / "tables" / "training_curve.csv")
    best_curve_row = curve.loc[curve["checkpoint_score"].idxmax()]
    best_epoch = int(best_curve_row["epoch"])
    if not np.isclose(
        float(best_curve_row["checkpoint_score"]),
        float(training["best_checkpoint_score"]),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Selected checkpoint and training curve disagree")
    metrics = pd.read_csv(root / "outputs" / "tables" / "metrics_by_control.csv")
    teacher = pd.read_csv(root / "outputs" / "tables" / "teacher_forced_by_control.csv")
    decode = json.loads(
        (root / "state" / "DECODE_SELECTION.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        summary["decode_method"] != decode["selected_decode_method"]
        or summary["decode_selection_sha256"]
        != sha256_file(root / "state" / "DECODE_SELECTION.json")
    ):
        raise RuntimeError("Evaluation decode method differs from frozen selection")
    if set(metrics["control"]) != set(CONTROLS):
        raise RuntimeError("Control set differs from frozen protocol")

    limits = config["go_no_go"]
    correct_identity = metric(metrics, "correct_embedding", "molecular_identity_recovery")
    best_wrong_target = max(
        metric(metrics, control, "molecular_identity_recovery")
        for control in CONTROLS[1:]
    )
    advantage = correct_identity - best_wrong_target
    specs = [
        ("generation_rows", evaluation["generation_rows_per_control"], limits["minimum_generation_rows"], "Final validation generation panel size"),
        ("correct_valid_smiles", metric(metrics, "correct_embedding", "rdkit_valid_smiles"), limits["correct_valid_smiles_rate_minimum"], "Correct-condition valid-SMILES rate"),
        ("correct_identity", correct_identity, limits["correct_molecular_identity_recovery_minimum"], "Correct-condition molecular identity recovery"),
        ("correct_scaffold", metric(metrics, "correct_embedding", "scaffold_recovery"), limits["correct_scaffold_recovery_minimum"], "Correct-condition scaffold recovery"),
        ("correct_morgan", metric(metrics, "correct_embedding", "morgan_similarity_to_target"), limits["correct_mean_morgan_similarity_all_rows_minimum"], "Correct-condition all-row mean Morgan similarity"),
        ("correct_reencoded_cosine", metric(metrics, "correct_embedding", "reencoded_cosine_to_supplied_condition", "median"), limits["correct_median_reencoded_cosine_to_condition_minimum"], "Median re-encoded cosine to correct condition"),
        ("correct_identity_advantage", advantage, limits["correct_identity_advantage_over_best_target_control_minimum"], "Correct target-identity advantage over best wrong control"),
        ("hard_wrong_source_identity", metric(metrics, "wrong_molecule_embedding", "condition_source_identity_recovery"), limits["hard_wrong_condition_source_identity_recovery_minimum"], "Nearest-wrong supplied-source identity recovery"),
        ("shuffled_source_identity", metric(metrics, "shuffled_embedding", "condition_source_identity_recovery"), limits["shuffled_condition_source_identity_recovery_minimum"], "Shuffled supplied-source identity recovery"),
    ]
    gate_rows = [
        {
            "gate": name,
            "description": description,
            "observed": float(observed),
            "comparator": ">=",
            "threshold": float(threshold),
            "passed": bool(math.isfinite(float(observed)) and float(observed) >= float(threshold)),
        }
        for name, observed, threshold, description in specs
    ]
    gates = pd.DataFrame(gate_rows)
    gate_status = {
        str(item.gate): bool(item.passed)
        for item in gates.itertuples(index=False)
    }
    condition_dependence = all(
        gate_status[name]
        for name in (
            "correct_identity_advantage",
            "hard_wrong_source_identity",
            "shuffled_source_identity",
        )
    )
    fidelity = all(
        gate_status[name]
        for name in (
            "correct_valid_smiles",
            "correct_identity",
            "correct_scaffold",
            "correct_morgan",
            "correct_reencoded_cosine",
        )
    )
    decision_label = "GO" if bool(gates["passed"].all()) else "NO-GO"
    if decision_label == "GO":
        interpretation = (
            "The decoder learned a faithful, condition-dependent inverse "
            "mapping on held-out validation molecules."
        )
    elif condition_dependence:
        interpretation = (
            "The decoder learned clear condition dependence, but it did not "
            "meet every predeclared fidelity criterion for a faithful inverse."
        )
    else:
        interpretation = (
            "The decoder did not establish a faithful, condition-dependent "
            "inverse under the predeclared criteria."
        )
    atomic_write_csv(root / "outputs" / "tables" / "go_no_go_gates.csv", gates, root)
    decision = {
        "schema_version": 1,
        "decision": decision_label,
        "all_predeclared_gates_pass": bool(gates["passed"].all()),
        "embedding_space": "released_hybrid_w3",
        "decoder_checkpoint_sha256": training["best_checkpoint_sha256"],
        "correct_target_identity_recovery": correct_identity,
        "best_wrong_condition_target_identity_recovery": best_wrong_target,
        "correct_identity_advantage": advantage,
        "condition_dependence_demonstrated": condition_dependence,
        "fidelity_thresholds_pass": fidelity,
        "gates": gate_rows,
        "interpretation": interpretation,
        "scope": "zero-perturbation reconstruction only",
        "next_step_boundary": "No latent perturbation or derivative generation was performed.",
    }
    atomic_write_json(root / "outputs" / "decoder_decision.json", decision, root)
    figures(curve, metrics, root)

    result_rows = [
        "| Condition | Valid | Policy | Exact canonical | Identity | Scaffold | Morgan target | Latent cosine target | Source identity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROLS:
        source_identity = metric(
            metrics, control, "condition_source_identity_recovery"
        )
        source_identity_text = (
            f"{source_identity:.4f}"
            if math.isfinite(source_identity)
            else "NA"
        )
        result_rows.append(
            f"| {LABEL[control]} | {metric(metrics, control, 'rdkit_valid_smiles'):.4f} | "
            f"{metric(metrics, control, 'gmolai_policy_accepted'):.4f} | "
            f"{metric(metrics, control, 'exact_canonical_reconstruction'):.4f} | "
            f"{metric(metrics, control, 'molecular_identity_recovery'):.4f} | "
            f"{metric(metrics, control, 'scaffold_recovery'):.4f} | "
            f"{metric(metrics, control, 'morgan_similarity_to_target'):.4f} | "
            f"{metric(metrics, control, 'reencoded_cosine_to_target'):.4f} | "
            f"{source_identity_text} |"
        )
    teacher_rows = [
        "| Condition | NLL | Token accuracy |",
        "|---|---:|---:|",
    ]
    for current in teacher.itertuples(index=False):
        teacher_rows.append(
            f"| {LABEL[str(current.control)]} | "
            f"{float(current.teacher_forced_nll):.4f} | "
            f"{float(current.teacher_forced_token_accuracy):.4f} |"
        )
    failed = gates.loc[~gates["passed"]]
    failed_text = (
        "None."
        if failed.empty
        else "; ".join(
            f"{item.gate} ({float(item.observed):.4f} < {float(item.threshold):.4f})"
            for item in failed.itertuples(index=False)
        )
    )
    stereo = int(summary["stereochemical_targets"])
    duration_text = (
        "The registered train-development-only extension activated after "
        "epoch 12; the same decoder and optimizer continued at the fixed "
        "learning-rate floor, and the epoch-12 state was archived. "
        if extension["status"] == "activated"
        else "The registered train-development-only extension did not "
        "activate. "
    )
    duration_text += (
        f"The frozen checkpoint was selected at epoch {best_epoch}; "
        f"training stopped at epoch {int(training['epochs_completed'])}, "
        "without final-validation "
        "model selection. The deterministic decode method selected on the "
        f"same development panel was {decode['selected_decode_method']}."
    )
    results = f"""# Decoder feasibility results

## Outcome

**{decision_label}.** {decision["interpretation"]}

A new {int(training["decoder_parameters"]):,}-parameter autoregressive decoder was trained; zero gMolAI parameters entered its optimizer. Fitting used 980,000 train-partition molecules, checkpoint selection used 20,000 scaffold-disjoint train-partition molecules, correct-condition teacher forcing used all 50,000 validation molecules, and all four autoregressive controls used a fixed 10,000-molecule validation panel. Locked-test rows and endpoint labels used: zero.

{duration_text}

## Reconstruction and explicit condition-use controls

{chr(10).join(result_rows)}

Deterministic 2,000-bootstrap 95% confidence intervals are in `outputs/tables/metrics_by_control.csv`. Correct target identity was {correct_identity:.4f}; the best wrong-condition target identity was {best_wrong_target:.4f}; the absolute condition-use gap was {advantage:.4f}. Shuffled and nearest-wrong supplied-source identity recovery were {metric(metrics, "shuffled_embedding", "condition_source_identity_recovery"):.4f} and {metric(metrics, "wrong_molecule_embedding", "condition_source_identity_recovery"):.4f}.

Correct valid-SMILES was {interval(metrics, "correct_embedding", "rdkit_valid_smiles")}; exact canonical reconstruction was {interval(metrics, "correct_embedding", "exact_canonical_reconstruction")}; all-row Morgan similarity was {interval(metrics, "correct_embedding", "morgan_similarity_to_target")}.

## Teacher-forced controls

{chr(10).join(teacher_rows)}

All-validation correct-condition NLL was {float(summary["teacher_forced_all_validation"]["correct_embedding_nll"]):.4f} and token accuracy was {float(summary["teacher_forced_all_validation"]["correct_embedding_token_accuracy"]):.4f}.

## Frozen latent consistency and chemistry

Correct policy-accepted outputs had median re-encoded cosine {metric(metrics, "correct_embedding", "reencoded_cosine_to_supplied_condition", "median"):.4f} and mean relative L2 error {metric(metrics, "correct_embedding", "reencoded_relative_l2_to_supplied_condition"):.4f} to the supplied released vector. Re-encoding used the immutable packaged checkpoint, calibrator, optimized inference path, and released x3 hybrid.

The final panel contained {stereo} stereochemical targets, so stereochemical recovery is {"reported in the raw table" if stereo else "not estimable for this dataset"}. Invalid outputs count as reconstruction failure and zero all-row Morgan similarity.

## Frozen decision audit

Failed gates: {failed_text}

This step ends at zero-perturbation reconstruction. No MMP-direction or derivative generation was performed.
"""
    atomic_write_text(root / "RESULTS.md", results, root)

    gate_lines = [
        "| Gate | Observed | Required | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for item in gates.itertuples(index=False):
        gate_lines.append(
            f"| {item.description} | {float(item.observed):.4f} | "
            f">= {float(item.threshold):.4f} | {'yes' if bool(item.passed) else 'no'} |"
        )
    decision_doc = f"""# Decoder decision

## {decision_label}

{decision["interpretation"]}

{chr(10).join(gate_lines)}

The rule in `PROTOCOL.md` was frozen before final evaluation. Correct embeddings must reconstruct their targets strongly, while shuffled and nearest-wrong embeddings must redirect output toward the molecule supplying the condition. This separates a conditional inverse from an unconditional SMILES language model.

The {decision_label} applies only to condition-dependent, zero-perturbation inversion of `released_hybrid_w3` on held-out validation molecules. No latent perturbation or derivative generation is included.
"""
    atomic_write_text(root / "DECISION.md", decision_doc, root)

    study = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "scientific_boundary": config["scientific_boundary"],
        "data": {
            "decoder_fit_rows": int(training["active_train_rows"]),
            "train_partition_dev_rows": int(training["dev_rows"]),
            "validation_teacher_forced_rows": int(summary["validation_rows_teacher_forced"]),
            "validation_generation_rows_per_control": int(summary["validation_rows_generation"]),
            "controls": CONTROLS,
            "test_rows": 0,
            "endpoint_labels_used": False,
        },
        "frozen_gmolai": {
            "checkpoint_sha256": input_hashes["checkpoint"],
            "calibrator_sha256": input_hashes["calibrator"],
            "embedding_space": "released_hybrid_w3",
            "parameters_in_optimizer": int(training["frozen_gmolai_parameters_in_optimizer"]),
        },
        "decoder": {
            "parameters": int(training["decoder_parameters"]),
            "checkpoint_sha256": training["best_checkpoint_sha256"],
            "inference_artifact": decoder_export["artifact"],
            "inference_artifact_sha256": decoder_export["sha256"],
            "selected_epoch": best_epoch,
            "epochs_completed": int(training["epochs_completed"]),
            "global_steps": int(training["global_steps"]),
            "maximum_gpu_memory_bytes": int(training["maximum_gpu_memory_bytes"]),
            "gpu": training["gpu"],
        },
        "development_duration": extension,
        "decode_selection": decode,
        "evaluation": summary,
        "decision": decision,
        "input_sha256": input_hashes,
    }
    atomic_write_json(root / "outputs" / "study_summary.json", study, root)
    ledger = root / "outputs" / "SHA256SUMS"
    atomic_write_text(
        ledger,
        hash_ledger(root / "outputs", exclude={"SHA256SUMS"}),
        root,
    )
    source_paths = [
        root / ".gitignore",
        root / "README.md",
        root / "DESIGN.md",
        root / "PROTOCOL.md",
        *sorted((root / "config").glob("*.json")),
        root / "inputs" / "manifest.json",
        *sorted((root / "scripts").glob("*.py")),
        *sorted((root / "scripts").glob("*.sh")),
    ]
    source_ledger = root / "state" / "SOURCE_SHA256SUMS"
    atomic_write_text(
        source_ledger,
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in source_paths
        ),
        root,
    )
    complete = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "decision": decision_label,
        "single_gpu": True,
        "train_rows": int(training["active_train_rows"]),
        "train_dev_rows": int(training["dev_rows"]),
        "validation_rows": int(summary["validation_rows_teacher_forced"]),
        "generation_rows_per_control": int(summary["validation_rows_generation"]),
        "controls": CONTROLS,
        "test_rows": 0,
        "endpoint_labels_used": False,
        "latent_perturbation_performed": False,
        "development_extension_status": extension["status"],
        "decode_method": decode["selected_decode_method"],
        "decoder_conditioning_representation": "released_hybrid_w3",
        "frozen_gmolai_parameters_in_optimizer": int(training["frozen_gmolai_parameters_in_optimizer"]),
        "decoder_checkpoint_sha256": training["best_checkpoint_sha256"],
        "decoder_inference_artifact_sha256": decoder_export["sha256"],
        "results_sha256": sha256_file(root / "RESULTS.md"),
        "decision_sha256": sha256_file(root / "DECISION.md"),
        "output_ledger_sha256": sha256_file(ledger),
        "source_ledger_sha256": sha256_file(source_ledger),
        "input_sha256": input_hashes,
    }
    atomic_write_json(root / "state" / "COMPLETE.json", complete, root)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
