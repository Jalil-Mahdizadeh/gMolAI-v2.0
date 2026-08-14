#!/usr/bin/env python3
"""Evaluate and, if triggered, activate the registered dev-only duration extension."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from study_common import atomic_write_json, ensure_within, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_copy(source: Path, destination: Path, root: Path) -> None:
    target = ensure_within(destination, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def epoch_row(curve: pd.DataFrame, epoch: int) -> pd.Series:
    found = curve.loc[curve["epoch"].astype(int) == epoch]
    if len(found) != 1:
        raise RuntimeError(f"Expected one training-curve row for epoch {epoch}")
    return found.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-02-decoder-feasibility"),
    )
    args = parser.parse_args()
    root = args.step_root.resolve()
    policy_path = root / "config" / "development_extension.json"
    config_path = root / "config" / "protocol.json"
    training_path = root / "state" / "TRAINING_COMPLETE.json"
    evaluation_path = root / "state" / "EVALUATION_COMPLETE.json"
    decision_path = root / "state" / "DEVELOPMENT_EXTENSION_DECISION.json"
    if decision_path.exists():
        print(decision_path.read_text(encoding="utf-8"))
        return
    if evaluation_path.exists():
        raise RuntimeError(
            "Refusing a development decision after final evaluation started"
        )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    curve_path = root / "outputs" / "tables" / "training_curve.csv"
    curve = pd.read_csv(curve_path)
    baseline_epoch = int(policy["evaluate_after_epoch"])
    if (
        training.get("status") != "complete"
        or training.get("mode") != "full"
        or int(training["epochs_completed"]) != baseline_epoch
        or int(curve["epoch"].max()) != baseline_epoch
        or int(
            config["training"].get(
                "baseline_epochs", baseline_epoch
            )
        )
        != baseline_epoch
        or int(config["training"]["maximum_epochs"])
        not in {
            baseline_epoch,
            int(policy["action_if_activated"]["maximum_epoch"]),
        }
    ):
        raise RuntimeError("Baseline training is not sealed at the registered epoch")

    earlier = epoch_row(curve, 9)
    final = epoch_row(curve, baseline_epoch)
    limits = policy["activate_only_if_all"]
    correct_identity = float(final["dev_generation_identity_correct"])
    identity_gain = correct_identity - float(
        earlier["dev_generation_identity_correct"]
    )
    best_control = max(
        float(final["dev_generation_identity_shuffled"]),
        float(final["dev_generation_identity_zero"]),
    )
    nll_improved = bool(
        float(final["dev_teacher_nll_correct"])
        < float(earlier["dev_teacher_nll_correct"])
    )
    checks = {
        "correct_identity_below_final_gate": bool(
            correct_identity
            < float(limits["correct_identity_below_final_gate"])
        ),
        "correct_identity_gain_epoch_12_minus_epoch_9": bool(
            identity_gain
            >= float(
                limits[
                    "correct_identity_gain_epoch_12_minus_epoch_9_minimum"
                ]
            )
        ),
        "best_control_below_maximum": bool(
            best_control
            <= float(
                limits[
                    "best_shuffled_or_zero_target_identity_maximum"
                ]
            )
        ),
        "correct_teacher_nll_improved": nll_improved,
    }
    activate = bool(all(checks.values()))
    decision: dict[str, Any] = {
        "schema_version": 1,
        "status": "activated" if activate else "not_activated",
        "decided_at": utc_now(),
        "policy_sha256": sha256_file(policy_path),
        "baseline_config_sha256": sha256_file(config_path),
        "baseline_training_summary_sha256": sha256_file(training_path),
        "baseline_best_checkpoint_sha256": training["best_checkpoint_sha256"],
        "baseline_curve_sha256": sha256_file(curve_path),
        "final_validation_generation_started": False,
        "observed": {
            "epoch_12_correct_identity": correct_identity,
            "epoch_9_correct_identity": float(
                earlier["dev_generation_identity_correct"]
            ),
            "identity_gain_epoch_12_minus_epoch_9": identity_gain,
            "epoch_12_best_shuffled_or_zero_identity": best_control,
            "epoch_12_correct_teacher_nll": float(
                final["dev_teacher_nll_correct"]
            ),
            "epoch_9_correct_teacher_nll": float(
                earlier["dev_teacher_nll_correct"]
            ),
        },
        "checks": checks,
        "action": (
            policy["action_if_activated"]
            if activate
            else policy["action_if_not_activated"]
        ),
    }
    if activate:
        archive = root / "state" / "epoch-12-baseline"
        archive.mkdir(parents=True, exist_ok=True)
        best_path = root / training["best_checkpoint"]
        archived_best = archive / "best.pt"
        archived_curve = archive / "training_curve.csv"
        atomic_copy(best_path, archived_best, root)
        atomic_copy(curve_path, archived_curve, root)
        action = policy["action_if_activated"]
        configured_maximum = int(
            config["training"]["maximum_epochs"]
        )
        if configured_maximum == baseline_epoch:
            config["training"]["maximum_epochs"] = int(
                action["maximum_epoch"]
            )
            config["training"]["cosine_schedule_epochs"] = int(
                action["cosine_schedule_epochs"]
            )
            config["training"][
                "development_extension_policy_sha256"
            ] = decision["policy_sha256"]
            atomic_write_json(config_path, config, root)
        elif (
            configured_maximum != int(action["maximum_epoch"])
            or int(config["training"]["cosine_schedule_epochs"])
            != int(action["cosine_schedule_epochs"])
        ):
            raise RuntimeError(
                "Preconfigured extension ceiling differs from policy"
            )
        archived_summary = archive / "TRAINING_COMPLETE.json"
        os.replace(training_path, archived_summary)
        decision["extended_config_sha256"] = sha256_file(config_path)
        decision["archive"] = {
            "training_summary": str(archived_summary.relative_to(root)),
            "training_summary_sha256": sha256_file(archived_summary),
            "best_checkpoint": str(archived_best.relative_to(root)),
            "best_checkpoint_sha256": sha256_file(archived_best),
            "training_curve": str(archived_curve.relative_to(root)),
            "training_curve_sha256": sha256_file(archived_curve),
        }
    atomic_write_json(decision_path, decision, root)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
