#!/usr/bin/env python3
"""Train only the conditional SMILES decoder on frozen gMolAI vectors."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdkit
import torch
import torch.nn.functional as F
from rdkit import RDLogger

from decoder_model import (
    ConditionalSmilesTransformer,
    decoder_parameter_count,
)
from gmolai_retrain.chem import Rejection, canonicalize
from study_common import (
    PAD_TOKEN,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    decode_tokens,
    derangement,
    load_validate_manifest,
    released_train_embeddings,
    sha256_file,
    stable_digest,
)

RDLogger.DisableLog("rdApp.*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_prepared(step_root: Path) -> dict[str, Path]:
    seal_path = step_root / "state" / "PREPARED.json"
    if not seal_path.is_file():
        raise RuntimeError("Train-only preparation stage is not sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    result: dict[str, Path] = {}
    for name, record in seal["outputs"].items():
        path = step_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Prepared artifact changed: {name}")
        result[name] = path
    return result


def cross_entropy_statistics(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    losses = F.cross_entropy(
        logits.transpose(1, 2).float(),
        targets,
        reduction="none",
        ignore_index=PAD_TOKEN,
    )
    mask = targets.ne(PAD_TOKEN)
    counts = mask.sum(dim=1).clamp_min(1)
    sequence_nll = (losses * mask).sum(dim=1) / counts
    mean_nll = (losses * mask).sum() / mask.sum().clamp_min(1)
    accuracy = (
        (logits.argmax(dim=-1).eq(targets) & mask).sum()
        / mask.sum().clamp_min(1)
    )
    return mean_nll, sequence_nll, accuracy


def teacher_forced_metrics(
    model: ConditionalSmilesTransformer,
    tokens: torch.Tensor,
    conditions: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "tokens": 0.0,
        "correct_loss": 0.0,
        "shuffled_loss": 0.0,
        "zero_loss": 0.0,
        "correct_tokens": 0.0,
    }
    mapping = derangement(len(indices), 98_721, "train-dev-teacher")
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            stop = min(offset + batch_size, len(indices))
            current_np = indices[offset:stop]
            wrong_np = indices[mapping[offset:stop]]
            current = torch.as_tensor(
                current_np, dtype=torch.long, device=device
            )
            wrong = torch.as_tensor(
                wrong_np, dtype=torch.long, device=device
            )
            sequence = tokens[current].long()
            inputs, targets = sequence[:, :-1], sequence[:, 1:]
            mask_count = float(targets.ne(PAD_TOKEN).sum().item())
            condition = conditions[current]
            variants = {
                "correct": condition,
                "shuffled": conditions[wrong],
                "zero": torch.zeros_like(condition),
            }
            for name, variant in variants.items():
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    logits = model(inputs, variant)
                loss, _, accuracy = cross_entropy_statistics(logits, targets)
                totals[f"{name}_loss"] += float(loss) * mask_count
                if name == "correct":
                    totals["correct_tokens"] += float(accuracy) * mask_count
            totals["tokens"] += mask_count
    count = max(totals["tokens"], 1.0)
    return {
        "dev_teacher_nll_correct": totals["correct_loss"] / count,
        "dev_teacher_nll_shuffled": totals["shuffled_loss"] / count,
        "dev_teacher_nll_zero": totals["zero_loss"] / count,
        "dev_teacher_token_accuracy": totals["correct_tokens"] / count,
    }


def policy_identity(
    value: str,
    target_hash: str,
    resolved_config: dict[str, Any],
) -> bool:
    if not value:
        return False
    data = resolved_config["data"]
    policy = data["canonicalization"]
    result = canonicalize(
        value,
        isomeric_smiles=bool(policy["isomeric_smiles"]),
        fragment_policy=str(policy["fragment_policy"]),
        allowed_elements={str(item) for item in policy["allowed_elements"]},
        min_atoms=int(policy["min_atoms"]),
        max_atoms=int(policy["max_atoms"]),
        buckets=int(data["hash_buckets"]),
        split_cfg=data["split"],
    )
    return not isinstance(result, Rejection) and result.molecule_hash == target_hash


def generation_metrics(
    model: ConditionalSmilesTransformer,
    tokens: torch.Tensor,
    conditions: torch.Tensor,
    panel_indices: np.ndarray,
    molecules: pd.DataFrame,
    resolved_config: dict[str, Any],
    *,
    batch_size: int,
    maximum_steps: int,
    device: torch.device,
) -> dict[str, float]:
    mapping = derangement(len(panel_indices), 78_131, "train-dev-generation")
    targets = molecules.iloc[panel_indices]
    target_smiles = targets["canonical_smiles"].astype(str).tolist()
    target_hashes = targets["molecule_hash"].astype(str).tolist()
    controls = {
        "correct": panel_indices,
        "shuffled": panel_indices[mapping],
        "zero": None,
    }
    metrics: dict[str, float] = {}
    model.eval()
    for control, source_indices in controls.items():
        decoded: list[str] = []
        for offset in range(0, len(panel_indices), batch_size):
            stop = min(offset + batch_size, len(panel_indices))
            if source_indices is None:
                condition = torch.zeros(
                    (stop - offset, conditions.shape[1]),
                    dtype=conditions.dtype,
                    device=device,
                )
            else:
                source = torch.as_tensor(
                    source_indices[offset:stop],
                    dtype=torch.long,
                    device=device,
                )
                condition = conditions[source]
            generated = model.generate(
                condition, maximum_steps=maximum_steps
            ).cpu().numpy()
            for row in generated:
                value, error = decode_tokens(row)
                decoded.append(value if not error else "")
        exact = np.asarray(
            [value == target for value, target in zip(decoded, target_smiles)],
            dtype=np.float64,
        )
        identity = np.asarray(
            [
                policy_identity(value, target_hash, resolved_config)
                for value, target_hash in zip(decoded, target_hashes)
            ],
            dtype=np.float64,
        )
        metrics[f"dev_generation_exact_{control}"] = float(exact.mean())
        metrics[f"dev_generation_identity_{control}"] = float(
            identity.mean()
        )
    return metrics


def learning_rate(
    step: int, total_steps: int, warmup: int, peak: float, minimum: float
) -> float:
    if step <= warmup:
        return peak * max(step, 1) / max(warmup, 1)
    progress = min(
        1.0, (step - warmup) / max(total_steps - warmup, 1)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine


def checkpoint_payload(
    *,
    model: ConditionalSmilesTransformer,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_score: float,
    patience: int,
    config_sha256: str,
    manifest_sha256: str,
    input_sha256: dict[str, str],
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "decoder_only",
        "decoder_architecture": model_config["architecture"],
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "patience": patience,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "frozen_input_sha256": input_sha256,
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/repo"))
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-02-decoder-feasibility"),
    )
    parser.add_argument(
        "--mode", choices=("pilot", "full"), default="full"
    )
    parser.add_argument("--pilot-rows", type=int, default=65_536)
    parser.add_argument("--pilot-epochs", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Bound a baseline stage while retaining the configured schedule",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    step_root = args.step_root.resolve()
    config_path = step_root / "config" / "protocol.json"
    manifest_path = step_root / "inputs" / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_paths, input_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )
    prepared = validate_prepared(step_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Decoder training requires exactly one visible GPU; observed {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    output_root = (
        step_root / "state" / "pilot"
        if args.mode == "pilot"
        else step_root
    )
    checkpoint_dir = (
        output_root / "checkpoints"
        if args.mode == "pilot"
        else step_root / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dir = (
        output_root
        if args.mode == "pilot"
        else step_root / "state"
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()

    raw_payload = torch.load(
        input_paths["train_raw_embeddings"],
        map_location="cpu",
        weights_only=False,
    )
    calibrator = torch.load(
        input_paths["calibrator"], map_location="cpu", weights_only=False
    )
    released = released_train_embeddings(raw_payload, calibrator)
    molecules = pd.read_parquet(input_paths["train_molecules"])
    expected_columns = {
        "molecule_index",
        "molecule_hash",
        "canonical_smiles",
        "scaffold",
        "heavy_atoms",
    }
    if set(molecules.columns) != expected_columns:
        raise RuntimeError(
            "Training chemistry table contains unexpected columns"
        )
    payload_hashes = [str(value) for value in raw_payload["molecule_hashes"]]
    if payload_hashes != molecules["molecule_hash"].astype(str).tolist():
        raise RuntimeError("Training embedding and molecule identities differ")
    del raw_payload, calibrator, payload_hashes
    gc.collect()

    tokens_np = np.load(prepared["tokens"], mmap_mode="r")
    split = np.load(prepared["splits"])
    train_indices = split["train_indices"].astype(np.int64)
    dev_indices = split["dev_indices"].astype(np.int64)
    if len(released) != len(tokens_np):
        raise RuntimeError("Released conditions and token rows differ")
    if args.mode == "pilot":
        if args.pilot_rows <= 1_024:
            raise ValueError("Pilot requires more than 1,024 train rows")
        train_indices = train_indices[: min(args.pilot_rows, len(train_indices))]
        dev_indices = dev_indices[: min(4_096, len(dev_indices))]
        maximum_epochs = int(args.pilot_epochs)
    else:
        configured_maximum_epochs = int(
            config["training"]["maximum_epochs"]
        )
        maximum_epochs = configured_maximum_epochs
        if args.stop_after_epoch is not None:
            if (
                args.stop_after_epoch <= 0
                or args.stop_after_epoch > configured_maximum_epochs
            ):
                raise ValueError("Invalid bounded training epoch")
            maximum_epochs = int(args.stop_after_epoch)

    print(
        f"loading {len(released):,} frozen conditions and token rows onto {device}",
        flush=True,
    )
    conditions = torch.from_numpy(released).to(device=device)
    tokens = torch.from_numpy(np.asarray(tokens_np)).to(device=device)
    del released, tokens_np
    gc.collect()

    model_config = {
        **config["model"],
        "vocab_size": int(config["data"]["vocab_size"]),
    }
    model = ConditionalSmilesTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    micro_batch = int(config["training"]["micro_batch_size"])
    effective_batch = int(config["training"]["effective_batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if micro_batch * accumulation != effective_batch:
        raise RuntimeError("Effective batch configuration is inconsistent")
    steps_per_epoch = math.ceil(len(train_indices) / effective_batch)
    schedule_epochs = (
        maximum_epochs
        if args.mode == "pilot"
        else int(
            config["training"].get(
                "cosine_schedule_epochs", maximum_epochs
            )
        )
    )
    if schedule_epochs <= 0 or schedule_epochs > maximum_epochs:
        raise RuntimeError("Cosine schedule epoch count is invalid")
    total_steps = steps_per_epoch * schedule_epochs
    global_step = 0
    start_epoch = 1
    best_score = -math.inf
    patience = 0
    curve: list[dict[str, Any]] = []
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    if (
        args.mode == "full"
        and last_path.is_file()
        and not args.restart
    ):
        saved = torch.load(
            last_path, map_location=device, weights_only=False
        )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        global_step = int(saved["global_step"])
        best_score = float(saved["best_score"])
        patience = int(saved["patience"])
        curve = list(saved["curve"])
        print(f"resuming decoder at epoch {start_epoch}", flush=True)

    dev_panel_count = min(
        int(config["training"]["train_dev_generation_rows"]),
        len(dev_indices),
    )
    dev_panel_order = sorted(
        range(len(dev_indices)),
        key=lambda position: stable_digest(
            seed,
            "dev-generation-panel",
            molecules.iloc[int(dev_indices[position])]["molecule_hash"],
        ),
    )
    dev_panel = dev_indices[
        np.asarray(dev_panel_order[:dev_panel_count], dtype=np.int64)
    ]
    resolved_config = json.loads(
        input_paths["resolved_config"].read_text(encoding="utf-8")
    )
    peak_lr = float(config["training"]["learning_rate"])
    minimum_lr = float(config["training"]["minimum_learning_rate"])
    warmup = int(config["training"]["warmup_optimizer_steps"])
    negative_fraction = float(
        config["training"]["wrong_condition_fraction"]
    )
    margin = float(
        config["training"]["wrong_condition_margin_nats_per_token"]
    )
    negative_weight = float(
        config["training"]["wrong_condition_loss_weight"]
    )
    minimum_epochs = (
        1
        if args.mode == "pilot"
        else int(config["training"]["minimum_epochs"])
    )
    early_patience = int(config["training"]["early_stopping_patience"])

    for epoch in range(start_epoch, maximum_epochs + 1):
        model.train()
        generator = np.random.default_rng(seed + epoch)
        permutation = generator.permutation(len(train_indices))
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_margin_sum = 0.0
        epoch_accuracy_sum = 0.0
        epoch_rows = 0
        epoch_started = time.perf_counter()
        for batch_number, offset in enumerate(
            range(0, len(train_indices), effective_batch), start=1
        ):
            selected = permutation[offset : offset + effective_batch]
            batch_np = train_indices[selected]
            batch_index = torch.as_tensor(
                batch_np, dtype=torch.long, device=device
            )
            sequence = tokens[batch_index].long()
            condition = conditions[batch_index]
            inputs, targets = sequence[:, :-1], sequence[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                logits = model(inputs, condition)
                correct_ce, correct_sequence_nll, accuracy = (
                    cross_entropy_statistics(logits, targets)
                )
                negative_rows = max(
                    2, int(len(batch_index) * negative_fraction)
                )
                wrong_condition = torch.roll(condition, shifts=1, dims=0)[
                    :negative_rows
                ]
                wrong_logits = model(
                    inputs[:negative_rows], wrong_condition
                )
                _, wrong_sequence_nll, _ = cross_entropy_statistics(
                    wrong_logits, targets[:negative_rows]
                )
                condition_margin = F.relu(
                    margin
                    + correct_sequence_nll[:negative_rows]
                    - wrong_sequence_nll
                ).mean()
                loss = correct_ce + negative_weight * condition_margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["gradient_clip_norm"]),
            )
            global_step += 1
            current_lr = learning_rate(
                global_step,
                total_steps,
                warmup,
                peak_lr,
                minimum_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.step()

            rows = len(batch_index)
            epoch_rows += rows
            epoch_loss_sum += float(loss.detach()) * rows
            epoch_ce_sum += float(correct_ce.detach()) * rows
            epoch_margin_sum += float(condition_margin.detach()) * rows
            epoch_accuracy_sum += float(accuracy.detach()) * rows
            if batch_number % 50 == 0 or offset + rows == len(train_indices):
                elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                print(
                    f"epoch {epoch}/{maximum_epochs} batch "
                    f"{batch_number}/{steps_per_epoch} "
                    f"rows={epoch_rows:,} rate={epoch_rows/elapsed:,.0f}/s "
                    f"loss={epoch_loss_sum/epoch_rows:.4f} "
                    f"lr={current_lr:.6g}",
                    flush=True,
                )

        dev_teacher = teacher_forced_metrics(
            model,
            tokens,
            conditions,
            dev_indices,
            batch_size=int(config["training"]["dev_batch_size"]),
            device=device,
        )
        dev_generation = generation_metrics(
            model,
            tokens,
            conditions,
            dev_panel,
            molecules,
            resolved_config,
            batch_size=min(512, int(config["training"]["dev_batch_size"])),
            maximum_steps=int(config["data"]["maximum_smiles_bytes"]),
            device=device,
        )
        control_max = max(
            dev_generation["dev_generation_identity_shuffled"],
            dev_generation["dev_generation_identity_zero"],
        )
        condition_gap = (
            dev_generation["dev_generation_identity_correct"] - control_max
        )
        score = (
            dev_generation["dev_generation_identity_correct"]
            + 0.25 * condition_gap
            - 0.01 * dev_teacher["dev_teacher_nll_correct"]
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_rows": epoch_rows,
            "train_loss": epoch_loss_sum / max(epoch_rows, 1),
            "train_cross_entropy": epoch_ce_sum / max(epoch_rows, 1),
            "train_condition_margin_loss": epoch_margin_sum
            / max(epoch_rows, 1),
            "train_token_accuracy": epoch_accuracy_sum / max(epoch_rows, 1),
            **dev_teacher,
            **dev_generation,
            "dev_condition_identity_gap": condition_gap,
            "checkpoint_score": score,
            "epoch_wall_time_seconds": time.perf_counter() - epoch_started,
        }
        curve.append(record)
        improved = score > best_score
        if improved:
            best_score = score
            patience = 0
        else:
            patience += 1
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
            patience=patience,
            config_sha256=sha256_file(config_path),
            manifest_sha256=sha256_file(manifest_path),
            input_sha256=input_hashes,
            curve=curve,
        )
        atomic_torch_save(last_path, payload, step_root)
        if improved:
            atomic_torch_save(best_path, payload, step_root)
        curve_path = (
            state_dir / "training_curve.csv"
            if args.mode == "pilot"
            else step_root / "outputs" / "tables" / "training_curve.csv"
        )
        atomic_write_csv(curve_path, pd.DataFrame(curve), step_root)
        print(
            f"epoch {epoch} dev_identity="
            f"{dev_generation['dev_generation_identity_correct']:.4f} "
            f"condition_gap={condition_gap:.4f} score={score:.4f} "
            f"best={improved}",
            flush=True,
        )
        if (
            args.mode == "full"
            and epoch >= minimum_epochs
            and patience >= early_patience
        ):
            print("early stopping threshold reached", flush=True)
            break

    finished_at = utc_now()
    wall_time = time.perf_counter() - started
    _, final_hashes = load_validate_manifest(
        repo_root, step_root, manifest
    )
    if final_hashes != input_hashes:
        raise RuntimeError("A frozen input changed during decoder training")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "mode": args.mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": wall_time,
        "epochs_completed": int(curve[-1]["epoch"]),
        "run_epoch_ceiling": maximum_epochs,
        "configured_maximum_epochs": int(
            config["training"]["maximum_epochs"]
        ),
        "cosine_schedule_epochs": schedule_epochs,
        "global_steps": global_step,
        "active_train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "dev_generation_rows": len(dev_panel),
        "best_checkpoint_score": best_score,
        "best_checkpoint": str(best_path.relative_to(step_root)),
        "best_checkpoint_sha256": sha256_file(best_path),
        "decoder_parameters": decoder_parameter_count(model),
        "frozen_gmolai_parameters_in_optimizer": 0,
        "embedding_space": "released_hybrid_w3",
        "gpu": torch.cuda.get_device_name(0),
        "maximum_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rdkit": rdkit.__version__,
        },
        "platform": platform.platform(),
        "python": sys.version,
        "input_sha256": input_hashes,
    }
    summary_path = (
        state_dir / "PILOT_COMPLETE.json"
        if args.mode == "pilot"
        else step_root / "state" / "TRAINING_COMPLETE.json"
    )
    atomic_write_json(summary_path, summary, step_root)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
