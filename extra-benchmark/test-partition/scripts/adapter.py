#!/usr/bin/env python3
"""Screen or encode a frozen panel inside a comparator-specific SIF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

from benchmark_io import (
    atomic_write_json,
    load_protocol,
    read_panel_tsv,
    sha256_file,
    sha256_lines,
)


NEURAL_MODELS = {"molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2"}
SUPPORTED_MODELS = NEURAL_MODELS | {"morgan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--mode", required=True, choices=("screen", "encode"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    return parser.parse_args()


def reject_reason(error: Exception) -> str:
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}"[:1000]


def screen_molai(smiles: str, state: dict[str, Any]) -> str | None:
    if "prepare_smiles" not in state:
        sys.path.insert(0, "/opt/molai")
        from encoder import prepare_smiles

        state["prepare_smiles"] = prepare_smiles
    prepare_smiles = state["prepare_smiles"]

    try:
        prepare_smiles([smiles], allow_unknown_zero=False)
    except Exception as error:
        return reject_reason(error)
    return None


def molformer_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "/opt/molformer/model",
        trust_remote_code=True,
        local_files_only=True,
    )


def screen_molformer(smiles: str, state: dict[str, Any]) -> str | None:
    if "tokenizer" not in state:
        state["tokenizer"] = molformer_tokenizer()
    tokenizer = state["tokenizer"]
    try:
        tokens = tokenizer.tokenize(smiles)
        if "".join(tokens) != smiles:
            return "tokenization_not_lossless"
        encoded = tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        if len(encoded) > 202:
            return f"token_length_{len(encoded)}_exceeds_202"
        unknown_id = tokenizer.unk_token_id
        if unknown_id is not None and unknown_id in encoded:
            return "unknown_token"
    except Exception as error:
        return reject_reason(error)
    return None


def smi_ted_tokenizer_state() -> tuple[Any, Callable[[str], str | None]]:
    sys.path.insert(0, "/opt/smi-ted/model")
    from load import MolTranBertTokenizer, normalize_smiles

    tokenizer = MolTranBertTokenizer(
        vocab_file="/opt/smi-ted/model/bert_vocab_curated.txt"
    )
    return tokenizer, normalize_smiles


def screen_smi_ted(smiles: str, state: dict[str, Any]) -> str | None:
    if "tokenizer_state" not in state:
        state["tokenizer_state"] = smi_ted_tokenizer_state()
    tokenizer, normalize = state["tokenizer_state"]
    try:
        normalized = normalize(smiles)
        if normalized is None:
            return "official_nonisomeric_normalization_failed"
        tokens = tokenizer.tokenize(normalized)
        if "".join(tokens) != normalized:
            return "tokenization_not_lossless_after_official_normalization"
        encoded = tokenizer(
            normalized,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        if len(encoded) > 202:
            return f"token_length_{len(encoded)}_exceeds_202"
        unknown_id = tokenizer.unk_token_id
        if unknown_id is not None and unknown_id in encoded:
            return "unknown_token"
    except Exception as error:
        return reject_reason(error)
    return None


def screen_molclr(smiles: str, state: dict[str, Any]) -> str | None:
    if "smiles_to_graph" not in state:
        sys.path.insert(0, "/opt/molclr")
        from encoder import smiles_to_graph

        state["smiles_to_graph"] = smiles_to_graph
    smiles_to_graph = state["smiles_to_graph"]

    try:
        smiles_to_graph(smiles)
    except Exception as error:
        return reject_reason(error)
    return None


def screen_kermt(smiles: str, _: dict[str, Any]) -> str | None:
    from rdkit import Chem

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return "rdkit_parse_failure"
        Chem.MolToSmiles(molecule)
    except Exception as error:
        return reject_reason(error)
    return None


def screen_morgan(smiles: str, _: dict[str, Any]) -> str | None:
    from rdkit import Chem

    try:
        if Chem.MolFromSmiles(smiles) is None:
            return "rdkit_parse_failure"
    except Exception as error:
        return reject_reason(error)
    return None


SCREENERS: dict[str, Callable[[str, dict[str, Any]], str | None]] = {
    "molai": screen_molai,
    "molformer": screen_molformer,
    "smi_ted": screen_smi_ted,
    "molclr_gin": screen_molclr,
    "kermt_v2": screen_kermt,
    "morgan": screen_morgan,
}


def run_screen(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    state: dict[str, Any] = {}
    accepted: list[int] = []
    rejections: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reason = SCREENERS[args.model](row["canonical_smiles"], state)
        if reason is None:
            accepted.append(index)
        else:
            rejections.append(
                {
                    "panel_index": index,
                    "molecule_hash": row["molecule_hash"],
                    "reason": reason,
                }
            )
    report = {
        "schema_version": 1,
        "status": "ok",
        "execution": "screen_only_no_model_forward",
        "model": args.model,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejections),
        "coverage_fraction": len(accepted) / max(1, len(rows)),
        "accepted_indices": accepted,
        "accepted_identity_sha256": sha256_lines(
            rows[index]["molecule_hash"] for index in accepted
        ),
        "rejections": rejections,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "accepted_indices" and key != "rejections"}, sort_keys=True))


def require_one_gpu() -> Any:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-GPU contract violated: visible GPUs={torch.cuda.device_count()}"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    return torch


def load_molai(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    sys.path.insert(0, "/opt/molai")
    from encoder import FrozenMolAI, prepare_smiles

    model = FrozenMolAI(device="cuda")

    def encode(values: list[str]) -> np.ndarray:
        prepared = prepare_smiles(values, allow_unknown_zero=False)
        return model.encode_prepared(prepared).numpy()

    return 512, encode, {"official_preprocessing": "canonical_nonisomeric_ClX_BrY"}


def materialize_readonly_tensors(model: Any) -> None:
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.detach().clone(memory_format=torch.preserve_format)
        for buffer in model.buffers():
            buffer.data = buffer.detach().clone(memory_format=torch.preserve_format)


def load_molformer(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_dir = "/opt/molformer/model"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        model_dir,
        deterministic_eval=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    materialize_readonly_tensors(model)
    model.to("cuda").requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        inputs = tokenizer(
            values,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(inputs["input_ids"].shape[1]) > 202:
            raise RuntimeError("MolFormer encode received a pre-screen length violation")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs).pooler_output
        return output.detach().float().cpu().numpy()

    return 768, encode, {"model_revision": "7b12d946c181a37f6012b9dc3b002275de070314"}


def load_smi_ted(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    import torch

    sys.path.insert(0, "/opt/smi-ted/model")
    from load import load_smi_ted as load_model

    model = load_model(
        folder="/opt/smi-ted/model",
        ckpt_filename="smi-ted-Light_40.pt",
        vocab_filename="bert_vocab_curated.txt",
    )
    model.eval().requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        with torch.inference_mode():
            output = model.encode(
                values,
                batch_size=len(values),
                return_torch=True,
            )
        return output.detach().float().cpu().numpy()

    return 768, encode, {
        "model_revision": "414c3ea0a8603ef49d1c5bb3db336e09877c01ce",
        "official_preprocessing": "canonical_nonisomeric",
        "warmup_required": True,
    }


def load_molclr(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    sys.path.insert(0, "/opt/molclr")
    from encoder import MolCLRGinEncoder

    model = MolCLRGinEncoder(device="cuda")

    def encode(values: list[str]) -> np.ndarray:
        return model.encode(values, batch_size=len(values)).numpy()

    return 512, encode, {"representation": "preprojection_graph_vector"}


def load_kermt(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    from rdkit import Chem
    from task.extract_embeddings import (
        extract_embeddings_batch,
        load_encoder_from_checkpoint,
        load_projection_from_checkpoint,
    )

    checkpoint = "/opt/kermt/model/kermt_contrastive_v2.0.pt"
    encoder, readout, model_args = load_encoder_from_checkpoint(checkpoint, device="cuda")
    if model_args.use_cuikmolmaker_featurization:
        raise RuntimeError("Released KERMT checkpoint requested unavailable cuik_molmaker")
    projection = load_projection_from_checkpoint(checkpoint, model_args, device="cuda")
    encoder.eval().requires_grad_(False)
    projection.eval().requires_grad_(False)

    def encode(values: list[str]) -> np.ndarray:
        canonical = []
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"KERMT could not parse {value!r}")
            canonical.append(Chem.MolToSmiles(molecule))
        output, validity = extract_embeddings_batch(
            encoder=encoder,
            readout=readout,
            smiles_batch=canonical,
            args=model_args,
            device="cuda",
            projection_extractor=projection,
        )
        if not all(validity):
            raise RuntimeError("KERMT rejected a pre-screened canonical molecule")
        return np.asarray(output["projected"], dtype=np.float32)

    return 512, encode, {"representation": "cmim_projected_mean_latent"}


def load_morgan(batch_size: int) -> tuple[int, Callable[[list[str]], np.ndarray], dict[str, Any]]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def encode(values: list[str]) -> np.ndarray:
        rows = []
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                raise ValueError(f"Morgan could not parse {value!r}")
            rows.append(generator.GetFingerprintAsNumPy(molecule))
        return np.asarray(rows, dtype=np.float32)

    return 2048, encode, {"radius": 2, "fp_size": 2048}


LOADERS = {
    "molai": load_molai,
    "molformer": load_molformer,
    "smi_ted": load_smi_ted,
    "molclr_gin": load_molclr,
    "kermt_v2": load_kermt,
    "morgan": load_morgan,
}


def run_encode(args: argparse.Namespace, rows: list[dict[str, str]]) -> None:
    if args.metadata is None:
        raise ValueError("--metadata is required for encode mode")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.suffix != ".npy":
        raise ValueError("Embedding output must end in .npy")
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("Refusing to overwrite an embedding or metadata output")
    if not rows:
        raise ValueError("Cannot encode an empty panel")

    protocol = load_protocol()
    expected = protocol["comparators"][args.model]
    if int(expected["batch_size"]) != args.batch_size:
        raise RuntimeError(
            f"Batch size differs from frozen protocol for {args.model}: "
            f"{args.batch_size} != {expected['batch_size']}"
        )

    started = time.perf_counter()
    torch = require_one_gpu() if args.model in NEURAL_MODELS else None
    dimension, encode_batch, implementation = LOADERS[args.model](args.batch_size)
    if dimension != int(expected["dimension"]):
        raise RuntimeError("Adapter dimension differs from frozen protocol")

    fixture = [row["canonical_smiles"] for row in rows[: min(2, len(rows))]]
    _ = encode_batch(fixture)
    first = np.asarray(encode_batch(fixture), dtype=np.float32)
    second = np.asarray(encode_batch(fixture), dtype=np.float32)
    if not np.array_equal(first, second):
        delta = float(np.max(np.abs(first - second)))
        raise RuntimeError(f"Fixed-batch deterministic repeat failed; max delta={delta}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output exists: {temporary}")
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), dimension),
    )
    try:
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            values = [row["canonical_smiles"] for row in rows[start:stop]]
            batch = np.asarray(encode_batch(values), dtype=np.float32)
            if batch.shape != (stop - start, dimension):
                raise RuntimeError(
                    f"Unexpected {args.model} output at {start}:{stop}: {batch.shape}"
                )
            if not np.isfinite(batch).all():
                raise RuntimeError(f"Non-finite {args.model} output at {start}:{stop}")
            norms = np.linalg.norm(batch.astype(np.float64), axis=1)
            if np.any(norms <= 1.0e-12):
                raise RuntimeError(f"Zero-norm {args.model} output at {start}:{stop}")
            matrix[start:stop] = batch
        matrix.flush()
        del matrix
        temporary.replace(args.output)
    except Exception:
        del matrix
        temporary.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "status": "ok",
        "execution": "inference_only",
        "model": args.model,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "ordered_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in rows
        ),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "dimension": dimension,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "fixed_batch_deterministic_repeat": True,
        "wall_seconds_model_load_warmup_and_export": elapsed,
        "rows_per_second_including_load_warmup_and_export": len(rows) / elapsed,
        "output_bytes": args.output.stat().st_size,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch is not None else None
        ),
        "gpu_name": torch.cuda.get_device_name(0) if torch is not None else None,
        "visible_gpu_count": torch.cuda.device_count() if torch is not None else 0,
        "implementation": implementation,
        "python": platform.python_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    atomic_write_json(args.metadata, report)
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    args = parse_args()
    rows = read_panel_tsv(args.input)
    if args.mode == "screen":
        run_screen(args, rows)
    else:
        run_encode(args, rows)


if __name__ == "__main__":
    main()
