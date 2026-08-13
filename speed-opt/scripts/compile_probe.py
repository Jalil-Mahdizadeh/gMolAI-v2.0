#!/usr/bin/env python3
"""Probe TorchInductor on the exact manual-scatter gMolAI inference core."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SPEED_OPT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fast_graph import pack_smiles_task
from model_core import ManualScatterRawCore, load_frozen_bundle, packed_to_device
from tune import compare, encode_packed, read_smiles


def main() -> None:
    cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")).resolve()
    if SPEED_OPT_DIR.resolve() not in cache.parents:
        raise RuntimeError("TORCHINDUCTOR_CACHE_DIR must be inside speed-opt")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    values = read_smiles(4096)
    bundle = load_frozen_bundle()
    manual = ManualScatterRawCore(bundle.model).eval()
    compiled = torch.compile(
        manual,
        dynamic=True,
        fullgraph=False,
        mode="default",
    )
    results = []
    for batch_size in (64, 256, 1024):
        packed = pack_smiles_task((0, values[:batch_size]))
        tensors = packed_to_device(packed, bundle.device)
        with torch.inference_mode():
            expected = manual(*tensors).detach().float().cpu().numpy()
            for _ in range(2):
                observed = compiled(*tensors).detach().float().cpu().numpy()
        torch.cuda.synchronize()
        qualification = compare(expected, observed)
        matrix, elapsed, peak = encode_packed(
            values, bundle, batch_size, compiled
        )
        results.append(
            {
                "batch_size": batch_size,
                "qualification_raw": qualification,
                "seconds": elapsed,
                "rows_per_second": len(values) / elapsed,
                "peak_gpu_memory_bytes": peak,
            }
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    destination = SPEED_OPT_DIR / "outputs/tuning-compile.json"
    destination.write_text(
        json.dumps({"results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
