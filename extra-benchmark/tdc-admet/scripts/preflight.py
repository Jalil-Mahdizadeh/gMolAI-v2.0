#!/usr/bin/env python3
"""Verify all frozen inputs and the single-GPU execution contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import platform
import subprocess

import numpy as np
from rdkit import __version__ as rdkit_version
import scipy
import sklearn
import torch

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    load_protocol,
    protocol_digest,
    require_hash,
    sha256_file,
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def main() -> None:
    protocol = load_protocol()
    integrity: list[dict[str, str]] = []
    for section, item in (
        ("config", protocol["gmolai"]["config"]),
        ("training_plan", protocol["gmolai"]["training_plan"]),
        ("calibrator", protocol["gmolai"]["calibrator"]),
        ("qualification_protocol", protocol["comparators"]["qualification_protocol"]),
        ("qualification_complete", protocol["comparators"]["qualification_complete"]),
        ("qualified_adapter", protocol["comparators"]["adapter"]),
    ):
        path = REPOSITORY_ROOT / item["path"]
        integrity.append(
            {"name": section, "path": str(path), "sha256": require_hash(path, item["sha256"])}
        )
    overlap_audit = protocol["selection_conditioning"]["audit"]
    overlap_path = REPOSITORY_ROOT / overlap_audit["path"]
    integrity.append(
        {
            "name": "prior_development_overlap_audit",
            "path": str(overlap_path),
            "sha256": require_hash(overlap_path, overlap_audit["sha256"]),
        }
    )
    checkpoint = (
        REPOSITORY_ROOT
        / protocol["gmolai"]["checkpoint"]["run_dir"]
        / protocol["gmolai"]["checkpoint"]["name"]
    )
    integrity.append(
        {
            "name": "gmolai_checkpoint",
            "path": str(checkpoint),
            "sha256": require_hash(checkpoint, protocol["gmolai"]["checkpoint"]["sha256"]),
        }
    )
    containers = []
    seen = set()
    for spec in protocol["comparators"]["models"].values():
        if spec["container"] in seen:
            continue
        seen.add(spec["container"])
        containers.append(
            {
                "path": spec["container"],
                "sha256": require_hash(spec["container"], spec["container_sha256"]),
            }
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-GPU contract violated: visible GPUs={torch.cuda.device_count()}"
        )
    report = {
        "schema_version": 1,
        "status": "pass",
        "protocol_sha256": protocol_digest(protocol),
        "protocol_file_sha256": sha256_file(BENCHMARK_DIR / "protocol.json"),
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_status": git("status", "--short"),
        },
        "integrity": integrity,
        "containers": containers,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit_version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    atomic_write_json(BENCHMARK_DIR / "state" / "preflight.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
