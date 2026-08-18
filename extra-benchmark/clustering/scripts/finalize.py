#!/usr/bin/env python3
"""Record runtime versions and seal a hash manifest after verification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import duckdb
import matplotlib
import numpy
import pandas
import pyarrow
from rdkit import __version__ as rdkit_version
import scipy
import sklearn
import torch

from benchmark_io import BENCHMARK_DIR, atomic_write_json, atomic_write_text, load_json, protocol_digest, load_protocol, sha256_file


def main() -> None:
    verification = load_json(BENCHMARK_DIR / "audit" / "verification.json")
    if verification.get("status") != "ok":
        raise RuntimeError("Cannot finalize a failed benchmark verification")
    versions = {
        "schema_version": 1, "python": platform.python_version(),
        "numpy": numpy.__version__, "pandas": pandas.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "pyarrow": pyarrow.__version__,
        "duckdb": duckdb.__version__, "rdkit": rdkit_version,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(), "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    version_path = BENCHMARK_DIR / "audit" / "software_versions.json"
    atomic_write_json(version_path, versions)
    excluded_roots = {BENCHMARK_DIR / "tmp", BENCHMARK_DIR / "logs"}
    manifest_path = BENCHMARK_DIR / "outputs" / "SHA256SUMS"
    complete_path = BENCHMARK_DIR / "state" / "COMPLETE.json"
    paths = []
    for path in BENCHMARK_DIR.rglob("*"):
        if not path.is_file() or path.is_symlink() or path in (manifest_path, complete_path):
            continue
        if any(root == path or root in path.parents for root in excluded_roots):
            continue
        if path.name == ".gitkeep" or path.name.endswith(".pyc") or "__pycache__" in path.parts:
            continue
        paths.append(path)
    lines = [f"{sha256_file(path)}  {path.relative_to(BENCHMARK_DIR)}" for path in sorted(paths)]
    atomic_write_text(manifest_path, "\n".join(lines) + "\n")
    protocol = load_protocol()
    complete = {
        "schema_version": 1, "status": "complete", "inference_only": True,
        "protocol_sha256": protocol_digest(protocol),
        "verification_sha256": sha256_file(BENCHMARK_DIR / "audit" / "verification.json"),
        "software_versions_sha256": sha256_file(version_path),
        "sha256_manifest": str(manifest_path), "sha256_manifest_sha256": sha256_file(manifest_path),
        "manifest_files": len(lines), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(complete_path, complete)
    print(json.dumps(complete, sort_keys=True))


if __name__ == "__main__":
    main()
