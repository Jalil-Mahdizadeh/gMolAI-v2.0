#!/usr/bin/env python3
"""Hash-pinned wrapper for the frozen released-hybrid-w3 gMolAI exporter."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import benchmark_io


UPSTREAM = (
    benchmark_io.REPOSITORY_ROOT
    / "extra-benchmark/tdc-admet/scripts/gmolai_adapter.py"
)
EXPECTED_SHA256 = "fe2c310b037d52ae72b3a3139c62684a7497502858df24198123ad060c249af5"


def compatible_protocol():
    protocol = copy.deepcopy(benchmark_io.load_protocol())
    checkpoint_path = Path(protocol["gmolai"]["checkpoint"]["path"])
    run_dir = checkpoint_path.parent.parent
    checkpoint = copy.deepcopy(protocol["gmolai"]["checkpoint"])
    checkpoint["run_dir"] = str(run_dir)
    checkpoint["name"] = str(checkpoint_path.relative_to(run_dir))
    protocol["gmolai"]["checkpoint"] = checkpoint
    expected = copy.deepcopy(protocol["models"]["gmolai"])
    expected["representation"] = protocol["gmolai"]["representation"]
    protocol["comparators"] = {"models": {"gmolai": expected}}
    return protocol


def main() -> None:
    if benchmark_io.sha256_file(UPSTREAM) != EXPECTED_SHA256:
        raise RuntimeError("Frozen upstream gMolAI adapter hash changed")
    spec = importlib.util.spec_from_file_location("_frozen_lbvs_gmolai", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen adapter from {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_protocol = compatible_protocol
    module.main()


if __name__ == "__main__":
    main()

