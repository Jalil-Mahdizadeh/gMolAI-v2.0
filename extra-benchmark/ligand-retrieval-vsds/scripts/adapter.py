#!/usr/bin/env python3
"""Hash-pinned wrapper around the manuscript's frozen comparator adapter."""

from __future__ import annotations

import copy
import importlib.util

import benchmark_io


UPSTREAM = (
    benchmark_io.REPOSITORY_ROOT
    / "extra-benchmark/test-partition/scripts/adapter.py"
)
EXPECTED_SHA256 = "7c87bfa567564765da7aca9ae417ef3b1f4c7252c7922b6ab4db2bb25c59a8e4"


def compatible_protocol():
    protocol = copy.deepcopy(benchmark_io.load_protocol())
    protocol["comparators"] = {
        model: copy.deepcopy(protocol["models"][model])
        for model in protocol["models"]["primary_order"]
        if model != "gmolai"
    }
    return protocol


def main() -> None:
    if benchmark_io.sha256_file(UPSTREAM) != EXPECTED_SHA256:
        raise RuntimeError("Frozen upstream comparator adapter hash changed")
    spec = importlib.util.spec_from_file_location("_frozen_lbvs_adapter", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen adapter from {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_protocol = compatible_protocol
    module.main()


if __name__ == "__main__":
    main()

