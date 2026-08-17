"""Narrow compatibility shim for the release encoder's effective backend label.

The public CLI flag is named ``optimized`` while release metadata deliberately
records the concrete implementation as ``optimized_gine_v1``.  Only the two
validators need the effective label; preparation and reporting continue to
record the user-facing CLI option from the frozen JSON configuration.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path


if Path(sys.argv[0]).name in {"analyze_results.py", "verify_results.py"}:
    import common

    _load_config = common.load_config

    def load_config_with_effective_backend(root: Path) -> dict:
        config = copy.deepcopy(_load_config(root))
        if config.get("reencoding", {}).get("backend") == "optimized":
            config["reencoding"]["backend"] = "optimized_gine_v1"
        return config

    common.load_config = load_config_with_effective_backend
