#!/usr/bin/env python3
"""Atomically append a speed-benchmark stage transition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--status", required=True, choices=("running", "complete", "failed")
    )
    parser.add_argument("--message", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = BENCHMARK_DIR / "state" / "status.json"
    state = load_json(path) if path.is_file() else {"schema_version": 1, "history": []}
    transition = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "status": args.status,
        "message": args.message,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
    }
    state["current"] = transition
    state.setdefault("history", []).append(transition)
    atomic_write_json(path, state)
    print(json.dumps(transition, sort_keys=True))


if __name__ == "__main__":
    main()

