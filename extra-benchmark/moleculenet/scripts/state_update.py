#!/usr/bin/env python3
"""Atomically record endpoint-benchmark pipeline progress."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    path = BENCHMARK_DIR / "state" / "status.json"
    history = []
    if path.exists():
        history = list(load_json(path).get("history", []))
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "status": args.status,
        "message": args.message,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    history.append(event)
    atomic_write_json(path, {**event, "history": history})


if __name__ == "__main__":
    main()
