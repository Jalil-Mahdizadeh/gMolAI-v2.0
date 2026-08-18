#!/usr/bin/env python3
"""Materialize a permitted larger SHA-ordered QMugs attempt prefix."""

from __future__ import annotations

import argparse

from benchmark_io import BENCHMARK_DIR, load_protocol, panel_columns, read_panel_tsv, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, type=int)
    args = parser.parse_args()
    protocol = load_protocol()
    rules = protocol["common_support"]
    initial, increment, maximum = (int(rules[key]) for key in (
        "qmugs_initial_attempt", "qmugs_increment", "qmugs_maximum_attempt"
    ))
    if args.size < initial or args.size > maximum or (args.size - initial) % increment:
        raise ValueError("Requested prefix violates frozen expansion sequence")
    source = BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_eligible.tsv"
    rows = read_panel_tsv(source)
    if len(rows) < args.size:
        raise RuntimeError("Not enough eligible QMugs identities")
    selected = [dict(row, panel_index=index) for index, row in enumerate(rows[:args.size])]
    output = BENCHMARK_DIR / "inputs" / "prepared" / f"qmugs_attempt_{args.size:06d}.tsv"
    write_tsv(output, selected, panel_columns(source))
    print(output)


if __name__ == "__main__":
    main()

