#!/usr/bin/env python3
"""Render the measured Step 03 result as a concise Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import atomic_write_text, load_json


def rate(summary: dict, name: str) -> tuple[float, float, float]:
    metric = summary["metrics"][name]
    return (
        float(metric["value"]),
        float(metric["ci95_lower"]),
        float(metric["ci95_upper"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.step_root.resolve()
    summary = load_json(root / "outputs" / "benchmark_summary.json")
    raw = rate(summary, "raw_proposals_per_second")
    raw_smiles = rate(summary, "raw_smiles_per_second")
    valid = rate(summary, "valid_unique_molecules_per_second")
    policy = rate(summary, "policy_unique_molecules_per_second")
    counts = summary["counts"]
    timings = summary["timings_seconds"]
    execution = summary["execution"]
    metrics = summary["metrics"]
    peak_gib = execution["peak_cuda_memory_allocated_bytes"] / 2**30
    valid_fraction = metrics["rdkit_valid_fraction"]["value"] * 100.0
    unique_fraction = metrics["rdkit_unique_fraction_of_raw"]["value"] * 100.0
    policy_fraction = metrics["release_policy_accepted_fraction"]["value"] * 100.0
    text = f"""# Step 03 results

The released stochastic decoder generated **{raw[0]:,.1f} raw proposals/s**
(batch-bootstrap 95% interval {raw[1]:,.1f}–{raw[2]:,.1f}) on one
**{execution['gpu_name']}**. After byte-token decoding, the rate was
{raw_smiles[0]:,.1f} raw SMILES/s ({raw_smiles[1]:,.1f}–{raw_smiles[2]:,.1f}).

The requested usable-output rate was **{valid[0]:,.1f} per-seed unique,
RDKit-valid molecules/s** ({valid[1]:,.1f}–{valid[2]:,.1f}). This numerator was
{counts['per_seed_unique_rdkit_valid_molecules']:,} first-occurrence identities
from {counts['raw_proposals']:,} proposal slots. {valid_fraction:.2f}% of raw
slots were RDKit-valid and {unique_fraction:.2f}% yielded a first unique valid
identity within its conditioning seed.

As a stricter secondary result, the released encoder policy accepted
{policy_fraction:.2f}% of raw slots and produced {policy[0]:,.1f} per-seed
policy-unique molecules/s ({policy[1]:,.1f}–{policy[2]:,.1f}).

## Measured workload

- 100 reproducibly sampled `released_hybrid_w3` molecular embeddings.
- Exactly 1,000 stochastic draws per embedding: 100,000 raw slots total.
- Released sampling settings: temperature 1.2, top-p 0.995, 128-byte maximum.
- Query batch size 2, yielding 50 measured batches on exactly one visible GPU.
- One full-shape warm-up batch was excluded.

## Timing audit

| Component | Seconds | Headline inclusion |
|---|---:|---|
| GPU generation + transfers | {timings['generation']:.3f} | both rates |
| Token-to-SMILES decode | {timings['token_decode']:.3f} | valid-unique only |
| RDKit validation/canonicalization/dedup | {timings['rdkit_validation']:.3f} | valid-unique only |
| Release-policy pass | {timings['release_policy']:.3f} | secondary policy rate only |
| Serialization | {timings['serialization_excluded']:.3f} | excluded |
| Warm-up | {timings['warmup_excluded']:.3f} | excluded |
| Model load | {timings['model_load']:.3f} | excluded |

Peak measured CUDA allocation was {peak_gib:.2f} GiB. The observed measured-loop
wall time, including serialization and Python bookkeeping, was
{timings['measured_observed_wall_including_serialization']:.3f} s.

## Interpretation

The raw rate measures decoder proposal slots, whereas the valid-unique rate
discounts malformed and repeated outputs and includes the corresponding chemistry
work. Uniqueness is intentionally reset per conditioning seed. The 95% intervals
resample the 50 batches from this single run; they quantify panel/batch
heterogeneity, not run-to-run hardware variance.

## Artifacts

- `figures/decoder_throughput.png` and `.svg`: headline comparison.
- `figures/batch_throughput_trace.png` and `.svg`: run-order stability.
- `figures/per_seed_valid_unique_yield.png` and `.svg`: validity/yield spread.
- `outputs/plot-data/`: exact source CSVs used for every figure.
- `outputs/tables/`: summary, per-batch, and per-seed tables.
- `outputs/raw/proposals.parquet`: all 100,000 proposal-level records.
- `outputs/verification.json`: integrity checks.
"""
    atomic_write_text(root / "RESULTS.md", text)
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main()
