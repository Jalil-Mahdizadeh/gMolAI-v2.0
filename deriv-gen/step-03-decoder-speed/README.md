# Step 03 — released decoder speed

This directory benchmarks the released gMolAI conditional SMILES decoder on one
GPU. The measured panel contains 100 reproducibly sampled molecular embeddings,
and each embedding receives exactly 1,000 stochastic decoder draws.

The two headline quantities are:

1. **Raw proposal throughput:** proposal slots divided by measured GPU generation
   time, including condition transfer and token transfer back to the host.
2. **Valid unique molecule throughput:** the sum of per-seed, first-occurrence
   RDKit-valid canonical identities divided by generation, token decoding, and
   RDKit validation time.

Model loading, release-artifact hashing, input preparation, one full-shape warm-up
batch, plotting, serialization, and report generation are excluded from those
denominators. This makes the timed region explicit while retaining component and
observed-wall timings for audit.

See [PROTOCOL.md](PROTOCOL.md) for the frozen definitions and [RESULTS.md](RESULTS.md)
for the measured result. Plot source data are retained under `outputs/plot-data/`,
and the 100,000 proposal-level records are retained as
`outputs/raw/proposals.parquet`.

## Reproduction

From the repository root, inside a one-GPU allocation:

```bash
bash deriv-gen/step-03-decoder-speed/run_benchmark.sh --overwrite
```

The explicit flag replaces only this directory's sealed benchmark products.
Without it, the launcher refuses to overwrite an existing run.
The launcher mounts the repository read-only at `/repo`, mounts only this study
directory read-write at `/step`, and redirects the container home, caches, and
temporary directory into `state/` here.
