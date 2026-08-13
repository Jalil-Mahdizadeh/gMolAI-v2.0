# Common locked-test encoding-speed benchmark

This directory contains the completed replacement throughput comparison for
the seven frozen encoders used in the representation benchmarks:

- gMolAI seed 42, step 10,000, using the repository-canonical optimized
  inference backend;
- Morgan radius-2, 2,048-bit fingerprints;
- MolAI epoch 6;
- MoLFormer;
- SMI-TED-Light;
- MolCLR-GIN; and
- KERMT v2.

## Completed execution

Slurm job 1230738 ran all models sequentially on Arrhenius node n54 in one
allocation with one NVIDIA GH200 120GB GPU and 48 CPUs. Every model processed
the same ordered 49,844-molecule all-model common panel once at batch sizes 64,
128, 256 and 512, following one untimed warm-up batch per condition. Morgan is
a CPU baseline and used the CPU resources of that same allocation.

gMolAI used `OptimizedSmilesEncoder` with the speed-optimization-selected 48
RDKit preprocessing workers. Every container received the same 48-CPU ceiling,
one-thread BLAS/OpenMP settings and, for neural models, the same single visible
GPU. Competitors retained their frozen native inference paths.

At batch sizes 64, 128, 256 and 512, optimized gMolAI encoded 13,040.69,
22,901.45, 40,068.75 and 58,330.38 molecules/s, respectively. That corresponds
to 1.21x, 2.12x, 3.68x and 5.36x the same-condition Morgan throughput, and
3.59x, 4.96x, 7.76x and 10.04x the fastest other representation-equivalent
neural encoder at each batch size. See `RESULTS.md` for the complete
seven-model table and interpretation limits.

## Timing and integrity

The primary timer spans canonical SMILES already in RAM through complete
ordered FP32 vectors in host RAM. It includes output allocation, required
parsing/tokenization/2D graph construction, device transfers, frozen forward
inference and host materialization. It excludes container/model loading,
gMolAI worker-pool startup, input-file I/O, warm-up, validation, hashing and
disk serialization.

Each batch-size output was compared with the batch-64 reference using frozen
scale-aware cosine and relative-L2 gates. Every model except KERMT passed.
KERMT's native batch-dependent adjacency padding produced materially different
vectors, so its four points are retained only as computational throughput and
are explicitly excluded from representation-equivalent scaling claims. See
`KERMT_BATCH_DEPENDENCE.md`.

After the completed execution, one generated KERMT footnote was corrected from
the obsolete phrase “three points” to the count-neutral “batch-size points”;
batch 512 had already been measured and present in every table and raw record.
The correction changed no timing, vector, integrity value, protocol condition,
or scientific interpretation. The checksum ledger was regenerated after this
documentation-only fix.

This is a deliberately bounded one-pass systems measurement. Values are
descriptive point measurements without run-to-run uncertainty or confidence
intervals; it is not an MLPerf benchmark.

## Artifacts

- `RESULTS.md`: concise human-readable results.
- `outputs/speed_results.{csv,json}`: complete compact results.
- `outputs/batch_latencies.csv` and `outputs/raw/*.json`: native batch
  latencies and per-model timing, integrity and provenance records.
- `outputs/throughput_by_batch_size.{png,pdf,svg}`: rendered comparison.
- `state/preflight.json` and `state/COMPLETE.json`: execution attestations.
- `outputs/SHA256SUMS`: checksum ledger for the retained result set.
- `protocol.json` and `PROTOCOL.md`: machine-readable and narrative frozen
  protocols.

Scientific embedding matrices are intentionally not persisted here.

## Reproduce on Arrhenius

From the repository root, submit a fresh single-GPU allocation:

```bash
sbatch --account=naiss2025-3-10-gpu extra-benchmark/speed/run_speed.sbatch
```

Inside an already allocated single-GPU interactive job, run the same entry
point directly:

```bash
bash extra-benchmark/speed/run_speed.sbatch
```

The script requests one GPU, 48 CPUs and two hours when submitted through
Slurm. It refuses mismatched inputs, sources, SIFs, batch sizes or GPU counts
before measuring.
