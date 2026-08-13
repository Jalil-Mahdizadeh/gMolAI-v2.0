# KERMT v2 native batch-dependence audit

## Finding

The frozen native KERMT v2 cMIM projected-mean embedding is not invariant to
batch composition/size in this benchmark implementation. This is a no-training
inference audit; no checkpoint, model source, benchmark embedding or prior
scientific result was changed.

The definitive 49,844-molecule execution, Slurm job 1230738, compared each
condition with the batch-64 output:

| Batch size | Minimum cosine vs. batch 64 | Maximum relative-L2 vs. batch 64 | Gate |
|---:|---:|---:|:---:|
| 64 | 1.0000000000 | 0.0000000000 | Reference |
| 128 | 0.9877585265 | 0.1597171261 | Fail |
| 256 | 0.9858886827 | 0.1821650703 | Fail |
| 512 | 0.9823950721 | 0.1973206413 | Fail |

These differences exceed the unchanged cross-batch gates (minimum cosine
0.9999 and maximum relative-L2 delta 0.005) by a wide margin. Fixed-batch
repeatability nevertheless passed exactly, confirming a deterministic
batch-composition effect rather than run-to-run instability.

## Root cause

The frozen KERMT graph builder in
`/opt/kermt/source/kermt/data/molgraph.py` sets `max_num_bonds` independently
for each batch and pads adjacency rows to that batch-local width with index 0.
The encoder does not keep the nominal padding state neutral: measured dummy-row
norms across its four outputs were 16.60–23.09 for an affected batch.

A controlled exact-panel comparison used rows 408–415 alone (adjacency width 3)
and the same rows as the final eight members of rows 384–415 (width 4). Native
outputs differed by maximum relative-L2 0.1488698272 with minimum cosine
0.9911860787. Forcing both graph batches to width 4 solely as a diagnostic
reduced maximum relative-L2 to 0.0001817552 and raised minimum cosine to
0.9999999835. This isolates variable adjacency padding as the mechanism.

## Benchmark treatment

The prior test-partition and MoleculeNet/HIV KERMT extractions used the native
path at fixed nominal batch size 64. Those ordered runs remain reproducible
artifacts, but the molecular representation is conditional on batching and
must not be described as molecule-only invariant.

This speed benchmark intentionally does not patch KERMT or regenerate any
scientific embedding. It retains the native path, records computational
throughput at batch sizes 64, 128, 256 and 512, preserves every failed integrity
value and marks KERMT as non-equivalent in tables and machine-readable outputs.
Its points must not be used to claim representation-equivalent batch scaling.
