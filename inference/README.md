# CSV molecular embedding inference

This folder is a self-contained entry point for generating the promoted gMolAI
384-dimensional molecular vector from ordinary SMILES rows. It does not require
the pretraining datasets, descriptor values, graph manifest, or graph shards.
The repository-canonical execution path is the speed-optimized inference
backend in `src/gmolai_retrain/fast_graph.py` and
`src/gmolai_retrain/fast_inference.py`; the original PyG path remains available
as a reference oracle.

## Contents

```text
inference/
├── generate_embeddings.py
├── model/
│   ├── representation-best.pt
│   ├── representation-calibrator.pt
│   ├── representation_selection.json
│   └── resolved_config.json
├── data/
│   └── example_smiles.csv
└── output/
```

The script verifies hard-coded SHA-256 values before deserializing the model,
then checks checkpoint, selection, feature-schema, configuration, training-plan,
calibrator, and embedding-definition identities. The checkpoint and calibrator
must always be distributed together.

## Run

Use the pinned project container or an environment containing PyTorch,
PyTorch Geometric, RDKit, and NumPy. From the repository root:

```bash
python inference/generate_embeddings.py
```

The defaults read `inference/data/example_smiles.csv`, select CUDA when
available (otherwise CPU), use batch size 192 and up to 48 RDKit workers within
the detected Slurm/CPU-affinity allocation, and write:

- `inference/output/embeddings.csv`;
- `inference/output/embeddings.rejections.csv`;
- `inference/output/embeddings.metadata.json`.

For another dataset:

```bash
python inference/generate_embeddings.py \
  --input /path/to/molecules.csv \
  --smiles-column smiles \
  --id-column molecule_id \
  --output-dir /path/to/output \
  --output-stem my_embeddings \
  --backend optimized \
  --workers auto \
  --batch-size 192 \
  --device cuda
```

Only the SMILES column is required. `--id-column auto` preserves a
`molecule_id` or `id` column when present; otherwise `input_row` provides the
stable join key. Output rows also contain the original SMILES, canonical
isomeric SMILES, molecule SHA-256, and `embedding_000` through
`embedding_383`.

By default, molecules rejected by the training-time policy are written to the
rejection CSV without silently changing accepted-row alignment. Use
`--invalid-policy error` to fail atomically on the first rejection. Existing
outputs are never replaced unless `--overwrite` is supplied.

The training policy accepts 2–256 atom, single-fragment molecules composed of
C, N, O, F, P, S, Cl, Br, I, H, B, or Si. Stereochemistry is retained.

## Inference backends

- `--backend optimized` is the production default. It uses the exact reduced
  donor/acceptor feature factory, direct NumPy graph packing, allocation-bounded
  multiprocess RDKit preprocessing, and the equivalent eval-only GINE core.
- `--backend reference` retains the original BaseFeatures/PyG execution for
  audits and debugging.
- `--backend verify` returns optimized vectors while checking the first
  `--verify-rows` accepted molecules against the reference backend with the
  frozen scale-aware numerical gate.

The optimized backend fails closed if the model architecture or feature schema
does not match the promoted chirality-enabled, position-dimension-zero
contract. Training and gradient computation continue to use the ordinary model
implementation; this backend is inference-only.

The metadata sidecar records the backend and implementation versions, worker
count, batch and node budgets, checkpoint/calibrator/schema identities, runtime
versions, row/rejection counts and output hashes. CPU workers never exceed the
detected Slurm or process-affinity allocation. For very small inputs the same
CLI remains valid; no minimum row count applies.

## Model identity

- checkpoint SHA-256:
  `02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e`;
- calibrator SHA-256:
  `5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b`;
- vector definition: standardized raw graph-256 concatenated with standardized
  mean-node-128, with the mean-node block weighted by 3.

The metadata sidecar records all model/data/output hashes, row counts,
rejection reasons, dimensions, runtime versions, and execution parameters.

The definitive single-GPU comparison reran all seven encoders on the complete
49,844-molecule common locked-test panel at batch sizes 64, 128, 256 and 512.
Optimized gMolAI produced **13,040.69**, **22,901.45**, **40,068.75** and
**58,330.38 molecules/s**, respectively. At batch 512 this was 5.36x Morgan
throughput and 10.04x the fastest other representation-equivalent neural
encoder. See `extra-benchmark/speed/RESULTS.md`; these are descriptive
single-pass measurements on one GH200, not hardware-independent estimates.

Inference is deterministic for a fixed runtime, device, and batching setup.
Different CPU/GPU scatter-reduction orders or batch shapes can change the last
floating-point bits; consumers should compare numerically rather than requiring
byte-identical CSVs across hardware.
