# CSV molecular embedding inference

This folder is a self-contained entry point for generating the promoted gMolAI
384-dimensional molecular vector from ordinary SMILES rows. It does not require
the pretraining datasets, descriptor values, graph manifest, or graph shards.

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
available (otherwise CPU), and write:

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

## Model identity

- checkpoint SHA-256:
  `02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e`;
- calibrator SHA-256:
  `5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b`;
- vector definition: standardized raw graph-256 concatenated with standardized
  mean-node-128, with the mean-node block weighted by 3.

The metadata sidecar records all model/data/output hashes, row counts,
rejection reasons, dimensions, runtime versions, and execution parameters.

Inference is deterministic for a fixed runtime, device, and batching setup.
Different CPU/GPU scatter-reduction orders or batch shapes can change the last
floating-point bits; consumers should compare numerically rather than requiring
byte-identical CSVs across hardware.
