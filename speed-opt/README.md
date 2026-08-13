# gMolAI inference speed optimization

> Historical optimization record. The selected implementation has now been
> promoted into `src/gmolai_retrain/fast_graph.py` and
> `src/gmolai_retrain/fast_inference.py`, and is the default backend of
> `inference/generate_embeddings.py`. The subsequent definitive seven-model
> rerun at batch sizes 64, 128, 256 and 512 is recorded in
> `extra-benchmark/speed/RESULTS.md` and
> `extra-benchmark/speed/outputs/speed_results.{csv,json}`.

This directory records the isolated, inference-only optimization phase for the
promoted seed-42 step-10,000 gMolAI encoder. During that phase no file outside
`speed-opt/` was changed, no training was run, and neither the checkpoint nor
the calibrator was modified.

## Outcome

On the complete 49,844-molecule all-model common locked-test panel, the
optimized path encoded canonical SMILES already in RAM to ordered FP32 vectors
in host RAM at **29,930 molecules/s** (1.665 s). An independent repeat gave
**31,236 molecules/s** (1.596 s). The earlier authoritative batch-64 path was
405.53 molecules/s (122.91 s), so the direct historical comparison is **73.8x**
faster for the fully audited run.

This is primarily a software-pipeline result, not a change to gMolAI:

- the original path spent about 91% of staged time in serial RDKit graph
  feature construction;
- a reduced RDKit feature factory computes exactly the donor/acceptor families
  used by gMolAI instead of all unused BaseFeatures families;
- NumPy arrays are packed directly, avoiding one PyG `Data` object per molecule;
- 48 CPU workers overlap graph construction with the single GPU;
- the four GINE layers are expressed as their equivalent gather, edge projection,
  ReLU and `index_add_` operations, reducing PyG dispatch/memory overhead;
- batch size 192 was selected from isolated tuning on this GH200 node.

The existing `gmolai-pyg-25.09-arm64.sif` already supplies the required RDKit,
PyTorch and PyG runtime. A second SIF was therefore unnecessary.

## Scientific equivalence

The clean full audit in `outputs/final_benchmark_clean.json` established:

- exact equality of node features, edge indices and edge features for all
  49,844 molecules (1,160,819 atoms and 2,487,686 directed edges);
- bitwise equality of all 49,844 optimized 384-D embeddings to the
  authoritative implementation when both use the same batch-192 boundaries;
- against the historical batch-64 boundaries: minimum cosine 0.999999533 and
  maximum relative-L2 difference 0.000969514, inside the benchmark's frozen
  0.9999 / 0.005 stability gates;
- two independent optimized full-panel executions and the production CLI
  produced the same matrix SHA-256:
  `66efc5aa10bafe783ca9552d54631d433e144656724295bf881134e24f993442`.

Floating-point scatter reductions can change at different batch boundaries;
therefore bitwise identity is correctly claimed at identical boundaries, while
the existing scale-aware gate is used across batch sizes.

## Production use

The input must be a TSV containing canonical SMILES, which matches the measured
benchmark boundary. Run from the repository root:

```bash
speed-opt/run_encode_example.sh \
  --input path/to/input.tsv \
  --smiles-column canonical_smiles \
  --output speed-opt/artifacts/my_embeddings.npy \
  --batch-size 192 \
  --workers 48
```

The output is an ordered `(rows, 384)` NumPy float32 matrix plus a JSON metadata
sidecar containing the input, checkpoint, calibrator and matrix hashes. The CLI
refuses to overwrite existing outputs and verifies exactly one visible GPU.
Use fewer workers when the Slurm allocation grants fewer CPUs.

To repeat the full fail-closed audit, choose unused output paths in
`scripts/final_benchmark.py`, or use `run_final.sh` when its default outputs do
not yet exist. The full audit is deliberately slow because it recomputes every
reference graph feature and embedding; those validation passes are excluded
from the one-pass optimized timer.

## Files

- `scripts/fast_graph.py`: exact reduced RDKit featurization and direct packing.
- `scripts/model_core.py`: frozen artifact loading and equivalent GPU cores.
- `scripts/encode_smiles.py`: reusable optimized encoder.
- `scripts/final_benchmark.py`: full-panel validation and timing harness.
- `scripts/tune.py`: batch/core/worker tuning harness.
- `outputs/final_benchmark_clean.json`: authoritative full audit.
- `outputs/replicate_benchmark.json`: independent timing replicate.
- `outputs/cli_embeddings.metadata.json`: production-CLI reproduction record.
- `artifacts/optimized_embeddings_clean.npy`: retained full validation matrix
  (ignored by Git because it is approximately 73 MiB and fully reproducible).

Exploratory records are retained under `outputs/tuning-*.json`. The
TorchInductor candidate was slower and introduced small additional numerical
drift, so it was rejected. No new SIF is part of the selected solution.
