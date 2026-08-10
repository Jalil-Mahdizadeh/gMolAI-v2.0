# gMolAI corrected retraining on Arrhenius

This directory is a resumable molecular-representation package for the combined ZINC and PubChem Feather inputs. It creates an immutable deduplicated/scaffold-split dataset, precomputes versioned graph shards, and trains a deterministic graph encoder on Arrhenius GH200 GPUs with PyTorch DDP. The primary output is one fixed-size molecule vector for downstream ML, clustering, and similarity search.

The original checkpoints are deliberately not loaded: the feature schema, denoising objective, symmetric decoders, pooling, latent width, split, and scaler contract have changed.

## Descriptor contract (resolved)

The source function was supplied after the audit. It filters `Descriptors._descList` by a set and returns `list(res.values())`; therefore the stored order follows `_descList`, not the set literal. The resulting mapping is:

`qed`, `MolWt`, `NumValenceElectrons`, `MaxPartialCharge`, `MinPartialCharge`, `BalabanJ`, `LabuteASA`, `TPSA`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`, `MolLogP`, `MolMR`.

This order was confirmed against the Arrow dtypes and fitted statistics, then directly checked by recomputing all 13 descriptors for batch 0 row 0 from both ZINC and PubChem with RDKit 2026.03.1. All values matched at `rtol=atol=1e-12`; the largest absolute difference was `2.84e-14`. [`configs/descriptors.yaml`](configs/descriptors.yaml) now records the named contract and enables the pipeline. The RDKit version originally used to create the Feather files was not recorded; the SIF pins RDKit 2025.9.3 for reproducibility.

Also review these recorded policy choices in [`configs/retrain.yaml`](configs/retrain.yaml):

- Disconnected molecules are rejected, not desalted.
- B and Si are now accepted in addition to the old element set.
- Stereochemistry is retained when present.
- Canonical positional encoding is disabled.
- Descriptor-conflicting duplicates are excluded; a conflict fraction over 0.1% stops the run.
- Split is 99% train, 0.5% validation, 0.5% test by stable scaffold hash.

## Layout on Arrhenius

Use project storage rather than home. With the example configuration, copy the repository subset like this:

```text
/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/
├── inputs/
│   ├── ZINC.feather
│   └── PubChem.feather
├── containers/
└── gMolAI-retrain/
```

The graph materialization is intentionally disk-heavy. Check quota before starting. If your project has a faster allocation, change `paths.work_dir` in `configs/retrain.yaml` and add its bind in `GMOLAI_EXTRA_BINDS`.

The Slurm account is `naiss2025-3-10-gpu`. Use `storagequota` before graph materialization to confirm available capacity.

## 1. Configure and build the SIF

```bash
cd /nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/00_build_sif.sbatch
```

`configs/arrhenius.env` is ready for the supplied account and path. Edit it only if the deployment layout changes.

The recipe [`arrhenius.def`](arrhenius.def) is pinned to the arm64 digest of NVIDIA PyG `25.09-py3` (CUDA 13.0.1, PyG 2.7) and adds pinned arm64 wheels for PyArrow, RDKit, DuckDB, PyYAML, and the test tools. It must be built on an Arrhenius GPU/aarch64 node. It refuses to overwrite an existing SIF.

The build script places the persistent Apptainer cache in project storage and temporary build layers on `$SNIC_TMP` (the node-local disk), avoiding the small home quota.

After the build finishes, validate the image and code on an arm64 GPU node. Do
not execute the arm64 SIF directly on the amd64 login node:

```bash
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/02_validate_runtime.sbatch
```

## 2. Run tests and a configuration check

The validation job above prints dependency versions, runs all unit tests, runs
the complete objective for two iterations with two CUDA/NCCL DDP ranks, checks
that every parameter receives gradients, verifies rank synchronization, and
checks the retraining configuration. Inspect `slurm-gmolai-validate-<jobid>.out`
and require a zero Slurm exit code before submitting preprocessing or training.

## 3. Submit preprocessing

```bash
bash slurm/submit_pipeline.sh configs/arrhenius.env
```

Arrhenius login nodes are amd64 and GH200 nodes are arm64. The wrapper therefore
executes no container locally: it submits a short arm64 orchestrator job, which
creates the task map and submits the dependency chain. The supplied allocation
is GPU-only, so all containerized preprocessing stages request one GH200 GPU to
obtain an arm64 execution node even though those stages themselves are CPU-heavy.
The wrapper prints the orchestrator job ID; child job IDs are recorded in its
`slurm-gmolai-orchestrate-<jobid>.out` file.

The default `AUTO_SUBMIT_TRAIN=0` submits the dependency chain through graph finalization, then prints the exact training submission command. This pause is intentional. Review:

- `work/dataset_manifest.json`: source/filter/dedup/split counts.
- `work/conflicts/`: descriptor disagreements for duplicate identities.
- `work/descriptor_scaler.json`: named, train-only fitted statistics.
- `work/graph_manifest.json`: graph counts, schema/hash, sizes, and shard list.

Set `AUTO_SUBMIT_TRAIN=1` before submission only if you want the training job chained automatically.

The stages are:

1. SHA-256 verification of both original Feather files.
2. Disjoint record-batch canonicalization array.
3. Hash-bucket deduplication array.
4. Coverage checks, deterministic split manifest, and train-only scaler.
5. One-time RDKit graph featurization array.
6. Graph/count/schema finalization.
7. Optional four-GPU training.

Every stage is idempotent for an unchanged configuration. If a configuration hash differs from existing output, it refuses to reuse that output; choose a new `work_dir` rather than mixing experiments.

## 4. Train and resume

### Current representation model (implementation v5)

`configs/retrain.yaml` is the immutable identity of the already-built graph
artifacts. Training-only changes are separate, composable plans, so an
architecture ablation never invalidates or silently relabels the 223,180,699
graphs. The production representation plan is
[`configs/representation-v1.yaml`](configs/representation-v1.yaml).

Implementation v5 replaces the stochastic per-node VGAE with a deterministic
residual GINE encoder. A mild NT-Xent loss acts directly on mean-atom latents;
the graph readout remains supervised by denoising, link reconstruction, and the
13 source descriptors. The released 384-dimensional molecule vector is the
concatenation of raw graph and mean-atom blocks, standardized by immutable
coordinate statistics fitted only on a stratified pretraining-train sample; the
mean-atom block then receives a fixed weight of 3 for cosine search and
clustering. Validation uses exact task denominators, retains model-hard
negatives, and reports collapse, reconstruction, retrieval, clustering, and
masked/clean-view diagnostics.

The deployment environment already names the production plan, run directory,
and immutable selected checkpoint:

```bash
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/60_train.sbatch
```

Multiple `--plan` arguments can be supplied directly to the CLI; later plans
override earlier ones. `resolved_config.json` and the checkpoint's effective
training-plan hash capture the fully merged result.

### Legacy VGAE v4 (forensic baseline)

Training implementation v4 fixed the train/evaluation posterior mismatch found in
the original `combined-zinc-pubchem-v1` run: descriptor supervision now always
uses the deterministic posterior mean, while sampled `z` remains exclusive to
variational reconstruction. Version 4 deliberately refuses to resume v3
checkpoints. Historical corrected v4 checkpoints live in
`runs/combined-zinc-pubchem-v2`; the immutable graph shards are reused as-is.

Before committing a full node allocation, the versioned performance benchmark
uses the real graph shards and objective but writes to an isolated run directory:

```bash
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/03_benchmark_training.sbatch
```

The default benchmark is 400 steps at the production batch budgets. Optional
submission-time overrides permit memory/throughput comparisons without changing
or regenerating graph artifacts:

```bash
sbatch -A "$NAISS_PROJECT" --export=ALL,\
BENCHMARK_NODE_BUDGET=65536,BENCHMARK_GRAPH_BUDGET=4096 \
  slurm/03_benchmark_training.sbatch
```

Each benchmark records its effective training plan hash and input-pipeline
timings in `runs/benchmark-v4-<jobid>/metrics.jsonl`; its checkpoint cannot be
mistaken for a production checkpoint. One-second GPU telemetry is captured
automatically beside it as `gpu-dmon.txt`.

Implementation v4 retains the optimized v3 input path: it prepares and pins one
deterministic batch ahead, vectorizes per-graph negative construction, performs
hard mining in one segmented GPU operation, and removes avoidable per-step
CUDA/collective synchronizations. Compare **graphs/second** after warm-up, not
GPU utilization alone. The v2 baseline on the original 32,768-node budget was
about 1,850 graphs/s per rank. If the default v4 benchmark remains input-bound,
run the 65,536-node benchmark above; only try 131,072 nodes / 8,192 graphs after
confirming ample HBM headroom.

If training was not chained:

```bash
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/60_train.sbatch
```

To use a benchmark-selected budget for production, export both values at
submission time; they become part of the checkpoint's effective-plan hash:

```bash
sbatch -A "$NAISS_PROJECT" \
  --export=ALL,GMOLAI_NODE_BUDGET_PER_GPU=65536,GMOLAI_GRAPH_BUDGET_PER_GPU=4096 \
  slurm/60_train.sbatch
```

`resolved_config.json` records those effective values. Saved-model evaluation
loads that resolved plan automatically, so it cannot silently evaluate under
different runtime settings.

Training is step-based because a full corpus pass is enormous. The production
screen runs 15,000 steps and retains immutable milestones every 2,500 steps;
semantic selection is independent of the online reconstruction score. Four
ranks receive disjoint shards, shuffle deterministically, and cycle without a
fixed-prefix cap. `last.pt` is written every 500 steps. Ten minutes before wall
time, Slurm sends `USR1`; the job requests a synchronized all-rank checkpoint
and requeues itself. Resume restores model, optimizer, scheduler, AMP state,
every RNG, and every rank's next data position. A changed world size or any
config/data/schema/scaler hash fails closed.

Monitor with:

```bash
squeue -u "$USER"
tail -f slurm-gmolai-train-<jobid>.out
tail -f "$GMOLAI_RUN_DIR/metrics.jsonl"
```

An emergency, graceful stop can be requested without killing a checkpoint write:

```bash
touch "$GMOLAI_RUN_DIR/REQUEST_CHECKPOINT"
```

The process exits with the requeue code within at most ten completed optimizer
steps; this bounded polling avoids an extra DDP collective on every step and is
well inside the ten-minute Slurm warning.

## 5. Independent test evaluation

Only run this after validation/downstream selection fixes one immutable retained
checkpoint. The test split is never used to choose a milestone or calibrator:

```bash
source configs/arrhenius.env
sbatch -A "$NAISS_PROJECT" --export=ALL slurm/70_evaluate.sbatch
```

Results are written to `test_metrics.json` and bind the exact checkpoint SHA,
configuration/plan/data/schema/scaler hashes, split, graph budget, and world
size. Besides exact component losses and descriptor statistics, v5 reports
atom/bond accuracy and macro-F1, exact-match rates, easy and model-hard edge
AUROC/AP, and graph-embedding effective-rank and masked/clean diagnostics. The
test split is not used for checkpoint selection.

## 6. Calibrate, probe, and promote embeddings

First export raw vectors from the selected retained checkpoint and fit
coordinate statistics on the pretraining **train** split only. Use a large,
stratified sample covering all 256 hash buckets:

```bash
python -m gmolai_retrain.cli --config configs/retrain.yaml embed \
  --run-dir "$GMOLAI_RUN_DIR" --checkpoint checkpoints/step-000010000.pt \
  --split train --max-graphs 100000 --sampling-seed 271828 \
  --embedding-definition raw_hybrid --output "$GMOLAI_RUN_DIR/calibration-source.pt"
python -m gmolai_retrain.cli fit-embedding-calibrator \
  --embeddings "$GMOLAI_RUN_DIR/calibration-source.pt" \
  --minimum-graphs 100000 --output "$GMOLAI_RUN_DIR/calibrator.pt"
```

Export calibrated train/validation samples with the same sampling seed, then
run the geometry probe and ten repeated frozen scaffold-split downstream probes:

```bash
python -m gmolai_retrain.cli --config configs/retrain.yaml embed \
  --run-dir "$GMOLAI_RUN_DIR" --checkpoint checkpoints/step-000010000.pt \
  --split train --max-graphs 10000 --sampling-seed 20260810 \
  --embedding-definition standardized_raw_hybrid --mean-node-weight 3 \
  --calibrator "$GMOLAI_RUN_DIR/calibrator.pt" \
  --output "$GMOLAI_RUN_DIR/train-calibrated.pt"
python -m gmolai_retrain.cli --config configs/retrain.yaml embed \
  --run-dir "$GMOLAI_RUN_DIR" --checkpoint checkpoints/step-000010000.pt \
  --split validation --max-graphs 50000 --sampling-seed 20260810 \
  --embedding-definition standardized_raw_hybrid --mean-node-weight 3 \
  --calibrator "$GMOLAI_RUN_DIR/calibrator.pt" \
  --output "$GMOLAI_RUN_DIR/validation-calibrated.pt"
python -m gmolai_retrain.cli --config configs/retrain.yaml probe \
  --train-embeddings "$GMOLAI_RUN_DIR/train-calibrated.pt" \
  --validation-embeddings "$GMOLAI_RUN_DIR/validation-calibrated.pt" \
  --similarity-graphs 5000 \
  --output "$GMOLAI_RUN_DIR/representation-probes.json"
python -m gmolai_retrain.cli --config configs/retrain.yaml benchmark-downstream \
  --run-dir "$GMOLAI_RUN_DIR" --checkpoint checkpoints/step-000010000.pt \
  --datasets-dir work/downstream_benchmarks/moleculenet --scaffold-splits 10 \
  --embedding-definition standardized_raw_hybrid \
  --calibrator "$GMOLAI_RUN_DIR/calibrator.pt" \
  --output "$GMOLAI_RUN_DIR/moleculenet-probes.json"
```

For full-corpus export, consecutive calls may use the same sampling seed with
`--skip-graphs 0`, `--skip-graphs N`, and so on. Windows are computed as
differences of nested balanced bucket prefixes: they cover the same deterministic
population without overlap and encode only the requested window. Every payload
records its `graph_offset`, graph IDs, molecule hashes, and source buckets.

For a large confirmation dataset, `--selected-only` skips the six diagnostic
feature baselines and evaluates only `molecule_embedding`. Selected-only panels
are intentionally rejected by `promote-representation`; promotion requires the
full seven-feature baseline/ablation panel on every one of the five development
datasets shown above, with finite primary metrics for every feature.

Promotion is fail-closed and copies both the checkpoint and calibrator only when
the 100k calibration, 50k validation/10k recurring-scaffold clustering, 5k
similarity-query protocol, effective rank, robust held-out topology, local
chemical retrieval, five-seed KMeans, and all five external development
datasets pass:

```bash
python -m gmolai_retrain.cli --config configs/retrain.yaml promote-representation \
  --run-dir "$GMOLAI_RUN_DIR" --checkpoint checkpoints/step-000010000.pt \
  --calibrator "$GMOLAI_RUN_DIR/calibrator.pt" \
  --representation-probe "$GMOLAI_RUN_DIR/representation-probes.json" \
  --downstream-benchmark "$GMOLAI_RUN_DIR/moleculenet-probes.json"
```

After promotion, `embed --checkpoint auto --embedding-definition auto` verifies
the hash-bound selection metadata and automatically loads both
`representation-best.pt` and `representation-calibrator.pt`.

## Model and objective (v5)

- Four residual GINE blocks, hidden width 256, atom latent width 128, and an
  explicit deterministic 256-dimensional molecule vector.
- Explicit 48-value node and 15-value edge schema; schema hash stored everywhere.
- 30% node-feature masking, 30% bond-feature masking, and 15% bond dropout.
- Symmetric edge decoders use `|z_i-z_j|` and `z_i*z_j`.
- Unique undirected positives; unique per-graph easy/hard negatives with disjoint pools.
- Mean, size-normalized sum, max, and log atom count feed the explicit graph
  readout; node, edge, link, and descriptor heads are graph-vector-conditioned.
- Masked and clean views share the encoder. Mild mean-atom contrastive learning
  prevents collapse without a KL posterior or stochastic downstream vectors.
- The canonical vector is calibrated with train-only coordinate statistics;
  its graph block has weight 1 and its mean-atom block has weight 3.
- BF16, AdamW, cosine schedule, gradient clipping, non-finite loss checks.
- Vectorized deterministic non-edge sampling, segmented hard-negative selection,
  pinned host transfers, and checkpoint-correct one-batch look-ahead.

This is a molecular encoder/reconstructor, not a graph-sampling or de novo
molecule generator.

## Important operational notes

- The package defaults to a single GPU node. Multi-node containers on Arrhenius require site-specific Slingshot/libfabric integration; contact NAISS support before extending it.
- Use `slurm/03_benchmark_training.sbatch` rather than editing the live YAML to benchmark throughput.
- `node_budget_per_gpu=32768` is conservative for a 96 GiB GH200. Increase only from measured peak memory and throughput.
- Never edit a live experiment's feature/schema/split config in place. Use a new
  `work_dir` for data/schema changes; use `GMOLAI_RUN_DIR` for a fresh training
  implementation or plan on the same immutable graph manifest.
- The authoritative audit mapping and remaining limitations are in [`AUDIT_RESPONSE.md`](AUDIT_RESPONSE.md).
