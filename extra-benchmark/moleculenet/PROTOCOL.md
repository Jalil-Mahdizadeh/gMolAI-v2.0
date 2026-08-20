# Frozen protocol: MoleculeNet development panel plus HIV confirmation

**Scientific protocol frozen before comparator execution:** 12 August 2026
**Execution mode:** frozen inference, native FP32 features, one NVIDIA GPU
**Three-dimensional information:** no conformers or coordinates used or generated

## Evidence scope

The benchmark compares seven fixed 1D/2D molecular representations on six
endpoint datasets. BACE, BBBP, ESOL, FreeSolv, and Lipophilicity retain their
existing role as selection-conditioned development/promotion evidence. HIV is
a separate external post-selection confirmatory endpoint. Neither role is
equivalent to the locked internal pretraining test partition.

This is a retrospective additive comparison. Results cannot alter the promoted
gMolAI seed-42 step-10,000 checkpoint, its train-only 100,000-graph calibrator,
the ×3 mean-node weighting, or the released 384-D representation. The protocol
does not claim that downstream molecules are structurally unseen; molecule
membership and actual pretraining exposure are addressed by the separate
repository audits.

## Frozen representations

| Representation | Frozen feature | Dimension | Primary batch |
|---|---|---:|---:|
| gMolAI seed-42/10k | calibrated graph-z plus 3× mean-node-z | 384 | repository encoder |
| Morgan radius-2 | RDKit 2,048-bit fingerprint as float32 | 2,048 | 4,096 |
| MolAI epoch 6 | three-LSTM hidden/cell concatenation, linear, tanh | 512 | 256 |
| MoLFormer | official pooler output | 768 | 128 |
| SMI-TED-Light | official model-encode output | 768 | 100 |
| MolCLR-GIN | official downstream pre-projection graph vector | 512 | 256 |
| KERMT v2 | cMIM projected mean latent | 512 | 64 |

All neural weights are frozen and set to evaluation/inference mode. Native
dimensions are retained; no PCA, feature selection, representation tuning, or
endpoint fine-tuning is performed. Only the downstream linear predictors are
fitted.

The selected gMolAI checkpoint SHA-256 is
`02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e`;
the calibrator SHA-256 is
`5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b`.
Every source dataset, reference artifact, script, checkpoint, calibrator, and
container image is pinned in `protocol.json`.

## Molecular preparation and split inheritance

Preparation reuses the repository implementation in
`src/gmolai_retrain/downstream.py`:

- repository-pinned isomeric canonical SMILES;
- fragment rejection, allowed-element checks, and 2–256 atom bounds;
- canonical-identity deduplication;
- averaging of duplicate regression labels; and
- rejection of conflicting duplicate classification labels.

Within each dataset, SHA-256 identity hashes are required to be unique and
collision-safe against their canonical SMILES. Dataset occurrences are not
globally deduplicated across endpoints because each endpoint is evaluated as a
separate task.

The ten authoritative outer splits are reconstructed using scaffold groups and
the accepted seeds from the existing seed-42/10k endpoint artifacts.
`GroupShuffleSplit(test_size=0.20)` assigns approximately 80/20% of scaffold
groups—not molecules—to the outer train/test roles; realized molecule fractions
therefore vary. The five development datasets must exactly match the prior
outer and inner identity manifests. HIV must match its authoritative accepted
outer seeds and molecule counts.

After adapter screening, every retained common molecule keeps its original
outer and inner role. No split is regenerated after intersection. Every outer
split is revalidated for complete partitioning, scaffold disjointness, and both
classes where applicable. Each inner fold is likewise revalidated as a
scaffold-disjoint partition of its inherited outer-training identities.

An “outer test fold” here means only a held-out fold within a MoleculeNet/HIV
nested scaffold split; it is never the 1,088,766-molecule locked internal
pretraining test partition.

## Common-coverage rule

Every adapter screens the complete prepared panel before endpoint fitting.
Invalid inputs, unsupported tokens, excessive sequence lengths, non-finite
vectors, and zero-norm vectors fail explicitly. Inputs are never truncated,
silently replaced, or mapped to zero vectors. The primary comparison is the
intersection supported by all seven representations.

MolAI retains its released 109-token vocabulary limit and does not enable the
legacy unknown-character-to-zero fallback. MoLFormer and SMI-TED retain their
202-token limits and lossless-tokenization checks. All exclusions and reasons
are recorded in `outputs/coverage.csv`.

## Identical downstream probes

Features are standardized independently inside each fit/train fold. Target
labels never affect neural representations.

Regression endpoints (ESOL, FreeSolv, Lipophilicity):

- Ridge regression with `solver=lsqr`;
- alpha grid: 0.1, 1, 10, 100, 1,000;
- selection by mean RMSE over three inherited inner grouped folds; and
- outer metrics: RMSE (primary), normalized RMSE, MAE, R², and Spearman.

Classification endpoints (BACE, BBBP, HIV):

- logistic regression with `solver=liblinear`, balanced class weights,
  `max_iter=3000`, and `random_state=0`;
- C grid: 0.01, 0.1, 1, 10;
- selection by mean ROC-AUC over three inherited inner grouped folds; and
- outer metrics: ROC-AUC (primary), average precision, and balanced accuracy.

Reported means and population standard deviations (`ddof=0`) summarize ten
overlapping outer scaffold splits. They are descriptive dispersions, not
standard errors or ten independent experimental replications. Per-split paired
differences and win/tie/loss counts are retained, but no unplanned significance
test or universal-superiority claim is made.

## Timing scope

The workflow records wall time, rows/s, output dimension, and peak GPU memory
for each common-panel export. These observations include model loading, warm-up,
preprocessing, inference, and serialization; Morgan runs on CPU, and the
runtime-recovery sequence placed encoders on two GH200 nodes. Consequently,
`outputs/encoding_runtime_observed.csv` is provenance metadata, not the
publication-grade sustained-throughput comparison. A separate controlled speed
benchmark would require identical timing boundaries, repeated steady-state
runs, and serialization-free model-only timing.

That requirement was subsequently addressed by the completed
[controlled speed protocol](../speed/PROTOCOL.md) and
[audited speed results](../speed/RESULTS.md). Those records, rather than the
incidental export timings described here, are authoritative for throughput
claims.

## Execution chronology and integrity

Initial job `1219337` completed and validated gMolAI, Morgan, MolAI, MoLFormer,
and SMI-TED-Light, then stopped before MolCLR-GIN inference because the launcher
overrode that container's built-in Python path. No MolCLR output was created.
The runtime wrapper was amended to preserve each comparator container's native
environment and to inherit the error trap. This amendment changed no molecule,
split, representation, weight, hyperparameter, metric, or scientific result;
it is recorded in `protocol.json`.

Both remaining containers passed deterministic 127-molecule inference smoke
tests. Job `1219522` then reverified all 36 immutable inputs, reconstructed the
same common panel, checksum/identity/shape-validated the five retained matrices,
encoded MolCLR-GIN and KERMT v2, evaluated all seven representations, and exited
`0:0` after 1:16:04 on node n501.

`state/COMPLETE.json` was written only after all per-model records were present
and internally consistent. Its protocol digest matches the current frozen
protocol, its summary and checksum-ledger digests match byte-for-byte, and all
28 entries in `outputs/SHA256SUMS` independently pass verification. No training,
fine-tuning, checkpoint mutation, calibrator mutation, or embedding regeneration
outside this benchmark occurred.
