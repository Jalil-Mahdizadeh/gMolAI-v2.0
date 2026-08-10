# Molecular representation repair and validation

## Goal and selected artifact

The target is one deterministic molecule vector that supports frozen downstream
models, clustering, and similarity search. Reconstruction and the 13 source
descriptors are auxiliary training signals, not success criteria.

The selected model is seed 42, step 10,000:

- run: `runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050`
- checkpoint: `checkpoints/step-000010000.pt`
- checkpoint SHA-256: `02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e`
- train-only calibrator: `embedding-calibrator-step10000-100k-seed271828.pt`
- calibrator SHA-256: `5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b`

Fail-closed promotion has completed. `representation-best.pt` and
`representation-calibrator.pt` are byte-identical copies of those selected
artifacts. `representation_selection.json` has SHA-256
`43f1f857576f10fd8aa7ed9276f9ce899ca90d011172225d04e8cff77a9333a1`.

## What failed

The original `combined-zinc-pubchem-v1` run had both a real representation
failure and misleading evaluation:

1. It exposed stochastic per-atom VGAE latents, not a canonical deterministic
   molecule vector.
2. Reconstruction trained from posterior samples while important evaluation
   paths used posterior means.
3. Validation combined losses using graph counts even when their actual
   denominators were masked atoms, bonds, candidate pairs, or nodes; model-hard
   negatives were absent from validation.
4. The finite “random” exporter traversed an ordered hash-bucket prefix, so a
   small probe was not representative of all 256 buckets.
5. Clustering used one KMeans seed and could report a lucky initialization.
6. Descriptor loss and atom-level active units were treated as proxies for a
   useful molecule space. They are not.

The original v1 JSONL already showed severe under-use: active atom coordinates
fell from 103 at step 15k to 52 at 35k, 50 at 45k, and 48 at 55k. Descriptor
validation loss worsened from 0.395 to 0.882. The intermediate v4 correction
fixed the posterior mismatch but did not fix the representation objective:
active coordinates still fell `256 -> 91 -> 57 -> 51` from 15k through 55k,
and median coordinate variance fell `0.03377 -> 0.00385`. Descriptor loss kept
improving while bond reconstruction and latent use degraded. Training longer
was therefore not a remedy.

## Model and objective repair

Implementation v5 is checkpoint-incompatible with the VGAE and starts cleanly:

- four residual GINE blocks, hidden width 256;
- deterministic 128-dimensional atom vectors and explicit 256-dimensional
  `graph_z`;
- mean/size-normalized-sum/max/log-count graph readout;
- masked node and bond-feature reconstruction plus dropped-bond prediction;
- unique per-graph easy and hard negative pools with no positive, cross-graph,
  or easy/hard overlap;
- graph-vector conditioning in the node, bond, existence, and descriptor heads;
- mild NT-Xent on clean/masked mean-node representations.

The selected objective uses node weight 1, bond-feature weight 1,
bond-existence weight 0.5, descriptor weight 0.5, and mean-node contrastive
weight 0.01 at temperature 0.1. KL, projector, VICReg invariance, variance, and
covariance weights are zero. Screens showed that the local contrastive signal
prevented collapse without forcing the public vector to imitate one descriptor
or one fingerprint geometry.

Seed 43 independently completed all 15,000 steps. Its clean graph space retained
all 256 active units throughout; effective rank increased from 26.51 at 2.5k to
37.45 at 10k and 39.12 at 15k, while median coordinate standard deviation
remained 0.72–0.77. The repair therefore does not reproduce the late-collapse
trajectory.

## Public 384-dimensional representation

The public vector is:

1. concatenate raw clean `graph_z` (256) and clean mean-node `z` (128);
2. subtract coordinate means and divide by population standard deviations fitted
   on 100,000 deterministic stratified **pretraining-train** molecules covering
   all 256 buckets;
3. multiply the standardized mean-node block by 3.

The fixed weight improves cosine retrieval and scaffold clustering. Linear
downstream pipelines standardize features inside each training fold, so the
constant block weight does not leak target information. Calibration is a
separate immutable artifact bound to checkpoint, step, config, training plan,
graph manifest, descriptor schema, source sample, and SHA-256.

## Evaluation and promotion repair

Evaluation now:

- aggregates every loss by its exact denominator;
- reports easy and model-hard AUROC/AP, feature accuracy/macro-F1, exact match,
  masked/clean agreement, spectra, and active units;
- samples without replacement across all 256 hash buckets;
- measures topology accessibility using 13 labels absent from training;
- measures Morgan recall@10 and global rank correlation as sanity checks;
- measures direct neighbor Tanimoto/scaffold enrichment;
- runs five KMeans seeds with 20 initializations each against a fixed Morgan
  baseline;
- uses ten repeated nested scaffold splits for external frozen probes;
- exports large embedding collections as deterministic, non-overlapping
  stratified windows, so `--skip-graphs` does not re-encode or duplicate prior
  chunks.

Promotion is fail-closed and requires at least 100k calibration molecules, 10k
probe-training molecules, 50k scaffold-validation molecules, 5k similarity
queries, 10k recurring-scaffold clustering molecules, all seven diagnostic
feature panels, and all five external development datasets. Robust topology
uses mean and median R2 plus standardized MAE so one extreme Kier-shape value
cannot dominate the decision. Checkpoint, calibrator, probes, downstream panels,
and their hashes must all agree.

## Verified results

### Final seed-42 large validation protocol

The authoritative probe is
`representation-probes-validation-standardized-raw-hybrid-w3-step10000-calibration100k-seed20260810-50k-sim5k.json`.

| Metric | Result |
|---|---:|
| Effective rank | 29.973 |
| Topology mean / median R2 | 0.9679 / 0.9675 |
| Scaffold-disjoint topology R2 | 0.9681 |
| Morgan recall@10 | 0.2023 |
| Cosine–Morgan Spearman | 0.4369 |
| Neighbor mean Tanimoto | 0.2290 |
| Neighbor Tanimoto enrichment | 1.9877x |
| Scaffold-neighbor enrichment | 33.800x |
| Latent clustering ARI / NMI | 0.3621 / 0.7884 |
| Morgan clustering ARI / NMI | 0.3171 / 0.7766 |

A second independent 50k validation sample also favored the latent over Morgan
(`ARI 0.3523 vs 0.3215`, `NMI 0.7846 vs 0.7828`). Weight screens at 1, 3, and
6 confirmed weight 3 as the strongest large-sample compromise.

The retained-step screen also supports the 10k selection. Frozen FreeSolv RMSE
for the raw graph-plus-mean-node vector at steps 5k, 7.5k, 10k, 12.5k, and 15k
was respectively `1.391`, `1.372`, `1.297`, `1.323`, and `1.304`; 10k was the
best milestone and the only one satisfying the 1.30 development gate.

### Cross-training-seed replication

Seed 43 at step 10k used its own independent 100k train-only calibrator and the
same final 50k/5k protocol. It passed every representation and five-dataset
promotion threshold:

| Metric | Seed 42 | Seed 43 |
|---|---:|---:|
| Effective rank | 29.973 | 30.420 |
| Morgan recall@10 | 0.2023 | 0.2040 |
| Cosine–Morgan Spearman | 0.4369 | 0.4442 |
| Scaffold enrichment | 33.80x | 34.13x |
| Clustering ARI | 0.3621 | 0.3966 |
| Clustering NMI | 0.7884 | 0.8140 |

### Frozen external scaffold probes (ten splits)

| Dataset | Metric | Seed 42 | Seed 43 |
|---|---|---:|---:|
| BACE | ROC-AUC | 0.8413 | 0.8468 |
| BBBP | ROC-AUC | 0.8789 | 0.8747 |
| ESOL | RMSE | 0.7315 | 0.7237 |
| FreeSolv | RMSE | 1.2971 | 1.2956 |
| Lipophilicity | RMSE | 0.8125 | 0.8051 |
| HIV | ROC-AUC | 0.7578 | -- |

The five development tasks pass their predeclared promotion gates for both
training seeds. The provenance-complete seed-42 panel is
`moleculenet-frozen-probes-standardized-raw-hybrid-w3-step10000-calibration100k-10splits-provenance-v2.json`
(SHA-256 `d91a9201e4c9e23c78849e4413159e45ac608feb494965b46b7766a74b97c11e`).

HIV was an additional confirmation, not a selection dataset. After the same
chemical filtering, it contained 37,225 molecules, 18,651 scaffold groups, and
3.092% positives. Across ten nested scaffold splits, the selected latent gave
ROC-AUC `0.7578 +/- 0.0164`, AP `0.1813 +/- 0.0202`, and balanced accuracy
`0.6856 +/- 0.0241`. The paired Morgan baseline gave ROC-AUC
`0.7440 +/- 0.0258`, AP `0.1945 +/- 0.0417`, and balanced accuracy
`0.6830 +/- 0.0190`. Thus the learned vector improved ranking ROC-AUC and
balanced accuracy, while Morgan retained a small AP advantage on this highly
imbalanced endpoint; the result does not support claiming universal fingerprint
dominance. The full seven-feature artifact has SHA-256
`8fa8fc2837c38578638c916b4a8d7b02dcdcd43803cb818f860d52acd90faba3`;
the independently generated selected-only, complete-provenance artifact has
SHA-256 `7e0d7b7b53a4e9ff241e68d1b87657f9e0a054b18f083203e87f98d1845fcb08`.

### Internal scaffold-hash test

The selected checkpoint was frozen before test access. Four-GPU evaluation on
250,000 test molecules retained all 256 active global units, clean effective
rank 37.675, median coordinate standard deviation 0.732, hard-edge AUROC/AP
0.9661/0.8286, and no non-finite values. `test_metrics.json` now records the
exact checkpoint SHA and all config/data/schema/scaler identities.

On a separate 50k test geometry sample, effective rank was 30.69, recall@10
0.2079, cosine–Morgan Spearman 0.4573, neighbor enrichment 2.001x, and scaffold
enrichment 26.36x. Latent scaffold clustering was strong in absolute terms but
below Morgan for this deliberately different scaffold-hash partition
(`ARI 0.3503 vs 0.3723`, `NMI 0.7224 vs 0.7796`). Weight 3 was still better than
weights 1 and 6 on this partition. Morgan is explicitly engineered from local
substructures, so it remains a useful specialized scaffold baseline; the
learned vector is not claimed to dominate it for every scaffold family. The
latent's broader evidence is its stable retrieval, topology, and external
transfer performance.

### Runtime verification

Slurm job `1066977` completed with exit code 0 in 63 seconds in the pinned GH200
container: all 43 tests passed, two-rank CUDA DDP smoke tests passed for both model
families, and the production config/plan/data hashes validated. Four-GPU
held-out job `1063140` also completed with exit code 0.

The promoted user path was then exercised on the allocated GH200 with
`--checkpoint auto --embedding-definition auto`. It loaded
`representation-best.pt` plus `representation-calibrator.pt` and exported 512
finite 384-dimensional test vectors with 512 unique graph IDs, 512 unique
molecule hashes, and coverage of all 256 source buckets. The smoke artifact has
SHA-256 `d546da0601efa3b7c4a36ceaed9d2a1baffbb361eef68869a44d2f6880fce8bc`.

## Method rationale and limits

The masked graph path is consistent with
[GraphMAE](https://arxiv.org/abs/2205.10803) and
[GraphMAE2](https://arxiv.org/abs/2304.04779). The local contrastive signal and
task-aligned chemical evaluation are motivated by
[MolCLR](https://arxiv.org/abs/2102.10056) and
[iMolCLR](https://arxiv.org/abs/2202.09346). These works motivate mechanisms;
the local ablations, cross-seed replication, scaffold-held-out probes, and
fail-closed gates determine the selected artifact.

This model is a molecular encoder/reconstructor, not a de novo graph generator.
For a fingerprint-specific scaffold search, Morgan may remain preferable. For
general downstream ML, learned similarity, and clustering, use the calibrated
384-dimensional vector and retain the checkpoint/calibrator pair together.
