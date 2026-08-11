# Manuscript rev3 audit artifacts

This directory contains the no-training analyses requested for manuscript revision 3. No pretrained checkpoint was executed by these analyses, and no model was trained or updated.

## Exact seed-42 training exposure

The checkpoint cursor audit reconstructs the number of training graphs consumed by every DDP rank. Training shards were exclusive by `manifest_index % world_size`; all four cursors were still in cycle 0, so total presentations and unique source graphs were identical.

| Step | Rank 0 | Rank 1 | Rank 2 | Rank 3 | Total/unique | Training partition seen |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 14,425,452 | 14,325,085 | 14,434,387 | 14,319,341 | 57,504,265 | 26.0025% |
| 15,000 | 21,644,057 | 21,472,251 | 21,645,531 | 21,474,193 | 86,236,032 | 38.9946% |

The immutable corpus contains 223,180,699 deduplicated graphs, including 221,148,895 in the training partition. Neither checkpoint completed one training-partition pass. The calculation and per-rank cursors are preserved in `training_exposure_seed42.json`; the compact table is `training_exposure_seed42.csv`.

## Exact downstream/pretraining identity overlap

Canonical downstream identities were constructed with the existing eligibility, fragment, stereochemistry and deduplication policy. Exact overlap was defined as equality of the SHA-256 digest of canonical isomeric SMILES. Counts are relative to accepted unique downstream molecules.

| Dataset | Accepted | Any corpus split, n (%) | Training split, n (%) | Validation | Test | Absent |
|---|---:|---:|---:|---:|---:|---:|
| BACE | 1,513 | 414 (27.36%) | 413 (27.30%) | 0 | 1 | 1,099 |
| BBBP | 1,860 | 1,090 (58.60%) | 1,080 (58.06%) | 4 | 6 | 770 |
| ESOL | 1,116 | 969 (86.83%) | 964 (86.38%) | 3 | 2 | 147 |
| FreeSolv | 639 | 526 (82.32%) | 524 (82.00%) | 1 | 1 | 113 |
| Lipophilicity | 4,198 | 2,513 (59.86%) | 2,493 (59.39%) | 11 | 9 | 1,685 |
| HIV | 37,225 | 27,377 (73.54%) | 27,145 (72.90%) | 129 | 103 | 9,848 |

The audit establishes corpus membership, not endpoint-label leakage: pretraining did not use these experimental endpoint labels. It does show that the downstream evaluations are not molecule-novel relative to the unlabeled/auxiliary pretraining corpus, so this limitation must be disclosed. Full provenance, accepted/overlap identity-set digests and split counts are in `pretraining_downstream_overlap.json`; the publication table source is `pretraining_downstream_overlap.csv`.

## Frozen 13-descriptor-only downstream control

The control used the same accepted molecules, ten reconstructed outer scaffold-group splits, inner fold assignments, fold-local `StandardScaler`, Ridge/logistic models, hyperparameter grids and metrics as the selected 10k gMolAI/Morgan artifact. The reconstructed outer seeds and train/test counts matched that artifact exactly. `GroupShuffleSplit(test_size=0.20)` assigns approximately 20% of scaffold groups, not molecules, to test; realized molecule fractions therefore vary.

For exact corpus matches, features were read from the immutable pretraining `d00`-`d12` values. For downstream identities absent from the corpus, the same frozen descriptor definitions were recomputed with the pinned RDKit 2025.09.3 runtime. Across all overlapping molecules, stored and recomputed values had zero tolerance failures (`atol=1e-8`, `rtol=1e-5`); maximum absolute differences were at most `1.71e-13`.

| Dataset | Metric | gMolAI | Morgan | 13 descriptors | Descriptive interpretation |
|---|---|---:|---:|---:|---|
| BACE | ROC-AUC ↑ | 0.8413 ± 0.0250 | 0.8560 ± 0.0366 | 0.6889 ± 0.0563 | Descriptor control lower than both |
| BBBP | ROC-AUC ↑ | 0.8789 ± 0.0432 | 0.8651 ± 0.0192 | 0.8388 ± 0.0505 | Descriptor control lower than both |
| ESOL | RMSE ↓ | 0.7314 ± 0.0586 | 1.5991 ± 0.2060 | 0.8771 ± 0.1237 | Descriptors account for 83.2% of the mean gMolAI-vs-Morgan RMSE gap; gMolAI remains 0.1457 lower |
| FreeSolv | RMSE ↓ | 1.2972 ± 0.1929 | 2.3810 ± 0.4724 | 1.6019 ± 0.1574 | Descriptors account for 71.9% of the mean gap; gMolAI remains 0.3047 lower |
| Lipophilicity | RMSE ↓ | 0.8125 ± 0.0223 | 0.9297 ± 0.0267 | 1.0071 ± 0.0380 | Descriptor control is worse than Morgan; the gMolAI gain is not explained by these descriptors alone |

Values are mean ± descriptive population standard deviation (`ddof=0`) over the same ten accepted outer splits. The gap fractions are descriptive ratios of means, not causal decompositions or tests of statistical superiority. A retraining ablation without descriptor supervision would still be required for causal attribution, but no such retraining was performed here.

The complete split-identity digests, per-split metrics and descriptor provenance are in `descriptor_only_control.json`; the compact publication table source is `descriptor_only_control.csv`.

## Promotion chronology and terminology

Git history proves that the criteria-bearing validator was committed and remained unchanged before the expanded complete 5,000-15,000-step audit. The first commit already contains the criteria, the original 10k selection and a preliminary retained-step FreeSolv screen, so it cannot prove prospective specification before those analyses. Git blob identities, SLURM job `1174926` timestamps, all 17 thresholds and the evidence limitation are preserved in `promotion_criteria_chronology.json`. The defensible description is therefore:

> Frozen promotion criteria applied uniformly in the complete retrospective 5,000-15,000-step checkpoint audit.

The 10k checkpoint remains the only evaluated checkpoint passing all 17 criteria. The retained 2.5k checkpoint was not part of the complete downstream sweep. No criterion or threshold was changed.

## Reproduction commands

The three audit commands below use the immutable Arrhenius container, with
`PYTHONPATH` set to `src`, and the base configuration plus the recorded seed-42
training-plan overlay. They are read-only with respect to checkpoints and the
pretraining corpus; their only writes are the named compact output artifacts.

```text
gmolai-retrain --config configs/retrain.yaml audit-downstream-overlap \
  --plan configs/representation-pilot-mean-node-contrastive-001-desc050.yaml \
  --datasets-dir work/downstream_benchmarks/moleculenet \
  --output artifacts/manuscript-rev3/pretraining_downstream_overlap.json \
  --summary-csv artifacts/manuscript-rev3/pretraining_downstream_overlap.csv

gmolai-retrain --config configs/retrain.yaml benchmark-descriptor-control \
  --plan configs/representation-pilot-mean-node-contrastive-001-desc050.yaml \
  --datasets-dir work/downstream_benchmarks/moleculenet \
  --reference-benchmark runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/promotion-trajectory-table5-rev1/step-000010000/moleculenet-full-diagnostic-standardized-raw-hybrid-w3-10splits.json \
  --output artifacts/manuscript-rev3/descriptor_only_control.json \
  --summary-csv artifacts/manuscript-rev3/descriptor_only_control.csv

gmolai-retrain --config configs/retrain.yaml audit-training-exposure \
  --plan configs/representation-pilot-mean-node-contrastive-001-desc050.yaml \
  --run-dir runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050 \
  --checkpoint checkpoints/step-000010000.pt \
  --checkpoint checkpoints/step-000015000.pt \
  --output artifacts/manuscript-rev3/training_exposure_seed42.json \
  --summary-csv artifacts/manuscript-rev3/training_exposure_seed42.csv
```

Install the `manuscript` optional dependency outside the immutable evaluation
container, then build the publication document without altering rev2:

```text
python scripts/update_manuscript_rev3.py \
  --input ../manuscript/gmolai-rev2.docx \
  --output ../manuscript/gmolai-rev3.docx \
  --descriptor-control artifacts/manuscript-rev3/descriptor_only_control.json \
  --overlap artifacts/manuscript-rev3/pretraining_downstream_overlap.json \
  --exposure artifacts/manuscript-rev3/training_exposure_seed42.json
```

The retained rev2 source has SHA-256
`4db2f590ef61b7d90b2e773eb2510be094a2b738b8d5e42f7f2a8274752ae815`.
The tracked `gmolai-rev3.docx` is byte-identical to the sibling manuscript copy.
