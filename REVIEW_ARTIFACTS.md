# External review artifacts

This repository carries the exact deployable representation and the compact
evidence needed to audit why it was selected. It does not carry source datasets
or derived graph corpora.

## Canonical embedding artifacts

The selected run directory is:

`runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050`

| Artifact | Size | SHA-256 | Purpose |
|---|---:|---|---|
| `representation-best.pt` | 26,796,827 bytes | `02f49a2a94ddfc9dc780cc3d5f1a3df54306ae0fdc5d4b3767e3fd2e7f27b05e` | Exact seed-42 step-10,000 encoder checkpoint |
| `representation-calibrator.pt` | 6,661 bytes | `5cbe3210b2fa6742b165c61e3562118553f567df13181d863776c9ca5527365b` | Train-only coordinate mean/std and public block-weight contract |
| `representation_selection.json` | 39,829 bytes | `43f1f857576f10fd8aa7ed9276f9ce899ca90d011172225d04e8cff77a9333a1` | Hash-bound promotion decision and gate evidence |
| `embedding-auto-promoted-smoke.pt` | 859,835 bytes | `d546da0601efa3b7c4a36ceaed9d2a1baffbb361eef68869a44d2f6880fce8bc` | 512-vector automatic-load/export smoke payload |

`representation-best.pt` and `representation-calibrator.pt` are the pair that
must be retained together for the public 384-dimensional molecular vector.
`--checkpoint auto --embedding-definition auto` validates the selection record
and both hashes before loading them.

Byte-identical copies of the promoted pair, selection record, and resolved model
configuration are packaged under `inference/models/` together with the compact
frozen Step-2 decoder export. The public CLI applies the training-time
canonicalization/feature policy and does not need graph shards:

```bash
python inference/gmolai.py validate
python inference/gmolai.py encode \
  --input inference/data/example_smiles.csv \
  --output inference/output/embeddings.npz
python inference/gmolai.py decode \
  --embeddings inference/output/embeddings.npz \
  --output-dir inference/output/candidates \
  --proposal-budget 1000
```

See [`inference/README.md`](inference/README.md) for the `.npz` schema, flags,
validity/uniqueness policy, frozen sampling contract, and provenance. The
byte-frozen legacy CSV encoder and compatibility symlink remain for historical
manifests.

With the immutable graph data available at the paths in `configs/retrain.yaml`,
the promoted export path is:

```bash
RUN_DIR=runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050
python -m gmolai_retrain.cli --config configs/retrain.yaml embed \
  --run-dir "$RUN_DIR" --checkpoint auto \
  --embedding-definition auto --split test \
  --max-graphs 10000 --sampling-seed 424242 \
  --output "$RUN_DIR/test-embeddings.pt"
```

The repository does not bundle the molecules or graph shards consumed by this
command. The included smoke payload lets reviewers inspect the exact output
schema and provenance without those data.

## Full retained-checkpoint selection audit

The primary seed-42 checkpoints at steps 5k, 7.5k, 10k, 12.5k, and 15k were
each rerun through the complete Table 5 protocol rather than compared with a
partial milestone screen. All five checkpoints passed all 105 protocol,
artifact-identity, source-integrity, completeness, and repository-validator
consistency checks. Their quality-gate outcomes were:

| Step | Table 5 criteria | Failing quality criteria | Full gate |
|---:|---:|---|---|
| 5,000 | 15/17 | Effective rank 24.8654; FreeSolv RMSE 1.39045 | Fail |
| 7,500 | 16/17 | FreeSolv RMSE 1.37227 | Fail |
| 10,000 | 17/17 | None | Pass |
| 12,500 | 16/17 | FreeSolv RMSE 1.32289 | Fail |
| 15,000 | 16/17 | FreeSolv RMSE 1.30346 | Fail |

The FreeSolv maximum was fixed at 1.30. Step 15k therefore misses promotion by
0.00346 RMSE even though its other 16 criteria pass; changing the threshold
after seeing this result would violate the frozen fail-closed gate.

Repository history supports a narrower chronology than prospective language:
the criteria-bearing validator was committed and remained unchanged before this
expanded complete sweep. The first repository commit, however, already contains
the criteria, the original 10k selection, and a preliminary retained-step
FreeSolv screen. The sweep is therefore described as a complete retrospective
audit using uniformly applied frozen criteria; the repository does not prove
prospective specification before the original selection or preliminary screen.
Exact Git and SLURM evidence is preserved in
`artifacts/manuscript-rev3/promotion_criteria_chronology.json`.

Versioned compact evidence is under
`runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/promotion-trajectory-table5-rev1/`:

- `Table5_full_promotion_trajectory_steps_5k-15k.csv`, SHA-256
  `7a2380647bc0df82a041806d1bf4f2e4e5b3352a9dbbb2d3fe1be6cdce40c1a2`;
- `promotion_trajectory_audit.json`, SHA-256
  `02eb29abca45d292f7fed56b9de72a3887c3cb6555946999b9894fdff47c597e`;
- the representation-probe and full MoleculeNet JSON for each checkpoint,
  plus its `COMPLETE` marker.

[`scripts/summarize_promotion_trajectory.py`](scripts/summarize_promotion_trajectory.py)
performs the independent identity/protocol audit and invokes the repository's
actual fail-closed validator for every checkpoint.
[`slurm/72_promotion_trajectory.sbatch`](slurm/72_promotion_trajectory.sbatch)
reproduces the five-step evidence generation. The 100k calibration, 10k probe
training, and 50k probe-validation tensors are reproducible bulk intermediates
and are deliberately not versioned.

## Manuscript rev4 no-training controls

[`artifacts/manuscript-rev4/`](artifacts/manuscript-rev4/) contains the exact
five-checkpoint DDP-cursor exposure audit, six-dataset canonical-identity overlap
audit, model-free 13-descriptor downstream control, and publication-formatted
rev4 manuscript. The rev3 bundle remains intact.

The new checkpoint-resolved audit scans the identity metadata for all 27,136
training graph shards (221,148,895 graph boundaries), reconstructs the exact
rank-specific cycle-0 shard and within-shard order, and applies the exclusive
cursor boundary at steps 5k, 7.5k, 10k, 12.5k, and 15k. Its compact CSV reports
corpus overlap, training-partition overlap, exact seen counts, percentage of the
full downstream dataset seen, and percentage of training overlaps seen for
BACE, BBBP, ESOL, FreeSolv, Lipophilicity, and confirmatory HIV. The row-level
ledger preserves canonical SMILES, SHA-256 identity, split, exact rank/shard/
graph location, and every checkpoint seen flag. Canonical equality is checked
alongside each digest, locations are unique, all four ranks are complete, and
the audit loads no graph tensor storage.

The descriptor artifact validates reconstructed 10k outer seeds/counts, records
outer and inner identity-set hashes, and includes the original gMolAI/Morgan
results alongside the new control. Stored pretraining descriptors and
pinned-RDKit recomputations agreed for every overlapping molecule at the
configured tolerances. Checkpoint mappings are deserialized with PyTorch's
restricted weights-only loader solely to read DDP cursors; no model was
instantiated or executed, no embeddings were regenerated, and no training was
performed. The exact presentation history is descriptive context and does not
by itself establish a causal effect on downstream performance.

## Manuscript rev5 editorial release

[`artifacts/manuscript-rev5/`](artifacts/manuscript-rev5/) contains the
structurally reorganized, publication-formatted rev5 manuscript, a reproducible
document-only builder and a hash-bound validation manifest. The revision makes
the chronology and evidence roles explicit, separates the locked internal test
from the external HIV confirmation, and preserves the rev4 numerical results,
tables, equations, checkpoints, calibrator and promotion decision unchanged.

## Compact training and validation record

The compact JSON, JSONL, CSV, and `COMPLETE` files generated under `runs/` at
review packaging time are tracked. They include:

- the original collapsing v1 history and deterministic-but-collapsing v4/v2
  history;
- representation objective screens and milestone comparisons;
- complete seed-42 and independent seed-43 training histories, resolved
  configurations, identities, and completion records;
- geometry, topology, retrieval, similarity, clustering, weight-ablation, and
  checkpoint-selection probes;
- repeated scaffold-split FreeSolv and five-dataset MoleculeNet panels;
- the full and provenance-complete HIV confirmation panels;
- held-out test metrics and the fail-closed promotion record.

Decisive evidence hashes include:

| Evidence | SHA-256 |
|---|---|
| Selected training `metrics.jsonl` | `b602bc1b03837c215a2072cdd16fe33d3f3a75d8de8c401fff6768522d2ad6e0` |
| Original v1 `metrics.jsonl` | `92a026b2156f25bcd640e21c1f3fb252809808b43ef1e0027ac24fddb5598a81` |
| Intermediate v4/v2 `metrics.jsonl` | `6169b40f705e4686d2d04c2b78a93c50ab3975c4893e2e65a0c83c0cf427f08a` |
| Independent seed-43 `metrics.jsonl` | `7abbe8c32931b5f1f22aa3a9e751b4ab7be64dbe927f9ad568d49f02441545d3` |
| Authoritative 50k/5k validation probe | `7142f7045f1e07441e4000bfc7aa2bdb89a9d7b9b8ba28001457df91487421e4` |
| Seed-43 50k/5k validation probe | `a49afdbd5d1ad7581a3156240ae650ddec212ff9bf0551d2f601a3594fbeb6cf` |
| Five-dataset provenance-complete panel | `d91a9201e4c9e23c78849e4413159e45ac608feb494965b46b7766a74b97c11e` |
| HIV provenance-complete panel | `7e0d7b7b53a4e9ff241e68d1b87657f9e0a054b18f083203e87f98d1845fcb08` |
| Four-GPU held-out `test_metrics.json` | `acb171a2248b77ba39e4ca7c721ba8c6ea5f23b4955adf7ad80fcfdae009dbd3` |

The final runtime logs are also tracked:

- `slurm-gmolai-validate-1066977.out`, SHA-256
  `7cef25c104adf8e75cb3103b436fd107a5348016292209c6ebefb7e7174cceb7`;
- `slurm-gmolai-test-1063140.out`, SHA-256
  `990b7118a08beaf457481d58f8f477bb3ef7c6ac157b50bb4e3dc39635f0c02f`.

## Deliberately excluded large artifacts

- ZINC, PubChem, HIV, and other benchmark source datasets;
- canonicalized/deduplicated rows, manifests with row-level records, conflicts,
  descriptor caches, and graph shards under `work/`;
- intermediate and rejected model checkpoints;
- 8–167 MB train/validation/calibration embedding tensors and repeated weight
  exports;
- container images, caches, and routine Slurm output.

These files are either private/source data, reproducible from the recorded
configuration, or redundant with the promoted checkpoint. Their identities,
counts, and relevant hashes remain recorded in the compact provenance artifacts.
