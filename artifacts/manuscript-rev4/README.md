# Manuscript rev4: exact downstream-molecule exposure audit

This directory is the self-contained evidence bundle for manuscript revision 4.
The new analysis reconstructs exact molecular identities consumed before every
retained seed-42 checkpoint from step 5,000 through step 15,000. It did not
train or resume a model, execute the pretrained model, regenerate embeddings,
modify checkpoints or promotion criteria, or alter the promoted step-10,000
artifact.

## Checkpoint-resolved downstream exposure

Canonical downstream identities use the frozen eligibility, fragment,
stereochemistry and deduplication policy. `Corpus` is exact overlap with any
immutable pretraining split; `Train` is the subset assigned to the
221,148,895-graph pretraining training partition. Checkpoint cells report
`n (% of full downstream dataset / % of Train overlap)`.

| Dataset (post-filter n) | Corpus, n (%) | Train, n (%) | Seen@5k | Seen@7.5k | Seen@10k | Seen@12.5k | Seen@15k |
|---|---:|---:|---:|---:|---:|---:|---:|
| BACE (1,513) | 414 (27.36) | 413 (27.30) | 56 (3.70/13.56) | 76 (5.02/18.40) | 109 (7.20/26.39) | 130 (8.59/31.48) | 161 (10.64/38.98) |
| BBBP (1,860) | 1,090 (58.60) | 1,080 (58.06) | 154 (8.28/14.26) | 215 (11.56/19.91) | 278 (14.95/25.74) | 333 (17.90/30.83) | 413 (22.20/38.24) |
| ESOL (1,116) | 969 (86.83) | 964 (86.38) | 111 (9.95/11.51) | 165 (14.78/17.12) | 226 (20.25/23.44) | 300 (26.88/31.12) | 371 (33.24/38.49) |
| FreeSolv (639) | 526 (82.32) | 524 (82.00) | 59 (9.23/11.26) | 92 (14.40/17.56) | 120 (18.78/22.90) | 152 (23.79/29.01) | 186 (29.11/35.50) |
| Lipophilicity (4,198) | 2,513 (59.86) | 2,493 (59.39) | 349 (8.31/14.00) | 504 (12.01/20.22) | 682 (16.25/27.36) | 837 (19.94/33.57) | 989 (23.56/39.67) |
| HIV* (37,225) | 27,377 (73.54) | 27,145 (72.92) | 3,515 (9.44/12.95) | 5,291 (14.21/19.49) | 7,038 (18.91/25.93) | 8,788 (23.61/32.37) | 10,596 (28.46/39.03) |

*HIV is confirmatory context and was not used for checkpoint selection.

The selected step-10,000 model had therefore consumed 109 BACE, 278 BBBP,
226 ESOL, 120 FreeSolv, 682 Lipophilicity and 7,038 HIV identities from the
post-filter downstream sets. Static corpus membership substantially exceeds
actual checkpoint exposure. This is not direct endpoint-label leakage because
the downstream experimental labels were not pretraining targets. It does show
that each evaluation contains a nonzero subset that influenced unsupervised/
auxiliary pretraining updates. The audit is descriptive and does not establish
that identity exposure caused checkpoint-specific endpoint performance.

## Exact reconstruction and validation

Every checkpoint was matched to the immutable graph-manifest identity and
contained all four rank cursors. All cursors remained in cycle 0. Training
shards were assigned exclusively by `manifest_index % world_size`; cycle-0
shard order and within-shard graph order were regenerated with the exact
`InfiniteGraphBatchIterator` seeds. A graph in the current shard counted as
seen only when its shuffled position was strictly less than `graph_position`,
which denotes the next unread graph.

The audit read the allowlisted `data.pkl` identity member from all 27,136
training shard archives and skipped tensor-storage members. It checked all
221,148,895 ordered training graph hashes. The union of the six downstream
training-overlap sets contained 31,848 distinct molecular identities; all
31,848 resolved to exactly one graph boundary. The following invariants passed:

- four complete, mutually exclusive rank shard assignments at every checkpoint;
- monotonically advancing cursors and nested seen-identity sets;
- exact full SHA-256 and canonical-SMILES equality after corpus joins;
- one graph location per distinct training-overlap identity;
- strict pre-checkpoint graph boundaries and no per-dataset double counting;
- no graph tensor storage, model execution, training, or embedding generation.

The row-level audit trail is
`downstream_checkpoint_exposure_identities.csv` (46,551 accepted downstream
rows, including absent and non-training-split identities). Its saved exposure
flags were independently recomputed from the serialized cursors after the
production run.

## Aggregate pretraining exposure

| Step | Unique graph presentations | Training partition seen |
|---:|---:|---:|
| 5,000 | 28,743,683 | 12.9974% |
| 7,500 | 43,109,793 | 19.4936% |
| 10,000 | 57,504,265 | 26.0025% |
| 12,500 | 71,870,280 | 32.4986% |
| 15,000 | 86,236,032 | 38.9946% |

No retained checkpoint completed one pass through the training partition.
Because every cursor remained in cycle 0, graph presentations and unique source
graphs are identical at all five checkpoints.

## Files

- `downstream_checkpoint_exposure.json`: full checkpoint, rank, dataset,
  identity-set digest and validation provenance;
- `downstream_checkpoint_exposure.csv`: compact numeric publication table;
- `downstream_checkpoint_exposure_identities.csv`: row-level canonical identity,
  exact graph location and checkpoint flags;
- `training_exposure_seed42_5k-15k.{json,csv}`: aggregate all-rank cursor audit;
- `descriptor_only_control.{json,csv}`,
  `pretraining_downstream_overlap.{json,csv}` and
  `promotion_criteria_chronology.json`: unchanged rev3 evidence copied
  byte-for-byte for a self-contained bundle;
- `gmolai-rev4.docx`: publication-formatted manuscript revision generated from
  retained `../manuscript/gmolai-rev3.docx`;
- `artifact_manifest.json`: SHA-256 manifest for every publication input and
  output in this bundle.

## Reproduction

Run in the immutable Arrhenius project container with `PYTHONPATH=src`:

```text
gmolai-retrain --config configs/retrain.yaml audit-downstream-exposure \
  --plan configs/representation-pilot-mean-node-contrastive-001-desc050.yaml \
  --run-dir runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050 \
  --checkpoint checkpoints/step-000005000.pt \
  --checkpoint checkpoints/step-000007500.pt \
  --checkpoint checkpoints/step-000010000.pt \
  --checkpoint checkpoints/step-000012500.pt \
  --checkpoint checkpoints/step-000015000.pt \
  --datasets-dir work/downstream_benchmarks/moleculenet \
  --output artifacts/manuscript-rev4/downstream_checkpoint_exposure.json \
  --summary-csv artifacts/manuscript-rev4/downstream_checkpoint_exposure.csv \
  --identity-ledger-csv artifacts/manuscript-rev4/downstream_checkpoint_exposure_identities.csv \
  --workers 8
```

Build the manuscript with the `manuscript` optional dependency:

```text
python scripts/update_manuscript_rev4.py \
  --input ../manuscript/gmolai-rev3.docx \
  --output ../manuscript/gmolai-rev4.docx \
  --downstream-exposure artifacts/manuscript-rev4/downstream_checkpoint_exposure.json
```

The tracked manuscript is byte-identical to the sibling
`../manuscript/gmolai-rev4.docx`; revision 3 remains unchanged.
