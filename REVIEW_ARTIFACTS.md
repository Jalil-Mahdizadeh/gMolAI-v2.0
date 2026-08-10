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

## Compact training and validation record

All 396 JSON, JSONL, and `COMPLETE` files generated under `runs/` at review
packaging time are tracked (23.14 MiB total). They include:

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
