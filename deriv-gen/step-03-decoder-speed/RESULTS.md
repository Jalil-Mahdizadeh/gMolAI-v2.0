# Step 03 results

The released stochastic decoder generated **1,737.8 raw proposals/s**
(batch-bootstrap 95% interval 1,583.5–1,903.0) on one
**NVIDIA GH200 120GB**. After byte-token decoding, the rate was
1,711.1 raw SMILES/s (1,561.4–1,878.0).

The requested usable-output rate was **91.6 per-seed unique,
RDKit-valid molecules/s** (57.9–128.0). This numerator was
6,460 first-occurrence identities
from 100,000 proposal slots. 95.45% of raw
slots were RDKit-valid and 6.46% yielded a first unique valid
identity within its conditioning seed.

As a stricter secondary result, the released encoder policy accepted
95.45% of raw slots and produced 58.3 per-seed
policy-unique molecules/s (36.0–83.2).

## Measured workload

- 100 reproducibly sampled `released_hybrid_w3` molecular embeddings.
- Exactly 1,000 stochastic draws per embedding: 100,000 raw slots total.
- Released sampling settings: temperature 1.2, top-p 0.995, 128-byte maximum.
- Query batch size 2, yielding 50 measured batches on exactly one visible GPU.
- One full-shape warm-up batch was excluded.

## Timing audit

| Component | Seconds | Headline inclusion |
|---|---:|---|
| GPU generation + transfers | 57.543 | both rates |
| Token-to-SMILES decode | 0.899 | valid-unique only |
| RDKit validation/canonicalization/dedup | 12.051 | valid-unique only |
| Release-policy pass | 40.379 | secondary policy rate only |
| Serialization | 0.246 | excluded |
| Warm-up | 3.093 | excluded |
| Model load | 0.338 | excluded |

Peak measured CUDA allocation was 2.56 GiB. The observed measured-loop
wall time, including serialization and Python bookkeeping, was
111.266 s.

## Interpretation

The raw rate measures decoder proposal slots, whereas the valid-unique rate
discounts malformed and repeated outputs and includes the corresponding chemistry
work. Uniqueness is intentionally reset per conditioning seed. The 95% intervals
resample the 50 batches from this single run; they quantify panel/batch
heterogeneity, not run-to-run hardware variance.

## Artifacts

- `figures/decoder_throughput.png` and `.svg`: headline comparison.
- `figures/batch_throughput_trace.png` and `.svg`: run-order stability.
- `figures/per_seed_valid_unique_yield.png` and `.svg`: validity/yield spread.
- `outputs/plot-data/`: exact source CSVs used for every figure.
- `outputs/tables/`: summary, per-batch, and per-seed tables.
- `outputs/raw/proposals.parquet`: all 100,000 proposal-level records.
- `outputs/verification.json`: integrity checks.
