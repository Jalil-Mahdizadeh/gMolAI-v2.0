# Frozen benchmark protocol

## Question

How quickly does the released gMolAI embedding-to-SMILES decoder produce raw
proposals, and how quickly does it produce distinct RDKit-valid molecules?

## Inputs

- Embedding space: `released_hybrid_w3`, 384 dimensions.
- Source: the frozen Step 02d fresh-validation panel and its matching condition
  matrix.
- Selection: 100 rows sampled without replacement with NumPy PCG64 and selection
  seed `20260817`. Sampling order is retained as benchmark order.
- The selected panel, selected condition matrix, source hashes, and selected-file
  hashes are copied into `inputs/` before model execution.

## Decoder

- Artifact: `inference/models/decoder_inference.pt`, loaded and hash-validated by
  the released `inference/gmolai.py` implementation.
- Primitive: released `generate_seeded_sample_pool`.
- Draws: exactly 1,000 for each of 100 conditioning vectors (100,000 slots).
- Query batch size: 2 conditioning vectors (2,000 simultaneous draw streams).
- Temperature: 1.2.
- Top-p: 0.995.
- Maximum decoded SMILES bytes: 128.
- CUDA autocast: bfloat16, matching the release primitive.
- Sampling seed: the released immutable per-molecule `sample_seed` mapping.
- Device rule: execution aborts unless PyTorch sees exactly one CUDA GPU.

This is a stochastic-sampling throughput benchmark, not the release CLI's hybrid
beam/sample candidate strategy. It therefore measures exactly the requested
1,000 samples per seed without mixing in beam search or a greedy proposal.

## Timing regions

All CUDA timing uses host monotonic time with a CUDA synchronization before and
after the measured region.

- `generation_seconds`: condition host-to-device transfer, autoregressive
  generation, and generated token/score/length transfer to the host.
- `token_decode_seconds`: conversion of byte tokens into raw SMILES strings.
- `rdkit_validation_seconds`: RDKit parsing, isomeric canonicalization, hashing,
  and within-seed first-occurrence marking.
- `release_policy_seconds`: a separate pass through the released encoder
  canonicalization/acceptance policy.
- `serialization_seconds`: Parquet writing; recorded but excluded from both
  headline rates.

One full-shape two-seed × 1,000-draw warm-up is run and excluded. Model loading,
artifact verification, warm-up, plotting, and file I/O are not included in the
headline denominators.

## Metrics

`raw_proposals_per_second = 100000 / sum(generation_seconds)`

`valid_unique_molecules_per_second = sum(per-seed first unique RDKit-valid identities) / sum(generation_seconds + token_decode_seconds + rdkit_validation_seconds)`

Uniqueness resets for every conditioning seed, so the same identity generated for
two different seed embeddings contributes once to each seed's usable output. A
global unique count is retained separately and is not used in the throughput
numerator.

Secondary metrics include raw-SMILES throughput (adding token decoding), token
decodability, RDKit-valid yield, release-policy acceptance, policy-unique
throughput, per-batch rates, component times, and peak CUDA memory.

## Uncertainty and interpretation

Percentile 95% intervals resample the 50 measured two-seed batches 10,000 times
and recompute ratio-of-sums throughput. They describe heterogeneity across this
fixed selected panel and run; they are not a substitute for independent hardware
runs and should not be interpreted as a hardware reproducibility interval.
