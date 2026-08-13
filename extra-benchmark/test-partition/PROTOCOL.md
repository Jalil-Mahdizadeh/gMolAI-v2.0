# Frozen protocol: locked internal test-partition benchmark

**Frozen before comparator execution on the locked identities:** 2026-08-12  
**Execution mode:** inference only, one NVIDIA GPU, FP32 outputs  
**3D policy:** no conformers or molecular coordinates are used or generated

## Scientific scope

The population is the 1,088,766-molecule locked internal pretraining test
partition. Its exact existing 50,000-molecule stratified export is the common
representation panel. The 13-target Ridge probe uses the exact existing 10,000
training-partition export. These data have no BACE/BBBP/ESOL/FreeSolv/
Lipophilicity endpoint labels, so no endpoint leaderboard is computed here.

This is a retrospective additive comparison. The seed-42/10k checkpoint,
train-only calibrator, and released 384-D representation were already frozen;
no result from this benchmark may alter them. Comparator adapters and batching
are qualified on pretraining-validation identities before the locked panel is
processed.

## Frozen comparator panel

| Comparator | Frozen feature | Dimension | Primary batch |
|---|---|---:|---:|
| gMolAI seed-42/10k | released calibrated graph-z + 3× mean-node-z | 384 | existing authoritative export |
| Morgan | RDKit radius 2, 2,048-bit fingerprint as float32 | 2,048 | CPU chunks of 4,096 |
| MolAI epoch 6 | three-LSTM hidden/cell concatenation, linear, tanh | 512 | 256 |
| MoLFormer | official `pooler_output` | 768 | 128 |
| SMI-TED-Light | official `model.encode` output after one warm-up | 768 | 100 |
| MolCLR-GIN | official downstream 512-D pre-projection graph vector | 512 | 256 |
| KERMT v2 | 512-D cMIM projected mean latent | 512 | 64 |

All native dimensions are retained. No PCA or fitted cross-model projection is
allowed. The KERMT ARM64 pilot passed, so it is included rather than treated as
conditional.

## Input and coverage policy

The ordered input is the canonical isomeric SMILES bound to each immutable
SHA-256 molecular identity in the existing gMolAI exports. Every model must
attempt the same ordered panel. Coverage is reported per model and the primary
paired diagnostics use the intersection successfully supported by every
comparator.

- Invalid inputs and non-finite or zero-norm outputs fail explicitly.
- Inputs are never silently truncated, replaced with zero vectors, or dropped.
- MoLFormer and SMI-TED inputs exceeding 202 tokens including special tokens
  are rejected before inference.
- SMI-TED and MolAI retain their official non-isomeric canonicalization; this
  loss of stereochemical information is reported rather than altered.
- MolAI inputs exceeding 109 tokenized characters or containing unsupported
  characters are rejected. Its legacy unknown-character-to-zero behavior is
  not enabled.
- All rejected row identities and reasons are retained.

The original 50,000 identities, similarity sample, and recurring-scaffold
subset remain auditable even if the all-model common set is smaller. Results
must state the realized common counts; they must not be presented as if every
model encoded all 50,000 molecules.

## Common diagnostics

The repository's existing implementation is reused without changing metric
definitions:

- effective rank and rank/dimension;
- participation ratio and ratio/dimension;
- coordinate dispersion and top-eigenvalue fraction;
- 13-target held-out topology Ridge probe with fold-local scaling and
  `alpha=10`;
- Morgan-neighbour recall, cosine/Tanimoto association, chemical-neighbour
  enrichment, and scaffold-neighbour enrichment on the seeded 5,000-row
  similarity sample;
- scaffold clustering using row-L2-normalized inputs followed by standard
  Euclidean K-means, five seeds, 20 initializations, and at most 500
  iterations. This is not spherical K-means because centroids are not
  constrained or renormalized.

Morgan-neighbour agreement is a similarity-to-Morgan diagnostic, not an
absolute definition of quality. Raw effective rank is dimension-dependent, so
normalized rank is always reported beside it. gMolAI-only reconstruction and
masked-graph metrics are excluded from the common leaderboard.

## Timing scope and subsequent controlled benchmark

The job records model-load-plus-representation export wall time, attempted and
accepted rows, output bytes, and observed peak GPU memory for the frozen
10k/50k panels. These remain provenance for this workflow rather than an
apples-to-apples throughput leaderboard. The subsequent dedicated speed design
was frozen on the 49,844-molecule all-model-common panel and completed on one
GH200 with common timing boundaries, unmeasured warm-up, four batch sizes, and
one full measured pass per condition; see
[`../speed/PROTOCOL.md`](../speed/PROTOCOL.md) and
[`../speed/RESULTS.md`](../speed/RESULTS.md). That completed protocol supersedes
the preliminary full-corpus, five-repetition plan recorded before the definitive
speed protocol. It governs the current reported comparison while explicitly
retaining the limitation that its point estimates have no confidence intervals.

## Prohibitions

No training, checkpoint resumption, fine-tuning, endpoint-label use,
checkpoint/calibrator mutation, embedding-layer selection on locked data,
silent truncation, or change to the promoted gMolAI artifact is permitted.
