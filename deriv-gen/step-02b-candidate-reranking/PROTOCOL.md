# Step 2b protocol: frozen-decoder candidate search and latent reranking

## Question

Step 2 showed that the decoder follows its supplied 384-D gMolAI condition,
but greedy exact identity on its held-out panel was 63.90%. Step 2b asks a
narrower question: is the missing top-1 fidelity mainly a search problem?

For each condition, the frozen decoder produces a compact candidate set. Every
accepted molecule is re-encoded by the frozen released gMolAI inference path.
Candidates are ranked only by relative L2 distance to the supplied condition,
with supplied-condition cosine and generator order as deterministic tie-breaks.
Target identity, target SMILES equality, Morgan similarity, and scaffold
identity are evaluation variables only and are never ranking inputs.

## Frozen boundary

- gMolAI seed 42 / step 10,000 checkpoint: immutable.
- Released calibrator and `released_hybrid_w3` definition: immutable.
- Step-2 selected decoder checkpoint: immutable; no optimizer is instantiated.
- Existing chemistry policy: unchanged.
- No endpoint labels, locked-test molecules, latent perturbation, MMP edit, or
  derivative generation.

All external files and source paths are hash-bound in `inputs/manifest.json`.
The Step-2b directory must contain no trained checkpoint.

## Development and final discipline

The candidate policy is selected on the fixed 2,048-molecule train-partition
decoder-development panel. Four preregistered policies share the original
greedy candidate: 64-wide beam search with length penalties 0.0 and 0.6, and
128 fixed-seed top-p samples at two registered temperature/top-p settings.

The chosen generation policy is sealed before final generation. Final
evaluation uses 10,000 deterministic validation molecules that are disjoint
from the complete original Step-2 10,000-molecule generation panel. This makes
the final autoregressive analysis prospective rather than post-hoc.

Candidate-set sizes are 1, 5, 10, 20, and 50. Slot 1 is always the original
greedy proposal; if greedy is invalid, size 1 has no accepted candidate.
Larger sets add unique, valid, policy-accepted molecules in target-blind
generator order.

## Metrics

Oracle exact Recall@k (target present anywhere in the set) is reported
separately from deployable latent-reranked exact identity@1. For every size and
condition the study also reports candidate availability and uniqueness,
scaffold recovery, Morgan similarity, latent cosine/L2/relative-L2, and the
gain over greedy. Shuffled and nearest-wrong conditions report both original
target and supplied-condition-source recovery.

The frozen GO/NO-GO thresholds and the operational definitions of
search-related versus compression-related residual error are recorded in
`config/protocol.json` before any development candidate results are generated.
