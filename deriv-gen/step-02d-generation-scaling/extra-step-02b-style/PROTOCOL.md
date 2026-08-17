# Frozen protocol: Step 2b-style analysis of Step 2d candidates

## Scope

This additive analysis reads the completed Step 2d final candidate library and
does not generate candidates, train a model, perturb a latent vector, or change
any completed Step 2d artifact. All new files, including caches and temporary
files, must remain under this directory.

The evaluated population is the frozen 10,000-seed Step 2d final panel under
`hybrid_b500_s500_t120`. Budgets 50, 100, 250, 500, and 1,000 are literal nested
prefixes of raw proposal slots. They are not relabelled as counts of accepted
unique candidates.

## Frozen inputs

- Step 2d raw proposal shards, raw-SMILES policy audit, candidate
  characterization, seed-budget metrics, final panel, and 384-dimensional final
  conditions.
- Released `optimized_gine_v1` encoder, coordinate calibrator, resolved model
  configuration, and `released_hybrid_w3` representation definition.
- Existing Step 2d Morgan similarity and scaffold annotations. These are reused
  rather than recomputed under a potentially different chemistry runtime.

Every input is hash-recorded in `inputs/manifest.json`. The repository is mounted
read-only during execution; this directory is mounted separately read-write.

## Direct search and structural metrics

For every seed and budget, report raw token-decode, RDKit-valid, and unchanged
policy-acceptance fractions; unique accepted identity count and yield; candidate
non-emptiness and count thresholds; the frozen greedy proposal outcome; the
first policy-accepted identity in generator order; and oracle exact-seed recall
anywhere in the prefix.

Greedy is identified by the retained raw `source_kind == "greedy"` row, not by
assuming raw proposal rank 1. Generator-order top-1 is the accepted identity with
the lowest first proposal rank. Exact identity uses the retained SHA-256 identity.
Scaffold recovery is exact equality of the retained seed and candidate scaffold
strings. Morgan similarity is the frozen Step 2d radius-2, 2,048-bit value.

An unavailable top-1 contributes zero to unconditional validity, identity,
scaffold, and Morgan summaries. Conditional Morgan summaries are also retained
so the denominator remains explicit.

## Frozen-latent reranking

All globally unique molecules in the Step 2d final analysis are re-encoded once
with the released optimized encoder using batch size 512, node budget 65,536,
48 RDKit workers, and the frozen calibrated 384-dimensional hybrid x3 output.
The encoder input order is the retained Step 2d `structure_index`; identity and
order must match exactly after encoding.

Within each seed and raw budget, candidates are ordered by:

1. lower relative L2 distance to the retained supplied seed condition;
2. higher cosine similarity to that condition;
3. lower first raw proposal rank;
4. lexical canonical SMILES.

Absolute L2, relative L2, and cosine are computed in float64 and stored as
float32. Exact seed identity, scaffold recovery, and Morgan similarity are
evaluation-only quantities and never enter ordering.

Report reranked exact identity@1, valid top-1 availability, selection efficiency
given oracle seed presence, paired gain over greedy and generator-order top-1,
top-1 scaffold recovery and Morgan similarity, and selected latent distance
summaries for every budget.

## Uncertainty and figures

All confidence intervals are deterministic paired seed bootstraps with 2,000
resamples and percentile 95% limits. Ratios such as selection efficiency are
recomputed within each resample. Figures are descriptive and do not define a
new GO/NO-GO gate. Each figure has an explicit CSV source table under
`outputs/plot-data/` and is exported as PNG and SVG.

## Exclusions

No shuffled-condition or nearest-wrong-condition control is included. No
property label, locked-test molecule, synthesis claim, property optimization,
or new candidate-generation result is introduced.
