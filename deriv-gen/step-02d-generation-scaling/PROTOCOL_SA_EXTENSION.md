# Step 2d post-completion synthetic-accessibility extension

## Status and scope

This is an additive descriptive amendment requested after the original Step 2d
results were complete. It is not part of the original prospective generation
protocol and does not change its candidate stream, selected strategy, gates, or
budget decision. The analysis rules below are frozen before any Step 2d
synthetic-accessibility scores are inspected.

The frozen encoder, calibrator, released 384-D weight-3 representation, decoder,
unperturbed conditions, final validation seeds, raw candidate streams, chemistry
policy, canonical identities, MMP labels, scaffold labels, and Morgan values are
inputs only. No model is run or trained, no candidate is regenerated or filtered,
and no test molecule or endpoint label is used.

## Population and score

The computation covers every globally unique canonical identity in
`intermediate/final_unique_molecules.parquet`. These are the policy-accepted,
RDKit-valid identities underlying the final Step 2d candidate table and its
seeds. Each identity is parsed and scored exactly once with the SA scorer and
fragment model bundled with the pinned RDKit installation in the registered
gMolAI container. The implementation file, fragment-model file, RDKit version,
and their SHA-256 hashes are sealed before scoring.

The SA score ranges from 1 (heuristically easier) to 10 (heuristically harder).
It is a fragment-frequency and molecular-complexity heuristic, not a
retrosynthetic route, reaction-yield estimate, availability assessment, or
experimental synthesizability claim.

For every unique seed-candidate pair:

`delta_sa = candidate_sa - seed_sa`

Negative values are easier than the matched seed under this heuristic; positive
values are harder. Exact seed-identity rows are retained as a zero-delta
integrity control but excluded from analogue summaries.

## Frozen summaries

At nested raw budgets 50, 100, 250, 500, and 1,000, report:

- candidate, matched-seed, and delta-SA count, mean, standard deviation,
  minimum, q10, q25, median, q75, q90, and maximum;
- fractions no harder than the seed, no more than 0.5 SA units harder,
  within absolute 0.5 SA units, at least 1 unit easier, and at least 1 unit
  harder;
- candidate count and seed coverage;
- both pair-weighted estimates and seed-macro estimates;
- deterministic seed-level bootstrap 95% confidence intervals for macro mean
  delta and macro fraction no harder.

Repeat pair summaries for one-cut MMPs, scaffold-preserving non-MMP analogues,
scaffold-changing analogues, and acyclic non-MMP analogues. Preserve the full
identity-level score table, seed-candidate comparison table, and seed-budget
table in machine-readable form.

## Frozen visualizations

Produce:

1. matched candidate/seed SA distributions and relative-SA scaling by budget;
2. category-specific delta-SA and fraction-no-harder comparisons at budget
   1,000;
3. an all-pair seed-versus-candidate SA density map at budget 1,000;
4. a corrected validity/locality figure in which valid and policy-accepted
   curves are distinguishable and their small gap is shown in basis points.

## Interpretation boundary

The extension answers whether generated candidates are systematically easier or
harder than their own seeds under one registered heuristic. It must not be used
to claim a feasible route, synthesis success, biological activity, property
improvement, latent control, or authorization for Step 3.
