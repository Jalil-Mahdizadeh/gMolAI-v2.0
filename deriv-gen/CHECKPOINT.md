# Derivative-generation checkpoint

Updated: 2026-08-14

## Current state

Steps 1, 1b, 2, 2b, 2c, and 2d are complete. The latest requested work—an
additive Step-2d synthetic-accessibility (SA) comparison—also completed and
passed verification. There is no active SLURM job: the direct n54 run finished
successfully before a submission was necessary (about 51 seconds end to end).
Do not submit a duplicate job or regenerate candidates.

## Frozen strategic decision

The planned controlled MMP-edit generation branch is retired. A known,
site-specific structural modification, such as a specified CH3-to-Cl
substitution, should be made directly in the molecular graph/SMILES; that route
is exact and computationally preferable to encoding, applying a latent edit,
and decoding. The previously reserved Step 3 and Step 4 directories contained
only unexecuted README placeholders and were removed without an archive.

Steps 1 and 1b and all of their frozen artifacts remain permanent
representation-interpretability evidence. Recurrent transformations define
transferable latent directions on unseen cores, the signal survives the
at-least-5, at-least-10, and at-least-20 independent-core cohorts, and the
released x3 hybrid remains a viable frozen edit geometry. This is evidence of
latent chemical organization, not a recommendation to use latent generation
for an explicitly known edit.

The full decision record is
[`STRATEGIC_DIRECTION.md`](STRATEGIC_DIRECTION.md).

The frozen scientific boundary remains intact:

- gMolAI seed-42/step-10,000 checkpoint, calibrator, and released 384-D x3
  representation were unchanged.
- The Step-2 decoder and Step-2d candidate-generation policy were unchanged.
- No training, latent perturbation, MMP-direction editing, property
  optimization, endpoint labels, or locked-test rows were used.
- The original Step-2d strategy and budget decision remain unchanged:
  `hybrid_b500_s500_t120`, 1,000 raw proposals per seed.

## Next objective (not started)

`seed molecule -> frozen gMolAI embedding -> frozen decoder -> large candidate library -> property-guided prioritization`

The future scientific question is whether that candidate library can contain
molecules with improved desired property profiles while retaining useful
similarity/scaffold relationships to the seed and acceptable
synthetic-accessibility characteristics.

No property target, property model, prioritization protocol, new candidate
generation, or evaluation has been selected or run. This checkpoint records
the direction only; it does not authorize or begin the experiment.

## SA extension

Protocol: `step-02d-generation-scaling/PROTOCOL_SA_EXTENSION.md`

The standard RDKit Contrib SA score (lower is heuristically easier) was computed
once for all 2,108,115 globally unique policy-accepted identities and joined
losslessly to 2,116,072 seed-candidate rows. Analogue summaries exclude exact
seed identities. Delta SA is candidate score minus its matched seed score.

At the 1,000-proposal budget, among 2,106,446 genuine non-seed pairs:

- candidate median SA: 3.654; matched-seed median SA: 3.331;
- median delta SA: +0.181 (IQR +0.005 to +0.478);
- 24.20% were no harder than the seed;
- 76.22% were no more than 0.5 SA point harder;
- seed-macro mean delta SA: +0.310 (95% seed-bootstrap CI +0.305 to +0.314).

Interpretation is bounded: SA score is a heuristic molecular-complexity proxy,
not a synthesis route, yield estimate, procurement assessment, or experimental
feasibility result.

## Integrity and outputs

Extension verification is `passed`; all checks passed for population
completeness, lossless joins, exact-seed zero-delta controls, summary coverage,
figures, protected base artifacts, and scientific boundaries. The full Step-2d
SHA-256 ledger also passes.

Primary files:

- `step-02d-generation-scaling/outputs/sa_extension_decision.json`
- `step-02d-generation-scaling/outputs/sa_extension_verification.json`
- `step-02d-generation-scaling/outputs/tables/final_sa_scores_by_identity.parquet`
- `step-02d-generation-scaling/outputs/tables/final_candidate_sa_comparison.parquet`
- `step-02d-generation-scaling/outputs/tables/final_seed_sa_metrics.parquet`
- `step-02d-generation-scaling/outputs/tables/final_sa_summary_by_budget.csv`
- `step-02d-generation-scaling/outputs/tables/final_sa_summary_by_category_budget.csv`
- `step-02d-generation-scaling/outputs/figures/synthetic_accessibility_scaling.png`
- `step-02d-generation-scaling/outputs/figures/synthetic_accessibility_by_category.png`
- `step-02d-generation-scaling/outputs/figures/synthetic_accessibility_seed_candidate_density.png`

The earlier validity/policy-overlap issue is fixed in
`quality_locality_diversity_scaling.png`: RDKit validity is now drawn as a
dashed hollow blue trace above policy acceptance, with a lower basis-point gap
panel so both quantities are visible.

To re-check integrity without changing artifacts:

```bash
cd deriv-gen/step-02d-generation-scaling
sha256sum -c --quiet outputs/SHA256SUMS
```

Expected exit code: 0 with no output. Preserve every prior artifact unchanged.
Do not recreate the retired Step 3/Step 4 controlled-edit branch without a
superseding strategic decision. The next property-guided prioritization
experiment remains unstarted.
