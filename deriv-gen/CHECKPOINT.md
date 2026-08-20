# Derivative-generation checkpoint

Updated: 2026-08-20

## Terminal state

Steps 1, 1b, 2, 2b, 2c, and 2d are complete. The additive Step-2d
synthetic-accessibility (SA) comparison also completed and passed verification.
Two bounded post-closure analyses completed on 17 August 2026: read-only latent
reranking of the existing Step-2d accepted library and a throughput-only test of
the frozen decoder. Neither modifies or extends the frozen Step-2d candidate
library, and neither changes the terminal decision.

Derivative generation is now closed at this state. No additional generation,
property-guided prioritization, latent optimization, decoder training, or
candidate evaluation is required for the current project or manuscript.

The future generation baseline is now frozen in the versioned, read-only
[`shared/frozen-generation-v1/`](shared/frozen-generation-v1/) contract. This
contract records an operational handoff only; it did not launch a run or alter
any completed-step file.

## Post-closure analyses

The
[latent-reranking addendum](step-02d-generation-scaling/extra-step-02b-style/RESULTS.md)
re-encoded 2,108,115 globally unique accepted molecules from 10,000 seeds
without candidate generation, training, latent perturbation, property analysis,
or locked-test access. At the 1,000-proposal budget, target-blind gMolAI
reranking reached 96.25% exact seed identity at rank 1 against a 96.26% oracle
ceiling, for 99.9896% selection efficiency.

The separate
[decoder-throughput benchmark](step-03-decoder-speed/RESULTS.md) used 100
embeddings and 1,000 stochastic draws per embedding solely for systems
measurement. On one GH200, its 100,000 proposal slots ran at 1,737.8 raw
proposals/s, 91.6 per-seed unique RDKit-valid molecules/s, and 95.455%
raw-slot RDKit validity. It did not train or change the encoder, decoder,
sampling policy, or candidate-selection contract. Neither post-closure analysis
used endpoint labels or supports a property-improvement claim.

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
- The canonical runtime decoder is Step-2 `checkpoints/best.pt` (SHA-256
  `bb9623080ddaed070278c8abca39252e070c110a6611b3bd7a75caf6c37a41f6`);
  its compact inference export is `decoder_inference.pt` (SHA-256
  `8b4f8db04499083ea2e9d028eaaae18d629b34ce773608d8e2c80863e9121d47`).
- No training, latent perturbation, MMP-direction editing, property
  optimization, endpoint labels, or locked-test rows were used.
- The original Step-2d strategy and budget decision remain unchanged:
  `hybrid_b500_s500_t120`, 500 beam plus 500 sample-stream hypotheses at
  temperature 1.2/top-p 0.995, and 1,000 raw proposals per seed.

## Future-work design (not an active objective)

`seed molecule -> frozen gMolAI embedding -> frozen decoder -> large candidate library -> property-guided prioritization`

The future scientific question is whether that candidate library can contain
molecules with improved desired property profiles while retaining useful
similarity/scaffold relationships to the seed and acceptable
synthetic-accessibility characteristics.

No property target, property model, prioritization protocol, new candidate
generation, or evaluation has been selected or run. This question is retained
as a possible separate future study and is not an unfinished stage of the
current work.

The frozen decoder output is a proposal library, not an optimized library.
Users may apply independent, project-specific prediction or experimental
filters to prioritize candidates, but the present evidence makes no claim that
those candidates improve an ADMET, bioactivity, or other desired property.

Any future candidate-library protocol must verify and consume
`shared/frozen-generation-v1` unchanged. A decoder or sampler change requires a
new versioned contract and explicit superseding decision.

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
superseding strategic decision. Property-guided prioritization remains
unstarted future work. The separate frozen-representation TDC ADMET and
external molecular-clustering benchmarks have completed; neither reopens this
checkpoint or validates generated-candidate properties.

To verify the cross-step decoder and sampler freeze from the repository root:

```bash
python deriv-gen/shared/frozen-generation-v1/verify.py
```

Expected output: `PASS: gmolai-deriv-gen-frozen-generation-v1`.
