# Step 2d results: frozen decoder candidate scaling

The development-only comparison selected **hybrid_b500_s500_t120** before any final
generation. Final evaluation used 10,000 fresh validation seeds
and literal nested raw proposal budgets of 50, 100, 250, 500, and 1,000.

## Main result

At 1,000 raw proposals per seed, 71.30% were
RDKit-valid and 71.27% passed the
unchanged gMolAI policy. The decoder yielded a mean of
211.61 unique accepted identities,
210.64 genuine non-seed molecules,
19.47 exact Step-1b one-cut MMP derivatives, and
37.70 novel useful-local analogues per seed.
Novelty among genuine non-seed molecules was
99.99% relative only to the
980,000 decoder-training identities.

Locality broadened with budget: the budget-1,000 median seed-candidate Morgan
similarity was 0.460; the
weighted sampled/exact within-set mean was
0.393. Non-empty scaffold
retention among eligible non-seed candidates was
12.53%;
acyclic seeds are tabulated separately in the machine-readable outputs.

## Scaling decision

The preregistered 90%-utility rule recommends **1000 raw proposals
per seed**, which yields 37.70 mean novel
useful-local analogues per seed versus 37.70 at 1,000.
Diminishing returns begin at 250;
strict saturation begins at no registered budget.

Prospective large-library classification: **SUPPORTED**
(5/5 gates passed). This is a bounded chemical-yield
statement, not evidence of synthesis feasibility, bioactivity, or property gain.

## Reproducibility notes

- Every budget is a prefix of the same frozen 1,000-slot stream.
- Invalid outputs, policy failures, duplicate strings, alternative strings for one
  molecular identity, and seed identities remain in their true raw denominators.
- MMPs use the exact imported Step-1b one-cut implementation.
- Morgan fingerprints use radius 2, 2,048 bits, and no chirality flag.
- Candidate novelty is assessed only against the decoder-fit 980,000 molecules.

<!-- STEP2D_SA_EXTENSION:START -->
## Post-completion synthetic-accessibility comparison

This additive analysis scored all 2,108,115
globally unique policy-accepted identities once with RDKit
2025.09.3 Contrib SA_Score, then evaluated all
2,116,072 unique seed-candidate rows.
Lower is heuristically easier; delta SA is candidate minus matched seed. Exact
seed identities are excluded below.

| Budget | Candidate SA median | Seed SA median | Median delta SA | No harder | No more than 0.5 harder |
|---:|---:|---:|---:|---:|---:|
| 50 | 3.427 | 3.284 | +0.056 | 33.10% | 90.20% |
| 100 | 3.474 | 3.290 | +0.077 | 31.06% | 87.32% |
| 250 | 3.543 | 3.308 | +0.113 | 28.26% | 83.10% |
| 500 | 3.597 | 3.319 | +0.145 | 26.15% | 79.67% |
| 1,000 | 3.654 | 3.331 | +0.181 | 24.20% | 76.22% |

At budget 1,000, seed-macro mean delta SA was
+0.310 (95% seed-bootstrap CI
+0.305 to
+0.314). Seed-macro mean fraction no
harder was 22.64% (95% CI
22.33% to
22.93%).

| Category at budget 1,000 | Pairs | Median delta SA | No harder |
|---|---:|---:|---:|
| One-cut MMP | 194,690 | +0.201 | 21.86% |
| Same-scaffold non-MMP | 182,363 | +0.105 | 25.77% |
| Scaffold-changing | 1,677,033 | +0.187 | 24.41% |
| Acyclic non-MMP | 52,360 | +0.286 | 20.87% |

This heuristic is not a route, yield, availability, or experimental
synthesizability claim. Generation, ranking, gates, and budget remain unchanged.
See PROTOCOL_SA_EXTENSION.md and the final_sa tables for full definitions.
<!-- STEP2D_SA_EXTENSION:END -->
