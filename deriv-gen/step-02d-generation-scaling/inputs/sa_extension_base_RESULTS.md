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
