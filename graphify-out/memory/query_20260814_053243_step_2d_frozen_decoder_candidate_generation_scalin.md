---
type: "codebase"
date: "2026-08-14T05:32:43.279183+00:00"
question: "Step 2d frozen decoder candidate-generation scaling: what strategy and budget are supported?"
contributor: "graphify"
outcome: "useful"
---

# Q: Step 2d frozen decoder candidate-generation scaling: what strategy and budget are supported?

## Answer

Completed and fail-closed verified on 10,000 fresh validation seeds. Development-only selection froze hybrid_b500_s500_t120: 500 beam hypotheses plus 500 fixed-seed samples at temperature 1.2/top-p 0.995, proportionally interleaved. At 1,000 raw slots/seed: 71.30% valid, 71.27% policy accepted, mean 211.61 unique accepted identities, 19.47 exact Step-1b one-cut MMPs, 37.70 novel useful-local analogues, 99.99% novelty among genuine non-seed molecules versus only the 980,000 decoder-fit identities, median seed Morgan 0.460. All 5 preregistered large-library gates passed. Useful yield had diminishing returns from budget 250 but no strict saturation through 1,000; the preregistered 90%-of-maximum utility rule therefore recommends 1,000. No training, test data, endpoints, latent perturbation, property optimization, or Step 3.

## Outcome

- Signal: useful