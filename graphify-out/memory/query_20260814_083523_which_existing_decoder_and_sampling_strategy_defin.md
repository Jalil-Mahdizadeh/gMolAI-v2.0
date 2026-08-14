---
type: "query"
date: "2026-08-14T08:35:23.057815+00:00"
question: "Which existing decoder and sampling strategy define the frozen deriv-gen generation baseline?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Decoder", "freeze_strategy.py", "generation_core.py", "policy"]
---

# Q: Which existing decoder and sampling strategy define the frozen deriv-gen generation baseline?

## Answer

The canonical runtime decoder is deriv-gen/step-02-decoder-feasibility/checkpoints/best.pt at SHA-256 bb9623080ddaed070278c8abca39252e070c110a6611b3bd7a75caf6c37a41f6; decoder_inference.pt at SHA-256 8b4f8db04499083ea2e9d028eaaae18d629b34ce773608d8e2c80863e9121d47 is its compact export. Step 2d selected hybrid_b500_s500_t120: 500 ordered beam plus 500 sample-stream hypotheses, temperature 1.2, top-p 0.995, deterministic proportional merge and seed derivation, with 1,000 raw proposal slots. The permanent cross-step binding is deriv-gen/shared/frozen-generation-v1.

## Outcome

- Signal: useful

## Source Nodes

- Decoder
- freeze_strategy.py
- generation_core.py
- policy