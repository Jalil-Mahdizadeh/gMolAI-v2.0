---
type: "query"
date: "2026-08-14T06:17:36.783546+00:00"
question: "I do not see the valid plot (blue) in quality_locality_diversity_scaling.png"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Molecular Canonicalization Policy"]
---

# Q: I do not see the valid plot (blue) in quality_locality_diversity_scaling.png

## Answer

Expanded via graph vocabulary: [plot, figure, fraction, policy, valid, similarity]. Direct source verification shows report_results.py plots raw_valid_fraction first in blue and raw_policy_acceptance_fraction second in green. The values differ by only 0.0020 to 0.0250 percentage points over budgets 50 to 1000, so the later green line covers the blue line at the rendered linewidth and marker size. The blue curve is present; the visualization should use distinct linestyle/markers, z-order, or a validity-minus-policy rejection-gap inset if revised. Scientific values are unaffected.

## Outcome

- Signal: useful

## Source Nodes

- Molecular Canonicalization Policy