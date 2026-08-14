# Latent control-space decision

Selected edit-control space: released_hybrid_w3

The released weight-3 hybrid is viable and within every predeclared non-inferiority margin of the Pareto frontier, so representation compatibility breaks the tie.

The released decoder-conditioning representation remains
released_hybrid_w3. This decision concerns only
the geometry used to define molecular edits. It does not change gMolAI and does
not establish a decoder or de novo generation capability.

## Weighting result

Weight 3 clearly improves directional alignment over weight 1; exact retrieval is numerically higher but its paired confidence interval includes zero.

## Mean-node result

Mean-node-128 remains the strongest directional-alignment space and is Pareto-optimal, but it is not the unique overall winner; released weight 3 has the highest exact recall and wins the compatibility-aware frozen rule.

## Directional-transfer result

Transfer remains positive at every evaluated support threshold, including 420
transformations at at least 10 train cores and 141 transformations at at least
20 train cores. Full estimates and hierarchical confidence intervals are in
RESULTS.md and the machine-readable tables.
