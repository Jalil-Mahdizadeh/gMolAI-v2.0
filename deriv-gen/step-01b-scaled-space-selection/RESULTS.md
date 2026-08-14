# Scaled latent-space selection results

## Scope

This inference-only study compared five diagnostic coordinate spaces without
retraining or changing gMolAI, its checkpoint, calibrator, or released weight-3
embedding definition. It used 1,000,000 pretraining-train molecules for
mining and 50,000 disjoint pretraining-validation molecules for
unseen-core validation. Locked-test molecules and endpoint labels were not used.

This is evidence about latent edit geometry and retrieval. It is not evidence
that gMolAI can decode embeddings or generate novel molecules.

## MMP scale and support

Train core-transformation observations:
1,461,391. Distinct train
transformations: 1,323,920. The fixed
retrieval panel contains 2,048 unseen-core queries.

| minimum_train_cores | transformations | core_transform_observations | transformations_with_unseen_core_validation | unseen_core_validation_observations |
|---|---|---|---|---|
| 2 | 89280 | 226751 | 3964 | 9074 |
| 5 | 4199 | 36502 | 1248 | 5311 |
| 10 | 799 | 16015 | 420 | 3396 |
| 20 | 201 | 8244 | 141 | 2123 |

## Directional transfer

Primary inference is the at-least-5-independent-core cohort. Values are
unweighted macro averages across transformations. The null is a deterministic
support-matched mismatched transformation, held fixed across all five spaces.

| space | alignment | null_alignment | alignment_gain | gain_ci_low | gain_ci_high |
|---|---|---|---|---|---|
| graph_256 | 0.8251 | 0.0157 | 0.8094 | 0.7986 | 0.8200 |
| mean_node_128 | 0.9456 | 0.0257 | 0.9199 | 0.9080 | 0.9319 |
| hybrid_w1 | 0.8566 | 0.0186 | 0.8380 | 0.8279 | 0.8483 |
| released_hybrid_w3 | 0.9138 | 0.0233 | 0.8904 | 0.8797 | 0.9013 |
| hybrid_w6 | 0.9351 | 0.0249 | 0.9102 | 0.8986 | 0.9216 |

Directional transfer remains positive through the at-least-20-core cohort:

| minimum_cores | transformations | observations | alignment_gain | ci_low | ci_high | survives |
|---|---|---|---|---|---|---|
| 2 | 3964 | 9074 | 0.8967 | 0.8906 | 0.9029 | True |
| 5 | 1248 | 5311 | 0.8904 | 0.8797 | 0.9013 | True |
| 10 | 420 | 3396 | 0.8831 | 0.8663 | 0.9000 | True |
| 20 | 141 | 2123 | 0.8739 | 0.8493 | 0.8993 | True |

## Derivative retrieval

The table reports transformation-macro metrics for the fitted MMP direction on
the identical 2,048-query validation panel. Recall is for the exact
held-out derivative identity; exact requested edit accepts any molecule with
the query core and requested target substituent.

| space | recall_at_1 | recall_at_10 | scaffold_retention | mmp_consistency | exact_requested_edit | mean_seed_retrieved_tanimoto |
|---|---|---|---|---|---|---|
| graph_256 | 0.9429 | 0.9927 | 0.9771 | 0.9525 | 0.9429 | 0.6840 |
| mean_node_128 | 0.9601 | 0.9984 | 0.9782 | 0.9687 | 0.9601 | 0.6885 |
| hybrid_w1 | 0.9677 | 0.9962 | 0.9782 | 0.9717 | 0.9677 | 0.6902 |
| released_hybrid_w3 | 0.9693 | 0.9988 | 0.9782 | 0.9753 | 0.9693 | 0.6899 |
| hybrid_w6 | 0.9638 | 0.9988 | 0.9782 | 0.9720 | 0.9638 | 0.6893 |

Every query had exactly one validation molecule for its requested
core-plus-target-substituent identity, so exact requested edit and exact
derivative recall@1 coincide in this panel. Both columns are retained to make
the intended metrics explicit.

For the selected released_hybrid_w3 space, fitted directions strongly exceed the
unperturbed seed, random, and support-matched mismatched controls:

| method | recall_at_1 | recall_at_10 | mmp_consistency |
|---|---|---|---|
| seed_nn | 0.3044 | 0.6947 | 0.5086 |
| isotropic | 0.3011 | 0.6882 | 0.4988 |
| global_covariance | 0.2719 | 0.6126 | 0.4396 |
| local_covariance | 0.2452 | 0.5803 | 0.3963 |
| mismatched_mmp_direction | 0.2840 | 0.6300 | 0.4424 |
| mmp_direction | 0.9693 | 0.9988 | 0.9753 |

## Frozen primary comparison

| space | primary_transformations | primary_queries | alignment_gain | positive_alignment_transformation_fraction | exact_derivative_recall_at_1 | exact_requested_transform | viable | pareto | selected_edit_control_space |
|---|---|---|---|---|---|---|---|---|---|
| graph_256 | 1141 | 2048 | 0.8094 | 1.0000 | 0.9429 | 0.9429 | True | False | False |
| mean_node_128 | 1141 | 2048 | 0.9199 | 1.0000 | 0.9601 | 0.9601 | True | True | False |
| hybrid_w1 | 1141 | 2048 | 0.8380 | 1.0000 | 0.9677 | 0.9677 | True | False | False |
| released_hybrid_w3 | 1141 | 2048 | 0.8904 | 1.0000 | 0.9693 | 0.9693 | True | True | True |
| hybrid_w6 | 1141 | 2048 | 0.9102 | 1.0000 | 0.9638 | 0.9638 | True | True | False |

Selected edit-control space: released_hybrid_w3

The released weight-3 hybrid is viable and within every predeclared non-inferiority margin of the Pareto frontier, so representation compatibility breaks the tie.

The decoder-conditioning representation remains
released_hybrid_w3; diagnostic weights do not
alter the released representation.

## Weight 3 versus weight 1

Weight 3 clearly improves directional alignment over weight 1; exact retrieval is numerically higher but its paired confidence interval includes zero.

Paired hierarchical-bootstrap differences:

| metric | delta_w3_minus_w1 | ci_low | ci_high | resolved_positive |
|---|---|---|---|---|
| alignment_gain | 0.0525 | 0.0480 | 0.0567 | True |
| exact_derivative_recall_at_1 | 0.0016 | -0.0067 | 0.0104 | False |
| exact_requested_transform | 0.0016 | -0.0067 | 0.0104 | False |

## Mean-node assessment

Mean-node-128 remains the strongest directional-alignment space and is Pareto-optimal, but it is not the unique overall winner; released weight 3 has the highest exact recall and wins the compatibility-aware frozen rule.

## Required answers

1. Released weight 3 clearly improves directional alignment over weight 1.
   Its exact retrieval point estimate is slightly higher, but that difference
   is not statistically resolved.
2. Mean-node-128 remains the directional-alignment leader and is Pareto-optimal,
   but it is not the unique overall winner.
3. MMP-direction transfer survives at 1M scale and in the at-least-5,
   at-least-10, and at-least-20 independent-core cohorts.
4. Freeze released_hybrid_w3 as the edit-control space under the predeclared
   compatibility-aware rule. Keep
   released_hybrid_w3 unchanged for decoder
   conditioning.

## Output map

- outputs/raw contains observation- and query-level machine-readable results.
- outputs/tables contains transformation summaries, hierarchical bootstrap
  intervals, paired comparisons, and the selection table.
- outputs/figures contains concise diagnostic plots in PNG and SVG.
- outputs/space_decision.json contains the machine-readable frozen decision.
- state/COMPLETE.json and outputs/SHA256SUMS provide execution and integrity
  seals.
