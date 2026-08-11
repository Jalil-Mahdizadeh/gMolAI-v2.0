---
type: "promotion_evaluation"
date: "2026-08-11T12:15:30.846195+00:00"
question: "i want you to conduct a full promotion evaluation on checkpoints 5k-15k according to the table 5 in the rev1 document. We must fully understand how the full promotion gate varies by the training steps. Afterwards, prepare a csv file (smilar to the table 5 in the rev1 document) with corresponding values at each checkpoint."
contributor: "graphify"
outcome: "useful"
source_nodes: ["Semantic Promotion Suite", "_validate_promotion_quality()", "run_representation_probes()", "benchmark_moleculenet()"]
---

# Q: i want you to conduct a full promotion evaluation on checkpoints 5k-15k according to the table 5 in the rev1 document. We must fully understand how the full promotion gate varies by the training steps. Afterwards, prepare a csv file (smilar to the table 5 in the rev1 document) with corresponding values at each checkpoint.

## Answer

Expanded from the original request via graph vocabulary: [promotion, gate, checkpoint, step, validation, downstream, benchmark, seed, standardized, representation, calibration, test]. Executed the complete Table 5 promotion protocol on the primary seed-42 checkpoints at steps 5000, 7500, 10000, 12500, and 15000 with checkpoint-specific 100000-molecule train-only calibrators, 10000 probe-training embeddings, 50000 validation embeddings, 5000 similarity queries, five KMeans repetitions, and all five MoleculeNet datasets with ten scaffold splits and all seven diagnostic feature panels. Slurm array 1174926 completed all five tasks with exit code 0. All 105 protocol, source, checkpoint, calibrator, identity, and validator-consistency checks passed at every step. Table 5 outcomes were 15/17 at 5000, 16/17 at 7500, 17/17 at 10000, 16/17 at 12500, and 16/17 at 15000. Step 5000 failed effective rank at 24.865449 and FreeSolv RMSE at 1.390450. Steps 7500, 12500, and 15000 failed only FreeSolv, with RMSE 1.372273, 1.322887, and 1.303465 respectively. Step 10000 alone passed every gate, including FreeSolv RMSE 1.297164. The repository fail-closed validator independently accepted only step 10000 and rejected the other checkpoints for the expected first failing gate. The Table-5-style CSV is manuscript/Table5_full_promotion_trajectory_steps_5k-15k.csv relative to the gMolAI parent, and the full hash-bound audit is in the seed-42 run under promotion-trajectory-table5-rev1/promotion_trajectory_audit.json.

## Outcome

- Signal: useful

## Source Nodes

- Semantic Promotion Suite
- _validate_promotion_quality()
- run_representation_probes()
- benchmark_moleculenet()