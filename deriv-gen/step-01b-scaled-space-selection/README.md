# Step 1b: scaled latent-space selection

This study extends Day 1 without training or modifying gMolAI. It keeps the
promoted seed-42/step-10,000 checkpoint, immutable train-only calibrator, and
released embedding definition unchanged.

It uses a deterministic 1,000,000-molecule pretraining-train sample for MMP
mining and the same independent 50,000-molecule validation bank used by Day 1.
Five coordinate spaces are compared on identical transformations, queries,
candidate molecules, controls, random seeds, and metrics:

1. `graph_256`;
2. `mean_node_128`;
3. `hybrid_w1`;
4. `released_hybrid_w3`;
5. `hybrid_w6`.

The released representation remains `[graph_std, 3 * mean_node_std]` regardless
of which space is selected as the molecular-edit control geometry. The final
decision therefore distinguishes the immutable decoder-conditioning vector
from the latent space used to define edit directions.

## Status

Completed and independently verified on one NVIDIA GH200. The study mined
1,461,391 independent train core-transformation observations and evaluated
9,074 unseen-core alignment observations plus 2,048 identical retrieval queries
per space. The frozen rule selected `released_hybrid_w3` as the edit-control
space. Mean-node-128 remains the directional-alignment leader but is not the
unique overall winner. Weight 3 improves alignment over weight 1; its small
exact-retrieval increase is not statistically resolved.

## Contents

- `PROTOCOL.md`: frozen scientific design and selection rule.
- `config/protocol.json`: machine-readable parameters.
- `inputs/manifest.json`: immutable external input identities.
- `scripts/`: resumable export, mining, analysis, smoke-test, report-refresh,
  and verification code.
- `exports/`: generated 1M train embedding export.
- `intermediate/`: molecule, fragmentation, MMP, direction, and query records.
- `outputs/`: machine-readable tables, raw results, figures, and checksums.
- `state/`: stage seals, logs, isolated caches, and final completion record.
- `DECISION.md`: concise decision report generated after the study completes.

Run from the repository root:

```bash
bash deriv-gen/step-01b-scaled-space-selection/scripts/run_study.sh
```

