# MoleculeNet development panel and HIV benchmark

This directory contains the completed second-phase frozen-encoder benchmark of
gMolAI, Morgan radius-2 fingerprints, MolAI epoch 6, MoLFormer,
SMI-TED-Light, MolCLR-GIN, and KERMT v2.

**Status:** complete and checksum-validated on 12 August 2026. Slurm job
`1219522` exited successfully on one Arrhenius GH200 GPU. The primary paired
comparison uses 45,504 all-model-common molecules across BACE, BBBP, ESOL,
FreeSolv, Lipophilicity, and HIV. See [`RESULTS.md`](RESULTS.md) for the
scientific findings and [`PROTOCOL.md`](PROTOCOL.md) for the frozen design.

The evidence roles remain distinct:

- BACE, BBBP, ESOL, FreeSolv, and Lipophilicity are the existing
  selection-conditioned development/promotion panel;
- HIV is a separate external post-selection confirmatory endpoint; and
- this benchmark is not the locked internal pretraining-test evaluation in
  `../test-partition/`.

The workflow:

1. verifies immutable datasets, selected checkpoint/calibrator, reference
   results, scripts, and all SIF images by SHA-256;
2. reconstructs the authoritative canonical molecules and ten accepted outer
   scaffold-group splits;
3. verifies the five development datasets against the prior exact outer/inner
   split-identity audit and HIV against its authoritative outer seeds/counts;
4. screens every frozen adapter, then forms one all-model coverage
   intersection without regenerating or reassigning any split;
5. exports native-dimensional FP32 representations with exactly one visible
   GPU for each neural encoder;
6. fits identical fold-local Ridge or balanced logistic probes with frozen
   inner-CV assignments; and
7. aggregates paired results and checksum-seals all generated outputs.

No neural model was trained or fine-tuned. The promoted seed-42/10k checkpoint,
its train-only calibrator, and the released 384-D representation were not
changed.

## Main artifacts

- `protocol.json`: machine-readable frozen protocol and runtime-amendment
  provenance;
- `inputs/dataset_manifest.json` and `inputs/common_manifest.json`: source,
  identity, coverage, and inherited split manifests;
- `outputs/common_panel_primary_metrics.csv`: concise paired leaderboard;
- `outputs/common_panel_all_metrics.csv`: all metrics and per-split values;
- `outputs/full_panel_reference_controls.csv`: historical full-panel
  gMolAI/Morgan and 13-descriptor controls, kept separate from common-panel
  rankings;
- `outputs/results/*.json`: complete per-model, per-dataset, per-split records;
- `outputs/coverage.csv`: adapter-specific support and rejection reasons;
- `outputs/encoding_runtime_observed.csv`: provenance timing, not a controlled
  throughput leaderboard;
- `state/COMPLETE.json`: sealed completion record; and
- `outputs/SHA256SUMS`: checksum ledger for all 28 local output artifacts.

Large TSV/NPZ inputs, native embedding matrices, and scheduler logs remain
local and intentionally ignored. Compact manifests, results, metadata, and
audit documents are versionable.
