# VSDS-vd TrueDecoy_gap ligand retrieval

This is the bounded ligand-based virtual-screening benchmark supporting the
gMolAI manuscript. It compares seven already-frozen molecular representations
on 5-shot active retrieval, a secondary 1-shot analysis, and one
scaffold-excluded robustness analysis. See `PROTOCOL.md` and `protocol.json`
before execution.

## Reproduction

The runner expects the two checksum-pinned source files at:

```text
inputs/raw/VSDS_vd-v3.rar
inputs/raw/VSDS_TrueDecoy_gap_Supplementary_Data_1.xlsx
```

On an Arrhenius GPU node, execute:

```bash
sbatch extra-benchmark/ligand-retrieval-vsds/run_lbvs.sbatch
```

The same script can be run directly on a single-GPU interactive node:

```bash
bash extra-benchmark/ligand-retrieval-vsds/run_lbvs.sbatch
```

It uses the dedicated SIF for each model, requires exactly one visible GPU per
neural export, and is restart-safe: completed artifacts are validated before
reuse and are never silently overwritten.

The runner performs no training, fine-tuning, target-specific fitting, or
representation selection. It first completes label-blind adapter screening and
all seven validated exports; only then does it freeze common support, target
eligibility, and deterministic anchors. Retrieval results cannot therefore
change the evaluated population.

The final completion seal is `state/COMPLETE.json`. Manuscript-facing tables
are under `results/tables`, figures under `figures`, and every plotted value is
preserved under `figures/source-data`. `RESULTS.md` is a generated compact
reading copy, while `results/SHA256SUMS` binds every retained analysis artifact.

The additional all-model ROC figure is a visualization of the already-frozen
primary five-shot ROC-AUC endpoint. Its macro curve averages the 20 draws within
each target before weighting the 70 targets equally. It introduces no new model
selection or inferential endpoint.

