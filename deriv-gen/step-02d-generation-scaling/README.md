# Step 2d: frozen decoder generation scaling

This directory is a self-contained, preregistered no-training study of chemical
yield as raw candidate budgets scale from 50 to 1,000 per seed. See
`PROTOCOL.md` for frozen rules, `RESULTS.md` for findings, `DECISION.md` for the
bounded decision, and `outputs/tables/` for machine-readable results.

Run `scripts/run_study.sh` inside a four-GPU SLURM allocation, or submit
`scripts/submit_step2d.slurm`.

<!-- STEP2D_SA_EXTENSION:START -->
## Synthetic-accessibility extension

The post-completion no-generation comparison is defined in
PROTOCOL_SA_EXTENSION.md. Machine-readable outputs are under outputs/tables and
the three SA figures plus corrected validity figure are under outputs/figures.
<!-- STEP2D_SA_EXTENSION:END -->
