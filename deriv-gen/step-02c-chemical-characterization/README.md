# Step 2c: frozen candidate-set chemical audit

This directory contains the no-training chemical characterization of all
10,000 frozen Step-2b correct-condition seeds. The bounded
outcome is **NOT SUPPORTED** (`useful_local_analogue_generator_not_supported`).

Run `scripts/run_study.sh` from the repository checkout to reproduce the audit
inside the pinned gMolAI container. The runner reads Step-1b/Step-2b artifacts
without modifying them and writes only inside this directory.

- `PROTOCOL.md` and `DESIGN.md`: frozen definitions and denominator discipline.
- `RESULTS.md` and `DECISION.md`: scientific report and bounded conclusion.
- `config/`: frozen analysis settings.
- `inputs/`: SHA-256-bound read-only inputs and source provenance.
- `scripts/`: registration, component tests, audit, reporting, and verification.
- `intermediate/`: independently audited chemistry and exact one-cut fragments.
- `outputs/tables/`: candidate-, seed-, transformation-, and summary-level data.
- `outputs/figures/`: concise PNG and SVG figures.
- `state/`: registration, stage, environment, and completion seals.

No model training/execution, candidate regeneration, latent perturbation, or
MMP-directed generation occurs here.
