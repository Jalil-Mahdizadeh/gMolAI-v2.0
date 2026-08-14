# Step 2: frozen-representation decoder feasibility

This isolated step trains only a new autoregressive SMILES decoder conditioned
on the frozen released gMolAI 384-D weight-3 hybrid. The gMolAI encoder,
checkpoint, calibrator, and public representation are read-only inputs.

The study is limited to zero-perturbation reconstruction and performs no
MMP-direction or derivative generation. Its frozen decision is **NO-GO for a
faithful inverse**, while the explicit controls demonstrate strong condition
use: the decoder redirects output toward the molecule supplying the embedding.

- `RESULTS.md`: held-out metrics, controls, latent consistency, and limitations.
- `DECISION.md`: audited frozen GO / NO-GO gates.
- `DESIGN.md`: repository evidence and decoder architecture rationale.
- `PROTOCOL.md`: frozen scientific design and decision thresholds.
- `config/protocol.json`: machine-readable architecture and evaluation settings.
- `inputs/manifest.json`: immutable input hashes and forbidden inputs.
- `scripts/`: preparation, training, evaluation, reporting, and verification.
- `prepared/`: train-only tokens and deterministic development split.
- `checkpoints/`: decoder-only training states and the 113 MB
  inference-only `decoder_inference.pt` (no optimizer or gMolAI weights).
- `outputs/`: curves, reconstructions, metrics, figures, examples, and checksums.
- `state/`: stage seals, logs, caches, and final integrity record.

## Reproduction

From the repository root, run:

```bash
bash deriv-gen/step-02-decoder-feasibility/scripts/run_study.sh
```

The runner performs the component test, preparation, one-GPU pilot,
decoder-only fit, held-out evaluation, report generation, and read-only
integrity verification. It resumes sealed stages and binds the released
checkpoint, calibrator, and source repository read-only. Large generated arrays
and decoder checkpoints remain local; their identities are recorded by SHA-256
in the stage seals and output ledger.
