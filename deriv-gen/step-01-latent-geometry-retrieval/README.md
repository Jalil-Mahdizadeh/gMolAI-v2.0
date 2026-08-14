# Step 1: latent geometry and derivative retrieval

This completed study asks whether movements in the frozen promoted gMolAI space
can retrieve chemically related molecules and whether matched-molecular-pair
(MMP) transformations create transferable displacement directions.

The study uses:

- 100,000 deterministic train molecules to fit geometry and MMP directions;
- 50,000 independent validation molecules as query seeds and retrieval bank;
- the promoted train-only calibrator;
- graph-256, mean-node-128, and unweighted standardized hybrid-384 spaces;
- exact molecule identities from the immutable probe cache.

It does **not** use the locked internal test partition, endpoint labels, model
training, or a molecular decoder.

## Contents

- `PROTOCOL.md`: scientific design, controls, and predeclared gates.
- `PROTOCOL_AMENDMENT_01.md`: pre-outcome MMP support amendment and audit.
- `config/protocol.json`: machine-readable final parameters and amendment metadata.
- `inputs/manifest.json`: immutable paths and SHA-256 identities.
- `scripts/`: reproducible launcher, analysis, and verifier.
- `intermediate/`: derived molecule, fragmentation, MMP, and query tables.
- `outputs/`: result tables, figures, examples, summaries, and hash ledger.
- `state/`: log, runtime provenance, completion seal, and isolated caches.
- `RESULTS.md`: bounded interpretation produced from the completed run.

Run `scripts/verify_day1.py` through the pinned container after execution to
check input identities, result completeness, and the output hash ledger.

