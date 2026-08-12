# Additional frozen-encoder benchmarks

This directory holds additive, no-training comparisons of the promoted gMolAI
seed-42/10k representation with frozen 1D/2D molecular encoders and Morgan
fingerprints.

- `test-partition/` contains the completed locked internal pretraining-test
  workflow and its compact audited results.
- `moleculenet/` is reserved for the later external endpoint-benchmark
  workflow; it has not been executed.

The two evidence sources are deliberately separated. The locked internal
partition has no endpoint labels and is used only for common representation
diagnostics and systems measurements. MoleculeNet uses endpoint labels and the
existing frozen nested scaffold-split protocol.

No script in this directory trains, resumes, fine-tunes, or modifies a neural
model.

The locked-test benchmark completed on 12 August 2026. See
[`test-partition/RESULTS.md`](test-partition/RESULTS.md) for the coverage,
representation diagnostics, clustering comparison, integrity record, and
bounded interpretation.
