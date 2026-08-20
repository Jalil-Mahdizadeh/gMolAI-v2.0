# Additional frozen-encoder benchmarks

This directory holds additive, no-training comparisons of the promoted gMolAI
seed-42/10k representation with frozen 1D/2D molecular encoders and Morgan
fingerprints.

- `test-partition/` contains the completed locked internal pretraining-test
  workflow and its compact audited results.
- `moleculenet/` contains the completed MoleculeNet plus HIV endpoint
  benchmark under the frozen nested scaffold-split protocol.
- `speed/` contains the completed optimized single-GPU encoding-throughput
  comparison on the 49,844-molecule common locked-test panel at batch sizes 64,
  128, 256 and 512.
- `tdc-admet/` contains the completed preregistered broad-transfer study: a
  complete 22-endpoint TDC ADMET comparison of the same seven frozen
  representations, with a separate 13-descriptor diagnostic control.
  SMI-TED-Light ranked first and gMolAI second by the category-balanced
  summary; gMolAI had the most endpoint wins.
- `clustering/` contains the completed external label-structure, local
  retrieval, and continuous-neighborhood benchmark on balanced ClassyFire-25
  and QMugs panels. The result is deliberately bounded: different
  representations lead different clustering and retrieval criteria.

The evidence sources are deliberately separated. The locked internal partition
has no endpoint labels and supports common representation diagnostics.
MoleculeNet/HIV evaluates endpoint prediction. The speed workflow measures
systems throughput under one fixed host, GPU and resource envelope. TDC ADMET
tests broader property-prediction transfer. The external clustering study tests
categorical label recovery, same-subclass retrieval, and neighborhood
preservation. Both remain scientifically separate from decoder-generated
candidates.

No script in this directory trains, resumes, fine-tunes or modifies a neural
model.

See `test-partition/RESULTS.md`, `moleculenet/RESULTS.md`, `speed/RESULTS.md`,
`tdc-admet/RESULTS.md`, and `clustering/RESULTS.md` for completed audited
results. The TDC study's frozen design and execution guide are in
`tdc-admet/PROTOCOL.md` and `tdc-admet/README.md`; the external clustering
design is in `clustering/PROTOCOL.md`.
