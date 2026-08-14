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
- `tdc-admet/` contains the final active study: a preregistered, complete
  22-endpoint TDC ADMET comparison of the same seven frozen representations,
  with a 13-descriptor diagnostic control.

The evidence sources are deliberately separated. The locked internal partition
has no endpoint labels and supports common representation diagnostics.
MoleculeNet/HIV evaluates endpoint prediction. The speed workflow measures
systems throughput under one fixed host, GPU and resource envelope. TDC ADMET
tests broader property-prediction transfer while remaining scientifically
separate from decoder-generated candidates.

No script in this directory trains, resumes, fine-tunes or modifies a neural
model.

See `test-partition/RESULTS.md`, `moleculenet/RESULTS.md` and
`speed/RESULTS.md` for completed audited results. The TDC study's frozen design
and live status are in `tdc-admet/PROTOCOL.md` and `tdc-admet/README.md`.
