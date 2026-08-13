# Locked internal test-partition encoder benchmark

This is the executable companion to `PROTOCOL.md`. It compares frozen
representations on the authoritative 50,000-molecule sample from the
1,088,766-molecule locked internal pretraining test partition, using the same
10,000 training-partition identities for the 13-target Ridge probe.

**Status:** complete and checksum-validated on 12 August 2026. The realized
all-model common panels contain 49,844 locked-test molecules and 9,958
probe-training molecules. Scientific results and interpretation are in
[`RESULTS.md`](RESULTS.md).

The workflow is fail-closed and resumable:

1. verify every source artifact and SIF against `protocol.json`;
2. reconstruct canonical identities from the immutable deduplicated corpus;
3. qualify every adapter on a validation-partition fixture;
4. screen the complete frozen train/test panels without silent truncation;
5. construct one common-coverage set across all comparators;
6. export native-dimensional FP32 representations;
7. run the repository's existing topology, geometry, neighbour, and scaffold
   clustering diagnostics on identical identities; and
8. write coverage, timing, integrity, and result summaries.

The five-development-dataset MoleculeNet evaluation plus HIV confirmation is a
separate completed workflow; see
[`../moleculenet/RESULTS.md`](../moleculenet/RESULTS.md).

## Launch on Arrhenius

From the repository root:

```bash
sbatch --account=naiss2025-3-10-gpu \
  extra-benchmark/test-partition/run_test_partition.sbatch
```

The job requests one GH200 GPU. Generated panels, large embeddings and
payloads, qualification arrays, and logs remain local under this directory.
The source scripts, frozen protocol, compact result summaries, seven
probe records, and integrity metadata are versioned.

## Audit status

The initial inference job, 1215114, generated and validated all comparator
matrices but failed during common-probe post-processing because the derived
gMolAI payload omitted authoritative hybrid block metadata. The guarded repair
in `package_payloads.py` permits replacement only when every tensor, identity,
and all metadata except `embedding_parameters` already match. Recovery job
1215856 reused the matrices, restored only those parameters, completed all
probes, and exited successfully. It did not execute an encoder.

`state/COMPLETE.json` was written only after all seven model probes and the
integrity checks passed. All 111 entries in `outputs/SHA256SUMS` were then
independently verified. The promoted seed-42/10k checkpoint and calibrator were
not changed. These jobs were limited to the locked internal test-partition
workflow. The independent MoleculeNet plus HIV benchmark subsequently completed
with its own checksum-validated record, and the controlled single-GPU
throughput follow-up also completed; see
[`../moleculenet/RESULTS.md`](../moleculenet/RESULTS.md) and
[`../speed/RESULTS.md`](../speed/RESULTS.md), respectively.
