# Locked internal test-partition encoder benchmark: completed results

- **Completion date:** 12 August 2026
- **Execution:** frozen inference and post-processing only
- **Evidence source:** 50,000-molecule sample from the 1,088,766-molecule
  locked internal pretraining test partition

This retrospective additive benchmark cannot alter the promoted gMolAI
seed-42/10k checkpoint, train-only calibrator, or released 384-D representation.
It contains no endpoint labels and is separate from the unexecuted MoleculeNet
benchmark.

## Execution and integrity

- Job 1215114 on n139 generated and validated every native comparator matrix,
  then exited during common-probe post-processing because the derived gMolAI
  payload omitted the authoritative hybrid block-width metadata.
- The guarded repair preserves the source metadata and can replace an existing
  derived gMolAI payload only when every tensor, molecular identity, and all
  other metadata already match exactly.
- Recovery job 1215856 on n566 reused the existing matrices, repaired only the
  two derived metadata records, completed all seven probes, and exited 0:0 in
  17 minutes 25 seconds. It did not execute an encoder.
- `state/COMPLETE.json` records inference-only completion, no training, no model
  weight changes, no gMolAI checkpoint/calibrator changes, and no MoleculeNet
  execution.
- All 111 entries in `outputs/SHA256SUMS` passed an independent SHA-256 check.

## Realized common coverage

| Panel | Attempted | All-model common | Common fraction |
|---|---:|---:|---:|
| Validation qualification | 128 | 127 | 99.2188% |
| Training topology probe | 10,000 | 9,958 | 99.5800% |
| Locked-test representation panel | 50,000 | 49,844 | 99.6880% |

gMolAI, Morgan, MoLFormer, SMI-TED-Light, MolCLR-GIN, and KERMT v2 accepted
every attempted train/test molecule. MolAI accepted 9,958/10,000 training rows
and 49,844/50,000 test rows. Its frozen tokenizer rejected unsupported `B`
and/or `i` characters; the benchmark did not enable the supplied legacy
unknown-character-to-zero fallback. Every reported cross-model metric uses the
same ordered common identities. The common identity SHA-256 digests are
`0e9477476ed5aade943809928ab96859a143087731f2843e2b8d946ae7b94417`
for training and
`a9e3e63eade2c542fef670184ed1a34e054167d085ad299ee080217d7301e237`
for locked test.

## Common representation diagnostics

| Encoder | Dim. | Topology mean R2 | Scaffold-disjoint mean R2 | Effective-rank fraction | Morgan recall@10 | Scaffold ARI | Scaffold NMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| gMolAI | 384 | **0.9705** | **0.9729** | 0.0798 | 0.2026 | 0.3578 ± 0.0112 | 0.7246 ± 0.0050 |
| Morgan | 2,048 | 0.8163 | 0.8173 | **0.3506** | 0.9700 | **0.3956 ± 0.0354** | **0.7928 ± 0.0162** |
| MolAI epoch 6 | 512 | 0.9561 | 0.9597 | 0.1456 | 0.1152 | 0.1326 ± 0.0030 | 0.3692 ± 0.0083 |
| MoLFormer | 768 | 0.9141 | 0.9169 | 0.1508 | 0.2363 | 0.3072 ± 0.0094 | 0.6569 ± 0.0039 |
| SMI-TED-Light | 768 | 0.9617 | 0.9640 | 0.0898 | 0.1875 | 0.2750 ± 0.0047 | 0.6035 ± 0.0093 |
| MolCLR-GIN | 512 | 0.6231 | 0.6142 | 0.0461 | 0.1360 | 0.1381 ± 0.0043 | 0.4027 ± 0.0035 |
| KERMT v2 | 512 | 0.9245 | 0.9322 | 0.2172 | **0.3313** | 0.3620 ± 0.0083 | 0.7548 ± 0.0063 |

Topology values are means across the unchanged 13-target Ridge probe. The
scaffold-disjoint subset contains test scaffolds absent from the common
probe-training panel and comprises 98.4151% of the test panel. Morgan recall is
a similarity-to-Morgan diagnostic rather than an absolute quality criterion.

Clustering used 16,360 molecules from the 32 most frequent recurring
Bemis-Murcko scaffolds, each occurring at least five times. Values are means ±
population standard deviations across five K-means seeds, each with 20
initializations. Inputs were row-L2-normalized before standard Euclidean
K-means; centroids were not constrained or renormalized, so this was not
spherical K-means despite immutable legacy JSON key names.

## Bounded interpretation

- gMolAI ranked first on the descriptor/topology probe, including the
  scaffold-disjoint subset. This is consistent with a descriptor-aware
  representation, not evidence of endpoint superiority.
- Morgan retained the highest normalized effective rank and the strongest
  recurring-scaffold clustering. The locked-test clustering reversal therefore
  persists and must not be suppressed.
- KERMT v2 was the closest learned comparator for scaffold clustering and had
  the highest learned-encoder Morgan-neighbour recall. Its mean ARI was close
  to gMolAI, while its NMI was higher.
- The benchmark measures representation geometry and recoverable molecular
  descriptors. It does not evaluate BACE, BBBP, ESOL, FreeSolv,
  Lipophilicity, HIV, or any other endpoint.
- Recorded comparator export timings include model loading, warm-up, and
  serialization; gMolAI was reused from its authoritative export. They are not
  a publication-grade, apples-to-apples throughput comparison. The completed
  frozen single-GPU benchmark in
  [`../speed/RESULTS.md`](../speed/RESULTS.md) supersedes these incidental
  timings for controlled throughput comparisons.

## Versioned audit artifacts

- `protocol.json` and `PROTOCOL.md`: frozen machine-readable and narrative
  protocol;
- `outputs/test_partition_summary.csv` and `.json`: compact cross-model
  results;
- `outputs/probes/*.json`: full per-target and per-seed diagnostics;
- `outputs/coverage_*.csv`: validation, training, and locked-test coverage;
- `inputs/panel_manifest.json`: source-panel identity and hash manifest;
- `state/preflight.json`, `state/packaged_*.json`, `state/status.json`, and
  `state/COMPLETE.json`: source verification, derived-payload identities,
  execution chronology, and sealed completion state;
- `outputs/SHA256SUMS`: complete local artifact hash ledger.

Large input panels, native matrices, PyTorch payloads, qualification arrays,
and scheduler logs remain intentionally untracked.
