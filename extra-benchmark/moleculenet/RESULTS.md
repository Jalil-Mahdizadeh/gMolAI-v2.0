# MoleculeNet development panel plus HIV: completed results

- **Completion date:** 12 August 2026
- **Execution:** frozen neural inference plus fold-local linear-probe fitting
- **Primary population:** all-model common coverage with inherited scaffold roles
- **Development evidence:** BACE, BBBP, ESOL, FreeSolv, Lipophilicity
- **Confirmatory evidence:** external post-selection HIV endpoint

This additive benchmark did not train or fine-tune any neural encoder and cannot
alter the promoted gMolAI seed-42/10k checkpoint, train-only calibrator, or
released 384-D representation. It is separate from the locked internal
pretraining-test representation benchmark.

## Completion and integrity

- Corrected Slurm job `1219522` completed on n501 with exit code `0:0` in
  1:16:04, using one GH200 GPU.
- All seven native-dimensional matrices passed ordered-identity, shape, dtype,
  finiteness, nonzero-norm, and visible-GPU checks.
- The completion seal binds 45,504 common rows, all six datasets, all seven
  representations, the frozen protocol digest, the result summary, and the
  output checksum ledger.
- All 28 entries in `outputs/SHA256SUMS` independently verified.
- The selected checkpoint and calibrator hashes are unchanged; the completion
  record explicitly reports no neural training/fine-tuning and no checkpoint or
  calibrator modification.

The first job (`1219337`) stopped before MolCLR-GIN inference because the
launcher replaced its container-native Python path. Five already completed
matrices were retained only after revalidation. The runtime-only repair and
deterministic smoke tests are documented in `PROTOCOL.md` and `protocol.json`.

## Realized common coverage

| Dataset | Prepared molecules | All-model common | Removed | Common fraction |
|---|---:|---:|---:|---:|
| BACE | 1,513 | 1,502 | 11 | 99.273% |
| BBBP | 1,860 | 1,833 | 27 | 98.548% |
| ESOL | 1,116 | 1,116 | 0 | 100.000% |
| FreeSolv | 639 | 638 | 1 | 99.844% |
| Lipophilicity | 4,198 | 4,187 | 11 | 99.738% |
| HIV | 37,225 | 36,228 | 997 | 97.322% |
| **Total** | **46,551** | **45,504** | **1,047** | **97.751%** |

gMolAI, Morgan, MolCLR-GIN, and KERMT v2 accepted all 46,551 prepared rows.
MolAI accepted 45,505 and rejected 1,046, chiefly because its released encoder
supports at most 109 tokens and lacks several encountered characters. MoLFormer
and SMI-TED-Light each accepted 46,488 and rejected the same 63 rows because of
their 202-token/lossless-tokenization constraints. Their rejected set was not
identical to MolAI's, producing a 45,504-row intersection. The benchmark did not
truncate inputs or enable MolAI's silent unknown-character-to-zero fallback.

## Primary paired endpoint results

Classification values are mean ROC-AUC; regression values are mean RMSE, where
lower is better. Each value is mean ± population SD over the same ten inherited,
overlapping outer scaffold splits. Bold marks the best mean, not a statistical
significance claim.

| Dataset (metric) | gMolAI | Morgan | MolAI | MoLFormer | SMI-TED | MolCLR | KERMT v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BACE (ROC-AUC) | 0.8410 ± 0.0269 | 0.8557 ± 0.0371 | 0.8017 ± 0.0301 | **0.8632 ± 0.0312** | 0.8575 ± 0.0238 | 0.7971 ± 0.0296 | 0.8412 ± 0.0348 |
| BBBP (ROC-AUC) | 0.8723 ± 0.0464 | 0.8546 ± 0.0212 | 0.8563 ± 0.0370 | 0.8815 ± 0.0380 | **0.8952 ± 0.0251** | 0.8201 ± 0.0397 | 0.8912 ± 0.0257 |
| ESOL (RMSE) | **0.7314 ± 0.0587** | 1.5991 ± 0.2060 | 0.8598 ± 0.0876 | 0.9780 ± 0.0547 | 0.8194 ± 0.0605 | 1.3684 ± 0.1045 | 1.0982 ± 0.1577 |
| FreeSolv (RMSE) | **1.3055 ± 0.1950** | 2.3666 ± 0.4589 | 1.4050 ± 0.1558 | 1.7008 ± 0.1151 | 1.3077 ± 0.0890 | 2.0993 ± 0.1449 | 1.8796 ± 0.2619 |
| Lipophilicity (RMSE) | 0.8086 ± 0.0205 | 0.9285 ± 0.0272 | 0.9725 ± 0.0332 | **0.8043 ± 0.0244** | 0.8402 ± 0.0183 | 0.9474 ± 0.0417 | 0.8379 ± 0.0637 |
| HIV (ROC-AUC) | 0.7507 ± 0.0191 | 0.7357 ± 0.0286 | 0.6852 ± 0.0239 | **0.7572 ± 0.0314** | 0.7293 ± 0.0229 | 0.6949 ± 0.0287 | 0.7544 ± 0.0282 |

The exact means, dispersions, ranks, per-split values, and paired gMolAI
win/tie/loss counts are in `outputs/common_panel_primary_metrics.csv` and
`outputs/common_panel_all_metrics.csv`.

## Development-panel interpretation

- **BACE:** MoLFormer had the highest mean ROC-AUC (0.8632). gMolAI ranked fifth
  at 0.8410: it was below MoLFormer, SMI-TED-Light, and Morgan on all or most
  paired splits, essentially tied in mean with KERMT v2 (difference 0.0002),
  and above MolAI and MolCLR-GIN. BACE therefore does not support a gMolAI
  superiority claim.
- **BBBP:** SMI-TED-Light led at 0.8952, followed by KERMT v2 and MoLFormer.
  gMolAI ranked fourth at 0.8723, while exceeding MolAI, Morgan, and MolCLR-GIN
  in mean. Paired split outcomes varied, so the rank should remain descriptive.
- **ESOL:** gMolAI had the lowest mean RMSE (0.7314). It beat SMI-TED-Light on
  9/10 paired splits, MolAI on 9/10, and each of MoLFormer, KERMT v2,
  MolCLR-GIN, and Morgan on 10/10. This is the clearest favorable endpoint in
  the common-panel comparison.
- **FreeSolv:** gMolAI (1.3055) and SMI-TED-Light (1.3077) were practically tied
  in mean; gMolAI won 6/10 paired splits. Both were ahead of the remaining
  comparators. The 0.0021 mean difference is too small to frame as decisive.
- **Lipophilicity:** MoLFormer (0.8043) and gMolAI (0.8086) were again close,
  with a 5/5 paired split balance. gMolAI was ahead of KERMT v2,
  SMI-TED-Light, Morgan, MolCLR-GIN, and MolAI in mean.

Thus, gMolAI is strongest on the descriptor-related regression panel but not
universally strongest. Sequence encoders lead the two small classification
development datasets, and the most defensible summary is endpoint-dependent
complementarity rather than blanket superiority.

## Descriptor-only context

Historical full-panel controls are deliberately kept separate from the primary
all-model-common ranking. On the unchanged full panels, the 13 frozen auxiliary
descriptors yielded RMSE 0.8771 on ESOL, 1.6019 on FreeSolv, and 1.0071 on
Lipophilicity, versus gMolAI RMSE 0.7314, 1.2971, and 0.8125, respectively.
This indicates that the descriptor channel alone does not reproduce gMolAI's
regression performance under the same linear-probe family. It does not show
that the channel is unimportant: gMolAI was explicitly trained with descriptor
supervision, so descriptor-aware pretraining remains a plausible contributor.

The full gMolAI, Morgan, and descriptor-control values and provenance paths are
in `outputs/full_panel_reference_controls.csv`; those values must not be mixed
into common-panel ranks.

## HIV post-selection confirmation

HIV is not part of the development/promotion panel and was not part of the
locked internal pretraining train/validation/test split. On the 36,228-molecule
common HIV panel, MoLFormer led in mean ROC-AUC (0.7572), followed by KERMT v2
(0.7544) and gMolAI (0.7507). The gaps from gMolAI were 0.0065 and 0.0036,
respectively. gMolAI exceeded Morgan (0.7357) on 8/10 paired splits and
SMI-TED-Light, MolCLR-GIN, and MolAI on 10/10.

The result supports competitive post-selection external performance, not
universal leadership. It also does not by itself establish molecule-level
novelty relative to pretraining; that question is governed by the separate
exact exposure audit.

## Observed representation-export timings

These timings include load, warm-up, preprocessing, inference, and export for
45,504 rows. They are retained as systems provenance only.

| Representation | Dim. | Wall time (s) | Rows/s | Peak GPU allocation |
|---|---:|---:|---:|---:|
| gMolAI | 384 | 135.5 | 335.8 | 0.78 GiB |
| Morgan radius-2 | 2,048 | 6.8 | 6,690.6 | CPU |
| MolAI epoch 6 | 512 | 12.2 | 3,719.6 | 1.15 GiB |
| MoLFormer | 768 | 19.3 | 2,359.0 | 0.57 GiB |
| SMI-TED-Light | 768 | 63.0 | 722.6 | 0.92 GiB |
| MolCLR-GIN | 512 | 31.8 | 1,431.7 | 1.56 GiB |
| KERMT v2 | 512 | 149.9 | 303.5 | 0.84 GiB |

Morgan is CPU-based, neural implementations have different preprocessing and
batching, the measurements include non-inference work, and the recovery used
two GH200 nodes. These values are therefore not a publication-quality speed
leaderboard. The dedicated single-GPU benchmark has since completed on the
49,844-molecule all-model-common locked-test panel with common timing
boundaries and batch sizes 64, 128, 256, and 512; see
[`../speed/RESULTS.md`](../speed/RESULTS.md). That controlled benchmark
supersedes these incidental export timings for throughput comparisons. Its
values remain descriptive single-pass point estimates without confidence
intervals, as bounded in the frozen speed protocol.

## Bounded conclusions

- gMolAI ranked first by mean on ESOL and FreeSolv, second on Lipophilicity,
  third on confirmatory HIV, fourth on BBBP, and fifth on BACE.
- Its largest and most consistent advantages occur on the three regression
  endpoints; those results should be interpreted in light of descriptor-aware
  pretraining and the descriptor-only control.
- MoLFormer, SMI-TED-Light, and KERMT v2 provide meaningful strong baselines;
  the results strengthen the manuscript precisely because they expose both
  gMolAI's advantages and its endpoint-specific limitations.
- The ten outer folds overlap. Their SDs and paired win counts are descriptive,
  not independent-replication uncertainty estimates.
- Native dimensions differ, so the comparison evaluates released/frozen
  representations with a common probe family, not parameter- or
  dimension-matched encoders.
- BACE/BBBP/ESOL/FreeSolv/Lipophilicity remain selection-conditioned evidence;
  only HIV is external post-selection endpoint confirmation in this phase.

Large embedding arrays and runtime logs remain local. The compact CSV/JSON
results, identity/split manifests, completion state, and checksum ledger retain
the full audit trail.
