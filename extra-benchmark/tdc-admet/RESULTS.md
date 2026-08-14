# TDC ADMET frozen-representation benchmark: completed results

- **Completion date:** 14 August 2026
- **Execution:** frozen inference plus fold-local shallow-probe fitting
- **Host/allocation:** n42, one NVIDIA GH200 120GB, interactive Slurm job
  `1255034`
- **Execution commit:** `bbc356d8a3c94e63f513c579157603cd4d50d649`
- **Primary population:** seven-representation common support across all 22
  endpoints

## Scientific result

SMI-TED-Light ranked first by the preregistered category-balanced summary.
gMolAI ranked second on both the complete 22-endpoint analysis and the
predeclared 19-endpoint selection-robust sensitivity. gMolAI nevertheless had
the largest number of endpoint wins: 9/22 in the complete panel and 8/19 after
excluding the three endpoints reused or closely related to prior development
evidence. Its median endpoint rank was 2 in both analyses.

This is strong evidence that the frozen released gMolAI representation supports
broad ADMET transfer with simple probes. It is not universal-superiority
evidence: SMI-TED-Light had the better balanced cross-category profile, and
several individual endpoints expose clear gMolAI weaknesses.

## Completion and integrity

- Direct execution inside the existing n42 allocation completed with exit code
  0. No neural encoder was trained, resumed, fine-tuned, or modified.
- All seven primary matrices and the separate descriptor matrix passed ordered
  identity, shape, dtype, support, and checksum validation.
- The result set contains 176 model-endpoint records and 880 five-seed selected
  test outcomes. Every recorded metric is finite and every selected
  regularization value belongs to its frozen grid.
- All 36 entries in `outputs/SHA256SUMS`, including the locally retained large
  embedding matrices, passed independent SHA-256 verification.
- `scripts/verify_results.py` independently rechecks the checksum ledger,
  completion seals, common population, split disjointness, seed outcomes,
  hyperparameter grids, summary ranks, row counts, and diagnostic amendment.
  Its completed-run output is:

  ```text
  PASS: tdc-admet-complete; checksums=36; models=8; endpoints=22; seed_outcomes=880
  ```

The first diagnostic descriptor export stopped before creating a matrix and
before any endpoint probe existed because RDKit returned exactly 12 undefined
partial-charge values on six qualified identities. The committed runtime
amendment permits only those exact values and median-imputes them within the
current training fold. The seven primary representations, common population,
splits, labels, grids, metrics, and ranks were unchanged.

## Realized common population

The DOI-backed snapshot contains 81,809 labeled occurrences. Repository
canonicalization and molecular policy retained 78,699 occurrences representing
44,143 unique identities. The all-model support intersection retained 78,131
occurrences and 43,730 identities: 99.278% and 99.064% of the policy-qualified
populations, respectively.

| Representation | Supported identities / 44,143 | Identity coverage | Supported occurrences / 78,699 | Occurrence coverage |
|---|---:|---:|---:|---:|
| gMolAI | 44,143 | 100.000% | 78,699 | 100.000% |
| Morgan radius-2 | 44,143 | 100.000% | 78,699 | 100.000% |
| MolAI epoch 6 | 43,730 | 99.064% | 78,131 | 99.278% |
| MoLFormer | 44,125 | 99.959% | 78,678 | 99.973% |
| SMI-TED-Light | 44,125 | 99.959% | 78,678 | 99.973% |
| MolCLR-GIN | 44,143 | 100.000% | 78,699 | 100.000% |
| KERMT v2 | 44,143 | 100.000% | 78,699 | 100.000% |
| **All-model intersection** | **43,730** | **99.064%** | **78,131** | **99.278%** |

MolAI was the limiting representation because of its frozen 109-token and
character-vocabulary constraints. MoLFormer and SMI-TED-Light each rejected 18
long or non-losslessly-tokenized identities. No input was truncated and no
unknown-character fallback was enabled.

After common-support intersection, every endpoint retained zero canonical
identity overlap and zero Bemis--Murcko scaffold overlap between `train_val`
and the fixed test role. The predeclared identity-disjoint sensitivity is
therefore numerically identical to the primary result for all 22 endpoints.

## Category-balanced summary

Lower rank is better. Endpoint wins and top-three counts refer only to the
seven primary representations. The 19-endpoint analysis excludes BBB,
Lipophilicity, and AqSolDB from this summary only; their endpoint results remain
reported below.

| Model | 22-endpoint mean rank | Overall rank | Endpoint wins | Top-three endpoints | 19-endpoint mean rank | Robust rank | Robust wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMI-TED-Light | **2.267** | **1** | 4 | **18** | **2.267** | **1** | 3 |
| gMolAI | 2.767 | 2 | **9** | 15 | 2.650 | 2 | **8** |
| MoLFormer | 3.233 | 3 | 4 | 14 | 3.333 | 3 | 4 |
| KERMT v2 | 3.733 | 4 | 4 | 9 | 3.967 | 4 | 3 |
| MolAI epoch 6 | 5.067 | 5 | 1 | 6 | 4.983 | 5 | 1 |
| MolCLR-GIN | 5.467 | 6.5 | 0 | 3 | 5.367 | 6 | 0 |
| Morgan radius-2 | 5.467 | 6.5 | 0 | 1 | 5.433 | 7 | 0 |

For the full analysis, gMolAI's category mean ranks were 2.500 for Absorption,
2.333 for Distribution, 2.833 for Metabolism, 3.667 for Excretion, and 2.500
for Toxicity. SMI-TED-Light's corresponding values were 2.667, 1.333, 2.833,
2.000, and 2.500. Thus gMolAI led more individual endpoints, while SMI-TED's
greater consistency in Distribution and Excretion produced the stronger
category-balanced result.

## Primary paired endpoint results

MAE is minimized; ROC-AUC, PR-AUC, and Spearman are maximized. Values are exact
test metric mean ± population SD across the five train/validation scaffold
seeds used for regularization selection. The test set is the same across those
five outcomes, so the SD is selection sensitivity rather than an independent
replication error or standard error. A displayed `0.0000` commonly means that
all seeds selected a pipeline with the same final `train_val` refit, not that
the endpoint has zero statistical uncertainty. Bold marks the best mean without
implying significance.

| Endpoint (metric) | gMolAI | Morgan | MolAI | MoLFormer | SMI-TED | MolCLR | KERMT v2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `caco2_wang` (MAE) | **0.3008 ± 0.0119** | 0.3792 ± 0.0014 | 0.3412 ± 0.0137 | 0.3769 ± 0.0000 | 0.3570 ± 0.0152 | 0.4050 ± 0.0114 | 0.3654 ± 0.0141 |
| `hia_hou` (ROC-AUC) | 0.9844 ± 0.0044 | 0.9470 ± 0.0000 | 0.9301 ± 0.0096 | 0.9828 ± 0.0010 | **0.9891 ± 0.0029** | 0.9183 ± 0.0120 | 0.9741 ± 0.0061 |
| `pgp_broccatelli` (ROC-AUC) | **0.9276 ± 0.0000** | 0.8808 ± 0.0000 | 0.9088 ± 0.0000 | 0.9086 ± 0.0000 | 0.9070 ± 0.0000 | 0.8691 ± 0.0002 | 0.8593 ± 0.0041 |
| `bioavailability_ma` (ROC-AUC) | 0.5770 ± 0.0082 | 0.6509 ± 0.0020 | 0.6555 ± 0.0045 | **0.7291 ± 0.0180** | 0.7186 ± 0.0000 | 0.6058 ± 0.0206 | 0.6158 ± 0.0076 |
| `lipophilicity_astrazeneca` (MAE) | 0.6581 ± 0.0050 | 0.7187 ± 0.0000 | 0.8265 ± 0.0000 | 0.6321 ± 0.0000 | 0.6999 ± 0.0086 | 0.7520 ± 0.0004 | **0.6155 ± 0.0079** |
| `solubility_aqsoldb` (MAE) | **0.7947 ± 0.0119** | 1.2597 ± 0.0000 | 0.8774 ± 0.0006 | 0.8552 ± 0.0046 | 0.8349 ± 0.0042 | 1.1851 ± 0.0351 | 0.8896 ± 0.0501 |
| `bbb_martins` (ROC-AUC) | 0.8746 ± 0.0010 | 0.7903 ± 0.0308 | 0.8429 ± 0.0065 | 0.8875 ± 0.0098 | **0.9122 ± 0.0000** | 0.7865 ± 0.0026 | 0.9010 ± 0.0134 |
| `ppbr_az` (MAE) | **9.0211 ± 0.0622** | 11.5023 ± 0.0000 | 10.2828 ± 0.1772 | 9.4089 ± 0.0000 | 9.0636 ± 0.0000 | 10.7909 ± 0.1103 | 9.6031 ± 0.0594 |
| `vdss_lombardo` (Spearman) | 0.4035 ± 0.0000 | 0.2061 ± 0.0000 | 0.1529 ± 0.0227 | 0.2962 ± 0.0957 | **0.4277 ± 0.0000** | 0.3043 ± 0.0170 | 0.3147 ± 0.0000 |
| `cyp2d6_veith` (PR-AUC) | **0.5721 ± 0.0134** | 0.5161 ± 0.0000 | 0.4424 ± 0.0040 | 0.5677 ± 0.0000 | 0.5699 ± 0.0000 | 0.4210 ± 0.0076 | 0.5299 ± 0.0088 |
| `cyp3a4_veith` (PR-AUC) | **0.8248 ± 0.0010** | 0.8210 ± 0.0000 | 0.7494 ± 0.0001 | 0.8129 ± 0.0045 | 0.8210 ± 0.0004 | 0.7405 ± 0.0027 | 0.8182 ± 0.0022 |
| `cyp2c9_veith` (PR-AUC) | **0.7586 ± 0.0008** | 0.6908 ± 0.0000 | 0.7048 ± 0.0050 | 0.7500 ± 0.0006 | 0.7246 ± 0.0016 | 0.6846 ± 0.0032 | 0.7212 ± 0.0058 |
| `cyp2d6_substrate_carbonmangels` (PR-AUC) | 0.7017 ± 0.0039 | 0.6311 ± 0.0277 | 0.5685 ± 0.0613 | 0.5770 ± 0.0317 | 0.6885 ± 0.0000 | 0.7061 ± 0.0169 | **0.7163 ± 0.0000** |
| `cyp3a4_substrate_carbonmangels` (ROC-AUC) | 0.5826 ± 0.0233 | 0.6478 ± 0.0108 | **0.6867 ± 0.0093** | 0.6342 ± 0.0116 | 0.6622 ± 0.0082 | 0.6335 ± 0.0186 | 0.6647 ± 0.0028 |
| `cyp2c9_substrate_carbonmangels` (PR-AUC) | 0.3513 ± 0.0218 | 0.3454 ± 0.0009 | 0.3600 ± 0.0101 | 0.3443 ± 0.0053 | 0.3898 ± 0.0050 | 0.3398 ± 0.0153 | **0.4232 ± 0.0270** |
| `half_life_obach` (Spearman) | 0.1480 ± 0.0957 | 0.1789 ± 0.0680 | 0.2024 ± 0.0922 | **0.3117 ± 0.0441** | 0.2595 ± 0.0525 | 0.2121 ± 0.0477 | 0.1415 ± 0.0273 |
| `clearance_microsome_az` (Spearman) | **0.6031 ± 0.0262** | 0.4758 ± 0.0000 | 0.3339 ± 0.0471 | 0.5439 ± 0.0594 | 0.5679 ± 0.0000 | 0.5388 ± 0.0040 | 0.4997 ± 0.0429 |
| `clearance_hepatocyte_az` (Spearman) | 0.3447 ± 0.0416 | 0.2445 ± 0.0179 | 0.2301 ± 0.0088 | **0.3973 ± 0.0496** | 0.3668 ± 0.0164 | 0.3566 ± 0.0108 | 0.3291 ± 0.0204 |
| `herg` (ROC-AUC) | 0.8065 ± 0.0177 | 0.7960 ± 0.0095 | 0.8221 ± 0.0000 | 0.7791 ± 0.0000 | 0.8140 ± 0.0161 | 0.6972 ± 0.0061 | **0.8666 ± 0.0000** |
| `ames` (ROC-AUC) | 0.8111 ± 0.0066 | 0.7717 ± 0.0000 | 0.7305 ± 0.0066 | 0.7999 ± 0.0000 | **0.8190 ± 0.0000** | 0.7572 ± 0.0035 | 0.8139 ± 0.0065 |
| `dili` (ROC-AUC) | **0.9396 ± 0.0091** | 0.7843 ± 0.0135 | 0.8208 ± 0.0000 | 0.8455 ± 0.0360 | 0.9109 ± 0.0000 | 0.8222 ± 0.0041 | 0.8148 ± 0.0436 |
| `ld50_zhu` (MAE) | 0.6919 ± 0.0043 | 0.7055 ± 0.0000 | 0.7143 ± 0.0012 | **0.6856 ± 0.0000** | 0.6944 ± 0.0000 | 0.7441 ± 0.0025 | 0.6938 ± 0.0050 |

## gMolAI interpretation

gMolAI ranked first on Caco-2, P-gp, AqSolDB, PPBR, all three Veith CYP
inhibition endpoints, microsomal clearance, and DILI. It exceeded the endpoint
mean of Morgan on 19/22 endpoints, MolAI and MolCLR-GIN on 17/22 each,
MoLFormer on 15/22, KERMT v2 on 14/22, and SMI-TED-Light on 12/22. Those are
descriptive paired endpoint counts, not inferential tests.

The clearest limitations are equally important. gMolAI ranked seventh for oral
bioavailability and CYP3A4 substrate classification, and sixth for half-life.
MoLFormer was strongest across the three Excretion endpoints by mean rank;
SMI-TED-Light was strongest for Distribution; KERMT v2 led three substrate or
toxicity endpoints. The result therefore supports complementary,
endpoint-dependent strengths rather than a blanket best-model claim.

The selection-robust analysis strengthens the broad-transfer inference rather
than weakening it. Removing the exact BBBP and Lipophilicity reuse and the
strong ESOL--AqSolDB overlap leaves gMolAI second overall, improves its
category-balanced mean rank from 2.767 to 2.650, and leaves 8 wins among 19
endpoints. All three excluded endpoints remain visible in the primary table.

## Descriptor diagnostic

The 13 hand-designed RDKit features were excluded from primary ranking because
several endpoints are directly physicochemical. Under the same shallow-probe
family, gMolAI had the favorable endpoint mean on 17/22 endpoints relative to
this diagnostic; the descriptor panel was favorable on Vdss, CYP3A4 substrate,
CYP2C9 substrate, half-life, and hERG.

This suggests that gMolAI's linear-probe utility is not reproduced by these 13
raw descriptors alone. It is not a causal ablation: gMolAI was trained with
descriptor supervision, the feature dimensions differ, and the diagnostic was
not designed as a primary competitor.

## Bounded conclusions

- The released 384-D gMolAI representation is a strong broad ADMET feature
  representation under frozen inference and a common simple-probe family.
- SMI-TED-Light is the overall category-balanced leader, while gMolAI has the
  most endpoint wins. Both facts should appear together in the manuscript.
- The public benchmark is retrospective, common-support filtered, and not a
  literal TDC leaderboard submission. Unknown pretraining-corpus overlap for
  external encoders remains a limitation.
- The five values per endpoint reuse one fixed test set and quantify only
  train/validation selection sensitivity. No p-value or independent-test
  replicate claim is supported.
- Native representation dimensions and pretrained objectives differ. The
  comparison evaluates released representations under one probe protocol; it
  is not a parameter-matched architecture ablation.
- This benchmark evaluates ADMET representation utility. It does not evaluate
  decoder-generated analogues, establish candidate property improvement, or
  reopen derivative generation. Property-guided candidate optimization remains
  future work.

## Versioned audit artifacts

- [`protocol.json`](protocol.json) and [`PROTOCOL.md`](PROTOCOL.md): frozen
  machine-readable and narrative protocol;
- [`outputs/tdc_admet_summary.json`](outputs/tdc_admet_summary.json): compact
  machine-readable headline and interpretation boundaries;
- [`outputs/model_summary.csv`](outputs/model_summary.csv),
  [`outputs/category_rank_summary.csv`](outputs/category_rank_summary.csv), and
  [`outputs/endpoint_primary_metrics.csv`](outputs/endpoint_primary_metrics.csv):
  primary rank and endpoint tables;
- [`outputs/all_metrics.csv`](outputs/all_metrics.csv): all secondary metrics
  and five-seed values;
- [`outputs/coverage.csv`](outputs/coverage.csv): endpoint- and model-specific
  support accounting;
- [`state/preflight.json`](state/preflight.json),
  [`state/status.json`](state/status.json), and
  [`state/COMPLETE.json`](state/COMPLETE.json): execution and completion state;
- [`outputs/SHA256SUMS`](outputs/SHA256SUMS): full local artifact ledger.

To verify the completed study without changing artifacts, run from the
repository root:

```bash
PYTHONPATH=extra-benchmark/tdc-admet/scripts \
  python extra-benchmark/tdc-admet/scripts/verify_results.py
```

Large embedding arrays, reconstructed source tables, support-screen payloads,
embedding metadata, and runtime logs remain local and ignored. Their identities
are retained by the tracked manifests, compact result records, and checksum
ledger.
