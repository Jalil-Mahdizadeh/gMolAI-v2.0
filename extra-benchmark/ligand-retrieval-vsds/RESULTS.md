# VSDS-vd TrueDecoy_gap ligand-retrieval results

This report is generated directly from frozen result tables. Values are target-level means across deterministic anchor draws, followed by target-stratified bootstrap 95% confidence intervals. Anchor draws are not treated as independent inferential replicates; no formal p-values were performed.

## Frozen study population

The all-seven intersection retained **81,126** unique molecules before target eligibility and **80,713** unique molecules across **70** eligible protein targets. The primary analysis used 5 active anchors and 20 deterministic draws per target.

| Model | Validated | Rejected | Coverage (%) |
| --- | --- | --- | --- |
| gMolAI | 81622 | 0 | 100.00 |
| Morgan (binary) | 81622 | 0 | 100.00 |
| MolAI | 81129 | 493 | 99.40 |
| MoLFormer | 81619 | 3 | 100.00 |
| SMI-TED-Light | 81622 | 0 | 100.00 |
| MolCLR-GIN | 81621 | 1 | 100.00 |
| KERMT v2 | 81622 | 0 | 100.00 |

## Primary five-shot retrieval

| Model | EF1% | BEDROC (alpha=20) | ROC-AUC | Average precision |
| --- | --- | --- | --- | --- |
| gMolAI | 23.344 [20.573, 26.357] | 0.457 [0.416, 0.497] | 0.798 [0.776, 0.819] | 0.322 [0.277, 0.368] |
| Morgan (binary) | 29.998 [27.330, 32.742] | 0.554 [0.515, 0.594] | 0.827 [0.806, 0.848] | 0.427 [0.382, 0.476] |
| MolAI | 22.010 [19.402, 24.748] | 0.377 [0.340, 0.416] | 0.698 [0.675, 0.721] | 0.275 [0.236, 0.313] |
| MoLFormer | 26.658 [23.641, 29.688] | 0.505 [0.465, 0.547] | 0.825 [0.803, 0.846] | 0.374 [0.330, 0.421] |
| SMI-TED-Light | 24.929 [22.197, 27.807] | 0.465 [0.427, 0.506] | 0.790 [0.771, 0.811] | 0.339 [0.297, 0.381] |
| MolCLR-GIN | 22.454 [19.681, 25.393] | 0.444 [0.405, 0.483] | 0.802 [0.783, 0.820] | 0.307 [0.266, 0.351] |
| KERMT v2 | 29.846 [27.151, 32.628] | 0.560 [0.521, 0.600] | 0.845 [0.825, 0.865] | 0.429 [0.383, 0.477] |

## Paired primary EF1%: gMolAI minus comparator

| Comparator | Mean difference [95% CI] | Wins/losses/ties |
| --- | --- | --- |
| Morgan (binary) | -6.654 [-8.004, -5.420] | 7/62/1 |
| MolAI | 1.334 [-0.132, 2.797] | 44/25/1 |
| MoLFormer | -3.314 [-4.279, -2.382] | 13/56/1 |
| SMI-TED-Light | -1.585 [-2.849, -0.375] | 27/41/2 |
| MolCLR-GIN | 0.890 [-0.022, 1.843] | 41/27/2 |
| KERMT v2 | -6.502 [-7.671, -5.349] | 5/63/2 |

## Prespecified sensitivity analyses

| Model | One-shot EF1% [95% CI] | Scaffold-excluded EF1% [95% CI] | Scaffold targets |
| --- | --- | --- | --- |
| gMolAI | 13.910 [12.060, 15.954] | 23.723 [20.794, 26.657] | 70 |
| Morgan (binary) | 17.149 [15.169, 19.261] | 30.567 [27.837, 33.404] | 70 |
| MolAI | 10.736 [9.074, 12.452] | 22.140 [19.468, 24.919] | 70 |
| MoLFormer | 15.896 [13.931, 18.016] | 26.990 [24.156, 30.052] | 70 |
| SMI-TED-Light | 14.389 [12.486, 16.471] | 25.242 [22.538, 28.064] | 70 |
| MolCLR-GIN | 13.051 [11.331, 14.919] | 22.599 [19.847, 25.498] | 70 |
| KERMT v2 | 17.885 [15.912, 20.010] | 30.305 [27.555, 33.134] | 70 |

## gMolAI pretraining exposure (descriptive only)

| Label | Memberships | Corpus overlap | Overlap (%) | Seen by step 10,000 | Seen (%) |
| --- | --- | --- | --- | --- | --- |
| active | 2333 | 586 | 25.12 | 161 | 6.90 |
| inactive or lower affinity | 94912 | 38360 | 40.42 | 9955 | 10.49 |

Exact molecule-level exposure was auditable only for gMolAI. These counts are descriptive and do not support an unseen-molecule or out-of-distribution performance claim for any model.

## Artifacts

The manuscript figure is `figures/main_lbvs_figure.{pdf,svg,png}`; the compact supplementary figure is `figures/si_lbvs_secondary_metrics.{pdf,svg,png}`; and the seven-model target-balanced ROC plot is `figures/five_shot_macro_roc_curves.{pdf,svg,png}`. Every plotted value is retained in `figures/source-data/`.
