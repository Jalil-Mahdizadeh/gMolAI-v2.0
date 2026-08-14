# Frozen scientific protocol

## Objective

Evaluate the frozen gMolAI representation on the entire 22-endpoint TDC ADMET
group against exactly the frozen comparator representations qualified in
`extra-benchmark/test-partition/`. This provides a broad property-prediction
study while keeping derivative generation closed at Step 2d.

## Data and roles

- Source: *TDC ADMET benchmark snapshot (2026-03-24)*, DOI
  `10.5281/zenodo.20180944`, downloaded originally with PyTDC 0.3.8.
- The archive SHA-256 and every one of the 44 CSV hashes are pinned in
  `protocol.json`.
- All 22 endpoints are included. No endpoint is selected using results.
- Each published `test.csv` remains test-only.
- For seeds 1, 2, 3, 4, and 5, the published `train_val.csv` is divided with an
  exact local reproduction of PyTDC 0.4.1 `create_scaffold_split` at fractions
  0.875/0.125/0.0, Murcko scaffolds, and `includeChirality=False`.
- Official source occurrences and labels are retained. Canonical identities are
  encoded once and remapped to occurrences; duplicates are not relabelled,
  averaged, or silently removed.

## Molecular policy and common panel

The repository-pinned isomeric canonicalization, element set, fragment policy,
and 2--256 atom limits are applied. Policy-invalid rows are counted by endpoint
and role. Each comparator then performs its already-qualified support screen.
The primary paired evaluation is the intersection accepted by all seven
representations. Split membership is intersected with this panel without
reassignment.

Exact canonical-identity and Bemis--Murcko scaffold overlap between train/valid
and test is audited. The primary result preserves the official roles. A
predeclared secondary identity-disjoint test sensitivity removes test
occurrences whose canonical identity is present in `train_val`; it is reported
only when its metric is defined and is never used for selection.

## Representations

All encoders, checkpoints, calibrators, preprocessing contracts, container
images, and batch sizes are inherited unchanged from the completed comparator
qualification and MoleculeNet study:

1. gMolAI released hybrid, 384 dimensions;
2. Morgan radius-2, 2,048 bits;
3. MolAI epoch 6, 512 dimensions;
4. MoLFormer, 768 dimensions;
5. SMI-TED-Light, 768 dimensions;
6. MolCLR-GIN, 512 dimensions;
7. KERMT v2, 512 dimensions.

The diagnostic control contains exactly 13 RDKit properties: QED, MolWt,
NumValenceElectrons, MaxPartialCharge, MinPartialCharge, BalabanJ, LabuteASA,
TPSA, HeavyAtomCount, NumHAcceptors, NumHDonors, MolLogP, and MolMR. It is not a
primary comparator because several TDC endpoints are closely related to these
hand-designed physicochemical variables.

## Probe fitting

For every endpoint, representation, and scaffold seed:

- fit `StandardScaler` on the seed's training fold only;
- regression: standardize the training target, select Ridge `alpha` from
  `[0.1, 1, 10, 100, 1000]` using the endpoint's official validation metric;
- classification: select balanced liblinear logistic-regression `C` from
  `[0.01, 0.1, 1, 10]` using the endpoint's official validation metric;
- ties resolve to the first value in the frozen ascending grid;
- refit the chosen pipeline on all common `train_val` occurrences, then evaluate
  the unchanged common test set once.

MAE is minimized; Spearman, ROC-AUC, and PR-AUC are maximized. Undefined
validation or primary test metrics are a protocol failure, not an invitation to
inspect the test set or choose a fallback.

## Outcomes

The primary endpoint outcome is the exact official TDC metric, with the five
selection-seed values, mean, and population standard deviation. Three-decimal
TDC-style values are supplied only as a compatibility view. Secondary metrics
are MAE, RMSE, R2, and Spearman for regression and ROC-AUC, average precision,
and balanced accuracy for classification.

The high-level primary summary ranks only the seven frozen representations per
endpoint, averages ranks within Absorption, Distribution, Metabolism,
Excretion, and Toxicity, and then weights those five scientific categories
equally. Endpoint median rank and gMolAI win/tie/loss counts are also reported.
The descriptor diagnostic remains separate. No p-values, universal-superiority
claim, or direct public-leaderboard rank is planned.

A source-only audit completed before representation execution found that
`bbb_martins` and `lipophilicity_astrazeneca` exactly reuse the earlier BBBP and
Lipophilicity development identities and targets. `solubility_aqsoldb` contains
1,098 ESOL identities with strongly concordant targets (Pearson 0.9925). All 22
endpoints remain in the complete panel and tables, while a predeclared
19-endpoint category-balanced sensitivity excludes those three
selection-conditioned endpoints. This sensitivity cannot be changed after
model outputs are observed.

## Leakage and inference constraints

- No TDC labels enter any encoder or representation adaptation.
- No test label enters scaling, regularization selection, or refitting.
- The same occurrence rows and split roles are paired across all models.
- Public benchmark test sets are not prospective validation.
- Pretraining-corpus overlap is not fully knowable for every external model;
  exact endpoint split overlap is reported, and claims are limited accordingly.
- The experiment tests ADMET representation utility only. It does not validate
  property-guided decoder outputs.
