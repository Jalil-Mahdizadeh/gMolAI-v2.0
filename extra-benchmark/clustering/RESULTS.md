# Frozen minimal clustering benchmark — results

## Outcome

The frozen benchmark completed on 36,200 balanced ClassyFire-25 molecules (1,448 per subclass) and 50,000 QMugs molecules from 60,000 attempted identities. All seven primary representations use identical identities within each endpoint, and every final vector was finite and nonzero. Count Morgan and the frozen 13-descriptor vector are diagnostics only.

## ClassyFire-25 structural organization

Values are estimates with molecule-stratified paired 95% bootstrap intervals. K-means estimates average the five prespecified algorithmic seeds; the seeds are not inferential replicates.

| Representation | ARI | AMI | NMI | Macro same-subclass@100 |
|---|---:|---:|---:|---:|
| gMolAI | 0.260 [0.257, 0.263] | 0.427 [0.426, 0.433] | 0.429 [0.427, 0.434] | 0.501 [0.499, 0.504] |
| Morgan (binary) | 0.234 [0.231, 0.237] | 0.453 [0.452, 0.458] | 0.454 [0.453, 0.459] | 0.560 [0.557, 0.562] |
| MolAI | 0.116 [0.115, 0.119] | 0.283 [0.282, 0.287] | 0.284 [0.284, 0.289] | 0.358 [0.356, 0.360] |
| MoLFormer | 0.248 [0.245, 0.252] | 0.431 [0.430, 0.436] | 0.433 [0.431, 0.438] | 0.517 [0.514, 0.519] |
| SMI-TED-Light | 0.209 [0.207, 0.213] | 0.410 [0.408, 0.414] | 0.411 [0.410, 0.416] | 0.497 [0.495, 0.500] |
| MolCLR-GIN | 0.156 [0.154, 0.159] | 0.330 [0.329, 0.335] | 0.332 [0.331, 0.337] | 0.386 [0.384, 0.388] |
| KERMT v2 | 0.267 [0.264, 0.271] | 0.440 [0.439, 0.445] | 0.441 [0.440, 0.446] | 0.533 [0.530, 0.536] |

ARI: gMolAI ranked 2/7 (0.260), with KERMT v2 first (0.267); AMI: gMolAI ranked 4/7 (0.427), with Morgan (binary) first (0.453); same-subclass@100: gMolAI ranked 4/7 (0.501), with Morgan (binary) first (0.560). In paired contrasts, gMolAI minus Morgan was 0.026 [0.022, 0.030] for ARI but -0.026 [-0.029, -0.022] for AMI; gMolAI minus KERMT v2 was -0.008 [-0.010, -0.005] for ARI. Across the five K-means seeds, gMolAI ARI ranged from 0.257 to 0.262 and AMI from 0.425 to 0.429. Thus, no representation dominates all structural endpoints. The taxonomy is an external, reproducible ClassyFire reference, but it is hierarchical ontology assignment rather than an error-free physical ground truth; ARI/AMI quantify agreement, not chemical correctness.

Main figures: [structural metrics](outputs/figures/figure_classyfire_main_metrics.pdf) and [visualization-only PCA](outputs/figures/figure_classyfire_pca.pdf).

## QMugs independent-property organization

NPD@100 is lower-is-better; Recall@100 is higher-is-better. QM properties are DFT HOMO energy, HOMO–LUMO gap, and log(1 + total dipole), robust-scaled on the frozen common panel. The frozen constants (in property-table units) were: DFT_HOMO_ENERGY: median -0.288100, IQR 0.024812; DFT_HOMO_LUMO_GAP: median 0.300431, IQR 0.037806; log1p_DFT_DIPOLE_TOT: median 1.652357, IQR 0.561402. No property bins or property-derived clusters were used.

| Representation | NPD@100 | Property-neighbor Recall@100 |
|---|---:|---:|
| gMolAI | 0.810 [0.809, 0.811] | 0.0113 [0.0111, 0.0115] |
| Morgan (binary) | 0.834 [0.833, 0.836] | 0.0108 [0.0106, 0.0110] |
| MolAI | 0.894 [0.893, 0.894] | 0.0067 [0.0066, 0.0068] |
| MoLFormer | 0.819 [0.817, 0.820] | 0.0104 [0.0103, 0.0106] |
| SMI-TED-Light | 0.811 [0.810, 0.812] | 0.0102 [0.0100, 0.0103] |
| MolCLR-GIN | 0.841 [0.840, 0.842] | 0.0085 [0.0084, 0.0087] |
| KERMT v2 | 0.832 [0.830, 0.833] | 0.0108 [0.0106, 0.0110] |

NPD@100: gMolAI ranked first (0.810); Recall@100: gMolAI ranked first (0.011). Relative to the closest NPD competitor, SMI-TED-Light, the paired gMolAI difference was -0.0011 [-0.0019, -0.0002]; relative to binary Morgan, the Recall@100 difference was 0.00048 [0.00036, 0.00060]. Random-neighbor expected recall is approximately 0.0020, so absolute recall remains modest even though every primary model exceeds chance. The 13-descriptor diagnostic was substantially weaker than gMolAI on both endpoints, arguing against the selected electronic-property result being explained by those descriptors alone. Heavy-atom-count-decile results and per-property deviations are reported in SI source tables and figures. This complement tests local organization by three independent electronic properties; it does not imply that this three-property geometry exhausts molecular-property similarity.

Main figures: [property metrics](outputs/figures/figure_qmugs_main_metrics.pdf), [per-property deviations](outputs/figures/figure_qmugs_property_deviations.pdf), and [visualization-only PCA](outputs/figures/figure_qmugs_pca_homo_lumo_gap.pdf).

## Coverage and exposure audit

ClassyFire began with 75,000 rows; 80 unsupported-element rows were excluded, leaving 74,920. The all-model intersection contained 68,847; MolAI rejected 6,073 identities, and the limiting subclass (steroidal glycosides) retained 1,448, fixing the balanced panel at 36,200. This strict intersection is fair across representations but materially conditions inference on MolAI-encodable chemistry and discards otherwise usable molecules. QMugs yielded 632,079 eligible unique identities; 59,176 of the first 60,000 were common, so the prespecified 50,000-molecule target was met without expansion.

ClassyFire: 25,711/36,200 exact corpus overlaps (71.02%), of which 6,652 had been presented before the frozen checkpoint. QMugs: 17,017/50,000 exact corpus overlaps (34.03%), of which 4,412 had been presented before the frozen checkpoint. Exact identity exposure is unknown for competitors whose released checkpoints do not provide molecule-level training manifests; consequently, no unseen-molecule or out-of-distribution claim is made. The explicit-H amendment was applied symmetrically before canonicalization, and the QMugs heavy-atom counts agreed exactly after normalization.

## Manuscript assessment

The benchmark is methodologically strong enough for manuscript inclusion as a compact external representation analysis: it uses independent labels/properties, identical molecular support, one common operator, exact neighborhoods, and query-level paired uncertainty. It should remain a supporting representation result, not be framed as a universal clustering benchmark.

No additional dataset or clustering algorithm is needed before integration. The manuscript must preserve the distinction between the seven-model primary ranking and both diagnostics, foreground the ClassyFire support contraction, report K-means seed ranges alongside query-bootstrap intervals, state the asymmetric pretraining-exposure knowledge, and use PCA only as an illustrative panel. Do not aggregate structural and property endpoints into one score or claim universal clustering superiority.

Full definitions and frozen choices are in [PROTOCOL.md](PROTOCOL.md); exact tables, per-query source data, neighbor/cluster artifacts, figure source data, and verification records are retained in this directory.
