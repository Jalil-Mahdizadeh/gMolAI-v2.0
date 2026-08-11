---
type: "manuscript_audit"
date: "2026-08-11T09:38:27.272171+00:00"
question: "Are the two manuscript documents accurate relative to this repository?"
contributor: "graphify"
outcome: "useful"
---

# Q: Are the two manuscript documents accurate relative to this repository?

## Answer

Largely accurate and strongly supported by code, configuration, promotion records, training logs, downstream artifacts, seed-43 replication, HIV, and locked-test records. Required corrections: cosine-Tanimoto Spearman used 200000 randomly sampled non-self pairs, not all pairs or a global statistic; Table 1 used the uncalibrated raw graph_z-plus-mean_node_z vector and omitted the retained 2500-step milestone; seed-42 ESOL 0.731371 rounds to 0.7314, not 0.7315; seed-43 used a separately fitted calibrator on the same deterministic 100000-molecule sample, not an independently sampled set; the two edge outputs use separate decoder MLPs; replace without numerical degradation with without numerical instability or latent collapse. Clarify that per-fold downstream StandardScaler makes coordinate calibration and constant block weighting irrelevant to linear/logistic probe scores, while they affect retrieval and clustering. Minor wording: molecule-hash buckets, population SD, MolWt unit consistency, and sampled Spearman protocol.

## Outcome

- Signal: useful