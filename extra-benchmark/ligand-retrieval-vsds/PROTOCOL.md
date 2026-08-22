# Frozen LBVS protocol

This directory implements one bounded ligand-based virtual-screening
experiment for the current gMolAI manuscript. The machine-readable authority is
`protocol.json`; this document is its concise human-readable interpretation.

## Population

The only source benchmark is VSDS-vd v3 (`10.5281/zenodo.14874127`). The
archive contains 147 TrueDecoy targets. The paper's official Supplementary Data
1 workbook defines the 71-target `TrueDecoy_gap` subset and supplies its target
classes and source counts. Both binaries are checksum-pinned.

Source SDF molecules are stripped of explicit hydrogens and passed through the
repository's frozen isomeric canonicalization policy. Disconnected structures,
unsupported elements, and structures outside the 2--256 atom contract are
excluded. Within a target, exact or full-InChIKey duplicates are collapsed;
active/inactive identity contradictions are excluded rather than adjudicated.
The negative label is described as *experimentally characterized inactive or
lower-affinity*, not as proof of absolute non-binding.

All adapters are screened without labels. Each model is encoded on its own
accepted panel, and the export must pass identity, shape, dtype, finiteness, and
nonzero-norm checks. Only then is the all-seven-model intersection constructed.
A primary target requires at least 10 common-support actives and one
inactive/lower-affinity molecule. Exact population and identity digests are
frozen before anchors or retrieval scores are generated.

## Representations and retrieval

The seven frozen representations are gMolAI released-hybrid-w3, binary Morgan
radius 2/2048, MolAI epoch 6, MoLFormer pooler output, SMI-TED-Light official
encoding, MolCLR-GIN preprojection graph vector, and KERMT v2 projected mean
latent. Learned vectors are row-L2-normalized and compared with cosine; Morgan
uses Tanimoto. No representation transformation, training, fine-tuning, or
selection is permitted.

For every eligible target, 20 deterministic hash-seeded draws select five
active anchors. Anchors are removed and every candidate receives its maximum
similarity to any anchor. The same anchors and candidates are used for every
model. A secondary one-anchor analysis uses the same 20-draw framework.

The scaffold robustness analysis reuses each five-anchor draw. Every candidate
of either label sharing a nonempty Bemis--Murcko scaffold with an anchor is
removed. Empty acyclic scaffolds are not exclusion keys. A draw requires at
least five remaining cross-scaffold actives and one remaining
inactive/lower-affinity candidate; a target requires at least 10 eligible draws
for scaffold-level summaries.

## Metrics and inference

The primary endpoint is five-shot EF1%. For candidate count `N`,
`k=max(1,ceil(0.01*N))` and
`EF=(A_k/A)/(k/N)`. A boundary tie receives fractional active credit. Secondary
metrics are tie-aware BEDROC at alpha 20, ROC-AUC, and scikit-learn average
precision. A deterministic random ranking is a diagnostic control, not an
eighth model.

Anchor draws quantify sampling sensitivity; they are not independent
replicates. Metrics are averaged within target before target-class-stratified,
target-level bootstrap intervals and paired gMolAI comparisons are calculated.
No formal p-values are used.

## Scope boundary

This benchmark does not test arbitrary protein--ligand binding prediction. It
does not use proteins, docking, trained classifiers, target-specific models,
additional datasets, hyperparameter sweeps, or checkpoint selection. Exact
gMolAI pretraining exposure is descriptive and is not used to remove molecules
from the primary analysis.

