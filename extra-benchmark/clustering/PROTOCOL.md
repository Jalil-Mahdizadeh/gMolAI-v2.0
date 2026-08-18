# Frozen minimal clustering benchmark protocol

This benchmark was frozen on 2026-08-18, before any representation was exported. The authoritative machine-readable specification is `protocol.json`. It records source URLs, exact SHA-256 checksums, model containers, checkpoints, representation definitions, support rules, metrics, random seeds, and plotting-only PCA rules.

## Scientific scope

The benchmark contains exactly two external tests: balanced ClassyFire-25 structural taxonomy and QMugs quantum-property neighborhoods. It does not add ClassyFire-120, another dataset family, another clustering operator, unknown-K tuning, or property-derived classes. All encoders are frozen and no benchmark result is used for training, projection learning, parameter selection, or support selection.

## Identity policy

Both sources are parsed with RDKit. Source-explicit hydrogen atoms are removed before the existing repository policy is applied; this is necessary because the official QMugs summary stores hydrogen-complete SMILES and is an idempotent operation for the ClassyFire strings. Canonical isomeric SMILES must be a single connected molecule with 2–256 atoms and only the elements frozen in `configs/retrain.yaml`. Identity is SHA-256(canonical isomeric SMILES). Identities are deduplicated before model execution; conflicting labels or cross-ChEMBL identities are excluded.

For QMugs, rows flagged `significant_negative_wavenumbers=True`, `nonunique_smiles=True`, or nonfinite in a required property are excluded. The lowest-`DFT_TOTAL_ENERGY` eligible conformer is retained, with `conf_id` as the tie-break.

## Common support

All six non-gMolAI primary adapters are screened without a forward pass. gMolAI accepts the repository-canonical molecules by construction. ClassyFire support is the all-seven-model intersection, followed by model-output-blind balancing: for every subclass retain the SHA-256-first `m` identities, where `m` is the minimum subclass count in the intersection.

Protocol amendment A001 removed conservative 90% overall / 80% per-subclass abort guards that were added during implementation but were not part of the requested design. This correction occurred after label-blind adapter screening and before any representation forward pass; no embeddings or endpoint results existed. The requested intersection-and-`m` rule is unchanged, and the resulting support attrition is reported in full.

QMugs identities are SHA-256 ordered. The first 60,000 are screened; only if fewer than 50,000 are common is the prefix expanded in fixed blocks of 10,000, up to 100,000. The final panel is the first 50,000 common identities, or every survivor if the target cannot be met.

## Structural endpoint

Each fixed-length vector is cast to float64, rejected if nonfinite or zero norm, and row-L2 normalized. Euclidean K-means uses K=25, k-means++, Lloyd, `n_init=20`, `max_iter=500`, `tol=1e-4`, and seeds 42–46. There is no centering, feature scaling, PCA, whitening, or learned projection. ARI and AMI are primary; NMI is retained for continuity. The geometry companion is macro same-subclass@100. Binary Morgan Tanimoto and count-Morgan generalized Tanimoto neighborhoods are explicitly labeled sensitivities, not substitutions for the common operator.

## Property endpoint

The three-dimensional reference is `DFT_HOMO_ENERGY`, `DFT_HOMO_LUMO_GAP`, and `log1p(DFT_DIPOLE_TOT)`, each scaled by the final-panel median and IQR. Representation neighbors are exact Euclidean neighbors after float64 row-L2 normalization. NPD@100 is each query's mean neighbor distance in robust property space divided by its mean distance to every other molecule in the same deterministic heavy-atom-count decile; the reported endpoint is the mean query ratio (lower is better). Property-neighbor Recall@100 is mean overlap with exact property-space neighbors (higher is better). Per-property robust-scaled deviations and heavy-atom-decile estimates are supporting outputs. No property clusters or bins are created.

## Statistics and visualization

Paired 95% bootstrap intervals use 2,000 resamples and the molecule/query as the unit, stratified by subclass or heavy-atom decile. K-means seeds are algorithmic sensitivity, never inferential replicates. Structural and property endpoints remain separate. PCA is deterministic visualization only and is never used by a benchmark metric.

The Morgan-count and frozen 13-RDKit-descriptor representations are SI diagnostics and are excluded from the seven-model competitive ranking.
