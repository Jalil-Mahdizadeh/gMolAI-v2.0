# Day-1 results: latent geometry and derivative retrieval

## Outcome

The final amended train/validation study is complete. It evaluated latent
organization, unseen-core matched-pair direction transfer, blind perturbations,
oracle interpolation, and retrieval from 50,000 known valid validation molecules.
The pre-outcome support amendment is recorded in PROTOCOL_AMENDMENT_01.md.

**Recommended next approach:** transformation-conditioned matched-pair retrieval with local manifold constraints, followed by a seed-conditioned edit decoder; do not assume a universal linear vector.

This is evidence about retrieval geometry. It is not evidence that the existing
gMolAI checkpoint can decode a 384D vector or generate a novel molecule.

## Audited data

- train geometry/directions: 100,000 molecules;
- independent validation bank: 50,000 molecules;
- train/validation identity overlap: 0;
- held-out MMP retrieval queries: 227;
- locked test molecules used: **0**;
- train eligible fragmentations: 622,020;
- validation eligible fragmentations: 288,622;
- train MMP pairs: 14,343;
- validation MMP pairs: 40,355.

## Geometry

| Space | Global effective rank | Components for 90% variance | Median local effective rank |
|---|---:|---:|---:|
| graph_256 | 40.97 | 40 | 23.21 |
| mean_node_128 | 23.54 | 19 | 17.69 |
| hybrid_384 | 42.62 | 39 | 22.87 |

For hybrid-384, the nearest-neighbor mean Morgan Tanimoto was
**0.4034**, versus
**0.1164** for random pairs. Same-scaffold
fractions were **0.4922** and
**0.0000**, respectively. These quantities
measure chemical enrichment, not guaranteed small edits.

## Unseen-core MMP displacement transfer

| Space | Validation pairs | Transformations | Mean alignment | Median alignment | Null mean | Null median |
|---|---:|---:|---:|---:|---:|---:|
| graph_256 | 685 | 21 | 0.6548 | 0.6637 | 0.0383 | 0.0177 |
| mean_node_128 | 685 | 21 | 0.8251 | 0.8324 | 0.0494 | 0.0265 |
| hybrid_384 | 685 | 21 | 0.6959 | 0.6992 | 0.0592 | 0.0280 |

Positive alignment above the mismatched-transformation null indicates some
cross-core directional organization. The exact retrieval test below determines
whether that signal is strong enough to be practically controlling.

## Hybrid-384 retrieval

Blind perturbations and the learned MMP direction use equal,
transformation-specific median train-pair step lengths. `seed_nn` has zero
perturbation. Oracle interpolation is reported separately in the machine-readable
tables and figures.

| Method | Exact target recall@10 | Top-1 seed Tanimoto | Top-1 same scaffold | Top-1 one-cut MMP |
|---|---:|---:|---:|---:|
| seed_nn | 0.7797 | 0.6477 | 0.9692 | 0.6035 |
| isotropic | 0.7768 | 0.6455 | 0.9633 | 0.6065 |
| global_covariance | 0.7474 | 0.6191 | 0.9471 | 0.5580 |
| local_covariance | 0.6916 | 0.6001 | 0.9148 | 0.5213 |
| mmp_direction | 0.9956 | 0.7402 | 1.0000 | 0.9515 |

## Predeclared gates

- **G1_local_chemical_organization: PASS** — tanimoto_enrichment=3.4645, scaffold_absolute_gain=0.4922
- **G2_transferable_mmp_direction: PASS** — validation_median_alignment=0.6992, null_median_alignment=0.0280, alignment_gain=0.6712
- **G3_controlled_retrieval: FAIL** — mmp_recall_at_10=0.9956, best_blind_recall_at_10=0.7768, absolute_gain=0.2188, fold_gain=1.2817
- **G4_interpolation_sanity: PASS** — nondecreasing=1.0000

Gate failures are retained. A gate was not changed after result inspection.
G3 is conjunctive: the learned direction gained
**0.2188** recall@10 absolute,
but its **1.2817×** improvement did
not meet the predeclared 2× clause. Because the blind baseline was already
**0.7768**, the
maximum possible fold at perfect recall was only
**1.2873×**.
The formal failure is therefore retained without discarding the strong absolute
retrieval signal.

## Bounded conclusion

The appropriate disclosure is that the promoted representation was tested for
seed-centered derivative retrieval under held-out chemical controls. The
recommended next architecture follows the result stated above. A later decoder
must still demonstrate clean reconstruction, condition use, molecular validity,
novelty, and re-encoding consistency before any generative claim.

## Artifacts

- `outputs/tables/`: global/local geometry, distance chemistry, MMP alignment,
  and retrieval summaries;
- `outputs/raw/`: complete per-pair and per-query records;
- `outputs/examples/`: ranked molecular examples including failures;
- `outputs/figures/`: covariance, chemistry-distance, alignment, retrieval, and
  interpolation plots;
- `outputs/study_summary.json`: gates and machine-readable conclusion;
- `outputs/SHA256SUMS`: output integrity ledger;
- `state/COMPLETE.json`: completion and provenance seal.
