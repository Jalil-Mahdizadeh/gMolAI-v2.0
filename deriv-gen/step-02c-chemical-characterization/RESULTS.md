# Step 2c results: chemical characterization of frozen candidate sets

## Bounded conclusion

**NOT SUPPORTED: `useful_local_analogue_generator_not_supported`.** This conclusion applies only to the frozen,
unperturbed Step-2b correct-condition sets. It does not demonstrate controllable
latent edits, MMP-direction generation, novelty, synthesizability, or activity
improvement. No model was executed or trained in Step 2c.

The frozen rule gates were: `genuine_nonseed_yield`=true, `mmp_seed_coverage`=true, `scaffold_locality`=false, `seed_similarity_local_not_trivial`=true, `unique_nonseed_yield`=true, `within_set_nontrivial_diversity`=true.

## Denominators, validity, and uniqueness

All 10,000 fresh-validation seeds were included. One seed can
have fewer than 50 retained candidates because Step 2b filtered invalid and
policy-rejected beam hypotheses and deduplicated canonical molecular identities.

- Persisted retained candidates: 354,262;
  independently reparsed valid-SMILES rate
  100.00% and unchanged-policy
  acceptance 100.00%.
- Filled, valid, unique-identity slots: 70.85%
  of the nominal 500,000 seed×50 slots; mean
  35.43 and median
  36 identities per seed.
- Raw 64-beam hypotheses: valid-SMILES rate
  64.78%, policy-accepted rate
  64.77%; greedy validity was
  97.01%.
- Accepted-beam molecular-identity redundancy was
  59,432
  (14.34%). Individual
  discarded raw strings were not stored, so this aggregate cannot separate
  verbatim duplicate strings from alternative spellings of the same identity.
- A seed identity appeared in 9,392
  sets: 8,960 used the exact
  canonical seed spelling and
  432 used an alternate
  raw SMILES spelling. Neither is counted as a derivative.

## Primary chemical classification

| Category | Candidates | Seeds | Retained fraction |
|---|---:|---:|---:|
| Exact seed spelling | 8,960 | 8,960 | 2.53% |
| Seed identity, alternate SMILES | 432 | 432 | 0.12% |
| One-cut MMP | 43,353 | 7,892 | 12.24% |
| Scaffold-preserving non-MMP | 43,425 | 6,510 | 12.26% |
| Acyclic non-MMP | 7,555 | 191 | 2.13% |
| Scaffold-changing | 250,537 | 9,858 | 70.72% |

Genuine derivatives/analogues are the 344,870
non-seed rows; the seed and an alternate SMILES for the seed are excluded.

## One-cut MMP derivatives

The exact hash-bound Step-1b fragmentation code identified
43,353 unique non-seed candidates as true one-cut
MMPs (12.57% of genuine
non-seed candidates). At least one MMP was present for
7,892 seeds
(78.92%).

| Minimum MMP derivatives among 50 | Seeds | Fraction of all seeds |
|---:|---:|---:|
| 1 | 7,892 | 78.92% |
| 5 | 3,941 | 39.41% |
| 10 | 1,257 | 12.57% |
| 20 | 60 | 0.60% |
| 30 | 4 | 0.04% |
| 40 | 2 | 0.02% |

All valid MMP explanations are in `mmp_explanations.parquet`.
18,109 MMP candidates had more
than one valid core explanation. The recurrent table below uses only the frozen
deterministic primary explanation, so its labels are summaries rather than
claims of a uniquely determined chemical edit.

| Primary seed→candidate substituent transform | Class | Candidates | Seeds | Median Morgan |
|---|---|---:|---:|---:|
| `C[*:1]>>CC[*:1]` | substituent_growth | 581 | 510 | 0.811 |
| `CC[*:1]>>C[*:1]` | substituent_truncation | 566 | 520 | 0.818 |
| `Cl[*:1]>>Br[*:1]` | equal_heavy_atom_replacement | 305 | 285 | 0.786 |
| `Br[*:1]>>I[*:1]` | equal_heavy_atom_replacement | 189 | 189 | 0.756 |
| `Cl[*:1]>>F[*:1]` | equal_heavy_atom_replacement | 187 | 171 | 0.800 |
| `C[*:1]>>Cl[*:1]` | equal_heavy_atom_replacement | 159 | 149 | 0.818 |
| `C[*:1]>>F[*:1]` | equal_heavy_atom_replacement | 153 | 150 | 0.824 |
| `CC(C)[*:1]>>CC[*:1]` | substituent_truncation | 143 | 142 | 0.746 |
| `CO[*:1]>>O[*:1]` | substituent_truncation | 108 | 108 | 0.778 |
| `C[*:1]>>CO[*:1]` | substituent_growth | 108 | 107 | 0.788 |
| `c1ccc([*:1])cc1>>C1=CCC=CC([*:1])=C1` | substituent_growth | 102 | 102 | 0.639 |
| `Br[*:1]>>Cl[*:1]` | equal_heavy_atom_replacement | 90 | 89 | 0.756 |

## Scaffold preservation and chemical proximity

Among genuine non-seed candidates whose seed had a non-empty Bemis-Murcko
scaffold, 17.88% retained it.
The exact scaffold-key rate including empty keys was
19.83%; both seed
and candidate were acyclic in
2.40% of
non-seed rows. The median seed produced
19 scaffold keys
(19 non-empty).

For all genuine non-seed candidates, seed-candidate Morgan Tanimoto had mean
0.535, median 0.527, IQR
0.423--0.643, and q10--q90
0.339--0.736.

| Population | n | Mean | Median | IQR | q10--q90 |
|---|---:|---:|---:|---:|---:|
| all_genuine_nonseed | 344,870 | 0.535 | 0.527 | 0.423--0.643 | 0.339--0.736 |
| One-cut MMP | 43,353 | 0.708 | 0.705 | 0.647--0.766 | 0.589--0.830 |
| Scaffold-preserving non-MMP | 43,425 | 0.561 | 0.565 | 0.467--0.655 | 0.382--0.725 |
| Acyclic non-MMP | 7,555 | 0.425 | 0.404 | 0.320--0.514 | 0.250--0.629 |
| Scaffold-changing | 250,537 | 0.503 | 0.493 | 0.403--0.594 | 0.327--0.690 |

## Within-set diversity and non-MMP graph changes

Across 6,431,473 within-set non-seed pairs,
Morgan Tanimoto had mean 0.451, median
0.437, and IQR
0.353--0.534.
Only 0.35% of all
distinct-identity within-set pairs had Tanimoto ≥0.90. Tanimoto 1.0 can still
occur for different canonical identities because this Morgan fingerprint does
not encode every stereochemical distinction.

For non-MMP candidates, `non_mmp_graph_delta_summary.csv` and
`non_mmp_descriptor_delta_patterns.csv` report signed heavy-atom, bond, ring,
heteroatom, formal-charge, aromatic, and elemental-count changes. These are
descriptor deltas, not a claimed unique atom mapping or graph-edit path.

## Artifact map

- Candidate-level table: `outputs/tables/candidate_characterization.parquet`
- Seed-level tables: `outputs/tables/seed_characterization.parquet` and `.csv`
- MMP explanations/transforms: `outputs/tables/mmp_explanations.parquet` and
  `mmp_transformation_counts.csv`
- Similarity/diversity: `outputs/tables/seed_candidate_similarity_summary.csv`,
  `within_set_pairwise_morgan_histogram.csv`, and
  `outputs/raw/within_set_pairwise_morgan.npz`
- Figures: `outputs/figures/`

No Step 3 or latent perturbation was performed.
