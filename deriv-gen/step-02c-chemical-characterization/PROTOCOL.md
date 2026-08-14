# Frozen protocol: Step 2c chemical characterization

## Question and boundary

This is a no-training, no-generation audit of the already frozen Step-2b
`beam64_lp00` correct-condition candidate sets. It asks whether candidates around
an unperturbed released 384-D gMolAI condition form a useful local chemical
neighborhood. It does not retrain or execute gMolAI or the decoder, regenerate
candidates, perturb a latent vector, use endpoint labels, read the locked test
partition, or perform MMP-directed generation.

All 10,000 fresh-validation seeds are retained, including a seed with no
accepted candidate. External inputs and relevant prior source files are
SHA-256-bound before analysis. Prior Step-1b and Step-2b artifacts are read-only.

## Candidate denominators

Step 2b persisted at most 50 unique, valid, policy-accepted molecular identities
per seed after canonicalization. The retained-set validity denominator is
therefore distinct from the 64 raw beam-hypothesis denominator. Both are
reported, together with the fraction of the nominal 500,000 retained slots that
were filled. The individual discarded raw strings were not persisted, so raw
beam exact-string duplication cannot be reconstructed; aggregate molecular-
identity redundancy remains recoverable from `final_generation_stats.csv`.

Every retained raw string is independently reparsed and passed through the same
hash-bound gMolAI canonicalization policy. Its recomputed canonical SMILES,
identity hash, scaffold, and atom count must reproduce the Step-2b record.

## Molecular identity and categories

Candidate identity is SHA-256 of canonical isomeric SMILES, exactly as in the
released chemistry policy. Classification is mutually exclusive and ordered:

1. exact seed identity using the seed's canonical spelling;
2. the seed identity expressed by an alternative raw SMILES spelling;
3. a genuine non-seed one-cut MMP derivative;
4. a scaffold-preserving non-MMP analogue;
5. an acyclic non-MMP analogue when both Bemis-Murcko scaffolds are empty;
6. a scaffold-changing analogue.

Thus alternate SMILES are never counted as derivatives. Empty Bemis-Murcko
scaffolds are reported separately rather than being interpreted as meaningful
scaffold retention.

## One-cut MMP definition

The audit imports the hash-bound Step-1b fragmentation implementation directly.
RDKit `rdMMPA.FragmentMol(maxCuts=1)` retains a fragmentation only when the core
has at least six heavy atoms, the variable fragment has 1--10 heavy atoms, and
the core is at least 60% of its parent. A seed/candidate match additionally
requires the same attachment-labelled core, different substituents, variable-
fragment heavy-atom difference at most six, and parent heavy-atom difference at
most eight.

All valid explanations are saved. If several cores explain a pair, MMP status
does not change; one primary label is selected deterministically by the largest
retained core, smallest total variable size, then lexical fields. Reported
growth/truncation/replacement labels describe substituent heavy-atom deltas and
are not reaction-mechanism claims.

## Metrics

- retained and raw-proposal validity, policy acceptance, capacity fill, raw and
  canonical uniqueness, seed identity, and accepted-beam identity redundancy;
- MMP derivative counts/fractions and seed coverage at 1, 5, 10, 20, 30, and 40;
- non-empty seed scaffold retention plus all scaffold-key and acyclic summaries;
- radius-2, 2,048-bit Morgan/Tanimoto distributions for all genuine non-seed
  candidates and each chemical category;
- descriptor deltas for heavy atoms, bonds, rings, heteroatoms, charge, and
  aromatic atoms/bonds; these are summaries, not unique graph-edit assignments;
- distinct identities/scaffolds and all within-set pairwise Morgan similarities.

The bounded conclusion follows the thresholds frozen in
`config/protocol.json`. It characterizes only the observed unperturbed Step-2b
sets and cannot establish controllable derivative generation.
