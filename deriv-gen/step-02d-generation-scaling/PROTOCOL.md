# Frozen protocol: Step 2d candidate-generation scaling

## Scientific boundary

Step 2d evaluates candidate generation from the unperturbed seed embedding only.
The seed-42/step-10,000 gMolAI checkpoint, immutable calibrator, released
384-dimensional ×3 representation, and selected Step-2 decoder checkpoint stay
frozen. No optimizer is created. There is no latent perturbation, MMP-direction
edit, property optimization, endpoint label, or locked-test input.

Prior Step-1b through Step-2c directories are hash-bound, read-only inputs. All
new files remain under this Step-2d directory.

## Development and final separation

Generation strategy is selected on 512 deterministic molecules from the
scaffold-disjoint decoder-development holdout, excluding the Step-2b development
panel. Seven registered streams compare wide beam search, three fixed-seed
temperature/top-p regimes, and two balanced beam/sampling hybrids. Base pools
are shared only to avoid duplicate computation; every reported strategy has an
explicit 1,000-proposal stream.

The selected strategy, its parameters, ordering, and per-molecule seed rule are
sealed before final generation. Final evaluation uses 10,000 validation
molecules excluded from both prior Step-2 and Step-2b generation panels.

## Budget and candidate definitions

Budgets 50, 100, 250, 500, and 1,000 are nested prefixes of **raw decoder
proposal slots**. This denominator permits direct reporting of invalid outputs,
policy rejection, raw-string duplication, canonical-identity duplication, and
unique chemical yield. A unique accepted candidate enters every budget at or
above its first proposal rank. Thus larger budgets never discard a molecule
seen at a smaller budget.

The unchanged gMolAI policy canonicalizes isomeric SMILES, rejects disconnected
molecules and disallowed elements, and enforces the same atom-count limits.
Identity is SHA-256 of canonical isomeric SMILES.

## Development selection

The primary utility is the mean count per seed, at budget 1,000, of novel
genuine non-seed candidates that either satisfy the exact Step-1b one-cut MMP
definition or retain a non-empty seed Bemis-Murcko scaffold. Invalid, rejected,
duplicate, seed-identical, remote non-MMP, and decoder-training identities add
zero primary utility.

Eligibility and the fixed 2% equivalence margin are specified in
`config/protocol.json`. Within the equivalence set, selection favors novel MMP
yield, then novel same-scaffold non-MMP yield, total novel non-seed yield,
policy acceptance, and registered order. A paired seed bootstrap quantifies
strategy differences but does not change this rule.

## Chemistry and novelty

One-cut MMP matching imports the exact hash-bound Step-1b fragmentation code:
core ≥6 heavy atoms, variable fragment 1--10 heavy atoms, core ≥60% of parent,
variable-size difference ≤6, and parent-size difference ≤8. Alternative SMILES
of the seed are seed identity, never derivatives.

Novelty means absence from exactly the 980,000 molecular identities used to fit
the decoder. The 20,000 decoder-development molecules, validation molecules,
and locked test partition are not novelty reference sets.

Bemis-Murcko retention is reported only for non-empty seed scaffolds. Both-empty
acyclic cases are separate. Morgan similarity uses radius 2 and 2,048 bits.
Within-set pairwise distributions are exact when at most 2,048 pairs exist and
otherwise use a deterministic uniform sample of 2,048 unordered pairs per
seed/budget; the approximation is labelled throughout.

## Scaling and decision rules

Every final seed is evaluated at every nested budget. Incremental interval
tables quantify new accepted identities, MMPs, same-scaffold analogues, novel
molecules, novel useful-local molecules, and scaffolds per additional raw
proposal. Diminishing returns begin when the interval utility rate is at most
50% of the first-50 rate; saturation requires at most 25% with all later
intervals also below that level.

The recommended budget is the smallest registered budget attaining at least
90% of the budget-1,000 mean novel useful-local yield. A prospective bounded
large-library classification uses the gates in `config/protocol.json`. These
rules cannot establish synthesizability, activity, property improvement, or
latent edit control.
