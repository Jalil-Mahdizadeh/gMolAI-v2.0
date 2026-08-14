# Frozen strategic direction

Decision date: 2026-08-14

Status: **decision recorded; next experiment not started**

## Decision

The project will not pursue the previously planned Step 3/Step 4 controlled
MMP-edit generation route. When a desired site-specific structural
modification is already known explicitly, such as a specified CH3-to-Cl
substitution, directly editing the molecular graph/SMILES is exact and
computationally preferable to an encoder-to-latent-edit-to-decoder workflow.
A gMolAI-native decoder will therefore not be developed for that objective.

This closes an execution strategy, not the scientific interpretation of the
latent-space studies.

## Evidence retained permanently

Step 1 and Step 1b, including their frozen artifacts and results, remain
permanent gMolAI representation-interpretability evidence:

- recurrent chemical transformations correspond to transferable latent
  directions;
- transfer generalizes across unseen molecular cores;
- the effect survives support thresholds of at least 5, 10, and 20 independent
  cores; and
- the released weight-3 (`released_hybrid_w3`) hybrid is a viable frozen edit
  geometry under the predeclared compatibility-aware selection rule.

These results establish latent chemical organization and transfer structure.
They do not establish latent editing as the preferred operational route for a
known edit, and they are not by themselves decoding, novel-generation,
property-improvement, or synthesis-feasibility claims.

Any prospective next-step recommendations inside the frozen Step 1/Step 1b
records remain part of their contemporaneous audit trail but are superseded for
execution planning by this decision.

The evidence remains in:

- [`step-01-latent-geometry-retrieval/`](step-01-latent-geometry-retrieval/)
- [`step-01b-scaled-space-selection/`](step-01b-scaled-space-selection/)

## Retired placeholders

The former `step-03-controlled-candidates/` and `step-04-native-decoder/`
directories were audited before removal. Each contained only a short reserved
README added in the derivative-generation feasibility commit. Neither
contained a protocol, implementation, execution record, output, result, or
scientific conclusion. They were therefore removed rather than archived; their
placeholder text remains recoverable from Git history.

## Next objective

The next project objective is:

`seed molecule -> frozen gMolAI embedding -> frozen decoder -> large candidate library -> property-guided prioritization`

Prioritization should seek improved desired property profiles while preserving
useful chemical locality, including similarity/scaffold relationships to the
seed, and reasonable synthetic-accessibility characteristics.

The scientific question is:

> Can the generated candidate library contain molecules with improved desired
> property profiles while retaining useful similarity/scaffold relationships
> to the seed and acceptable synthetic-accessibility characteristics?

This direction uses the frozen encoder/embedding definition and frozen decoder;
it does not reopen controlled MMP-direction editing. No endpoint, property
model, seed panel, ranking rule, threshold, protocol, or experiment has yet been
selected or run.
