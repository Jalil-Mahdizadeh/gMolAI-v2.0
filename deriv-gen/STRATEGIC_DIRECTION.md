# Frozen strategic direction

Decision date: 2026-08-14

Status: **derivative-generation study closed; future-work design retained**

## Decision

The project will not pursue the previously planned Step 3/Step 4 controlled
MMP-edit generation route. When a desired site-specific structural
modification is already known explicitly, such as a specified CH3-to-Cl
substitution, directly editing the molecular graph/SMILES is exact and
computationally preferable to an encoder-to-latent-edit-to-decoder workflow.
A gMolAI-native decoder will therefore not be developed for that objective.

This closes an execution strategy, not the scientific interpretation of the
latent-space studies. The completed Step 2d frozen-library result is also the
terminal derivative-generation experiment in the current project. No further
generation, property-guided selection, latent optimization, or decoder
development is required for manuscript completion.

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

## Closed generation scope

The completed generation evidence supports the bounded statement that the
frozen encoder, Step-2 decoder, and Step-2d sampling policy can propose large
libraries containing valid, unique, non-seed, chemically local, and
decoder-training-novel molecular identities, with heuristic SA-score context.
It does not establish property improvement, bioactivity, experimental
synthesizability, or an end-to-end molecular-optimization method.

The decoder should therefore be presented as a **candidate-proposal tool**.
Users may evaluate its exported candidates with independent, project-specific
property models, structural alerts, experimental measurements, availability
constraints, or synthesis planning, then retain the candidates appropriate to
their own objectives. Those downstream choices are intentionally outside the
frozen generator and are not validated by the present study.

## Future work, not current execution

The previously selected follow-up question remains scientifically legitimate:

`seed molecule -> frozen gMolAI embedding -> frozen decoder -> large candidate library -> property-guided prioritization`

Prioritization should seek improved desired property profiles while preserving
useful chemical locality, including similarity/scaffold relationships to the
seed, and reasonable synthetic-accessibility characteristics.

The scientific question is:

> Can the generated candidate library contain molecules with improved desired
> property profiles while retaining useful similarity/scaffold relationships
> to the seed and acceptable synthetic-accessibility characteristics?

It is retained as a possible separate future study, not as the next experiment
or an unresolved claim in the current project. If revisited, it must use
independent property evidence and preserve the frozen encoder, decoder, and
generation contract. No endpoint, property model, seed panel, ranking rule,
threshold, protocol, candidate generation, or evaluation has been selected or
run for that future work.

The final active benchmark direction for the current project is instead the
standardized frozen-representation
[TDC ADMET transfer comparison](../extra-benchmark/tdc-admet/). That benchmark
is scientifically separate from derivative generation: it tests whether
released representations support ADMET endpoint prediction and cannot be used
as evidence that generated candidates have improved ADMET profiles.

## Frozen generation baseline

The decoder and raw candidate-generation strategy for this direction are
permanently pinned by
[`shared/frozen-generation-v1/`](shared/frozen-generation-v1/). The contract
binds the exact existing Step-2 runtime checkpoint and inference export, and
the selected Step-2d `hybrid_b500_s500_t120` policy: 500 beam hypotheses plus
500 sample-stream hypotheses, temperature 1.2, top-p 0.995, deterministic
ordering/seed semantics, and 1,000 raw proposals per seed.

This is an immutable archival and future-work baseline, not an active
optimization protocol. Any future work must verify and consume it unchanged.
Any different decoder or sampling policy requires a new versioned contract and
an explicit superseding scientific decision.
