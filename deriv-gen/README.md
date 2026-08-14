# gMolAI derivative generation studies

This directory contains an additive, versioned investigation of molecular
derivative generation from the frozen promoted gMolAI representation, together
with the frozen strategic interpretation of the completed evidence. Nothing in
this directory changes the released encoder, calibrator, inference package, or
earlier benchmarks.

**Project status:** closed at Step 2d on 14 August 2026. All completed evidence
and frozen artifacts are retained, but no additional derivative-generation or
property-guided optimization experiment is part of the current project.

## Study layout

| Step | Directory | Status | Purpose |
|---:|---|---|---|
| 1 | `step-01-latent-geometry-retrieval/` | Completed; permanently retained | Test latent organization, matched-pair direction transfer, interpolation, and retrieval |
| 1b | `step-01b-scaled-space-selection/` | Completed, verified, and permanently retained | Test transfer at 1M scale and freeze the released x3 hybrid as a viable diagnostic edit geometry |
| 2 | `step-02-decoder-feasibility/` | Completed and verified; faithful-inverse NO-GO | Train and control-test a decoder conditioned on the frozen released 384D representation |
| 2b | `step-02b-candidate-reranking/` | Completed and verified; GO | Test frozen-decoder search and target-blind frozen-gMolAI latent reranking on a fresh validation panel |
| 2c | `step-02c-chemical-characterization/` | Completed and verified | Chemically classify the frozen unperturbed candidate sets |
| 2d | `step-02d-generation-scaling/` | Completed and verified | Scale unperturbed candidate generation from 50 to 1,000 proposals and audit SA-score context |

The previously reserved Step 3 controlled-candidate and Step 4 native-decoder
branches have been retired. Their directories contained only unexecuted
placeholder READMEs, so no scientific or protocol artifact required archival.
The rationale and replacement direction are frozen in
[`STRATEGIC_DIRECTION.md`](STRATEGIC_DIRECTION.md).

Steps 1 and 1b are permanently retained as representation-interpretability
studies. They show that recurrent chemical transformations correspond to
transferable latent directions across unseen molecular cores, that the signal
survives support thresholds of at least 5, 10, and 20 independent cores, and
that the released x3 hybrid is a viable frozen edit geometry. These findings
describe latent chemical organization; they do not make latent editing the
preferred way to execute a known structural change and are not decoding or
generation claims.

Step 2 trained only a new decoder and demonstrated strong use of the released
condition, but greedy decoding did not meet the frozen fidelity gates. Step 2b
froze that decoder and showed prospectively that the deficit was primarily
search-related: target-blind latent reranking reached 93.92% exact identity on
a fresh 10,000-molecule validation panel. Every completed step uses immutable
train/validation inputs; the locked internal test partition and all endpoint
labels are excluded.

Step 2c established that the frozen decoder's unperturbed candidate sets
already contain genuine non-seed analogues and one-cut MMP derivatives rather
than merely alternative SMILES. Step 2d then quantified their scaling,
locality, diversity, decoder-training novelty, and heuristic synthetic
accessibility through 1,000 proposals per seed. No latent perturbation,
MMP-direction editing, or property optimization occurred. The controlled-edit
branch is now closed: when a desired site-specific structural modification is
already explicit, direct graph/SMILES editing is exact and computationally
preferable to an encoder-to-latent-edit-to-decoder workflow.

The operational generation baseline is now permanently versioned in
[`shared/frozen-generation-v1/`](shared/frozen-generation-v1/). It binds the
exact existing Step-2 runtime checkpoint and inference export, plus the full
`hybrid_b500_s500_t120` strategy: 500 beam and 500 sample-stream hypotheses,
temperature 1.2, top-p 0.995, deterministic ordering and seed derivation, and
1,000 raw proposals per seed. This additive contract does not modify any
completed-step artifact or start a new experiment.

## Closure and future-work boundary

The current derivative-generation study ends with the frozen Step-2d candidate
library. It establishes proposal yield, locality, diversity,
decoder-training-novelty, and heuristic SA-score context; it does not establish
property improvement, bioactivity, synthesis feasibility, or optimized
molecules.

The released decoder is therefore a candidate-proposal mechanism. Users may
apply independent, project-specific prediction models, experimental evidence,
structural filters, and synthesis constraints to the exported molecules. The
paper must describe that process as downstream user prioritization rather than
as a property-optimization result of gMolAI.

The following question is retained only as possible separate future work:

`seed molecule -> frozen gMolAI embedding -> frozen decoder -> large candidate library -> property-guided prioritization`

Candidate prioritization must retain useful chemical locality and
similarity/scaffold relationships to the seed while maintaining reasonable
synthetic-accessibility characteristics. The scientific question is:

> Can the generated candidate library contain molecules with improved desired
> property profiles while retaining useful similarity/scaffold relationships
> to the seed and acceptable synthetic-accessibility characteristics?

No property target, prioritization protocol, experiment, or new
candidate-generation run has been started, and none is required to close the
current project. Any later protocol using this route must consume and verify the
[`frozen-generation-v1`](shared/frozen-generation-v1/) baseline rather than
retraining the decoder or retuning its sampling strategy in place.

The final active study before manuscript preparation is a standardized
frozen-representation TDC ADMET benchmark. It is separate from this directory
and must not be interpreted as validation of the generated candidates.

## Reproducing Step 1

From the repository root:

```bash
bash deriv-gen/step-01-latent-geometry-retrieval/scripts/run_day1.sh
```

The launcher exposes exactly the current single GPU to the pinned gMolAI SIF,
binds the repository read-only, and grants write access only to `deriv-gen/`.
All runtime caches and temporary files are redirected beneath this directory.

Read the Day-1 interpretation in
[`step-01-latent-geometry-retrieval/RESULTS.md`](step-01-latent-geometry-retrieval/RESULTS.md).

Reproduce the scaled five-space study with:

```bash
bash deriv-gen/step-01b-scaled-space-selection/scripts/run_study.sh
```

Its verified result and frozen decision are in
[`step-01b-scaled-space-selection/RESULTS.md`](step-01b-scaled-space-selection/RESULTS.md)
and
[`step-01b-scaled-space-selection/DECISION.md`](step-01b-scaled-space-selection/DECISION.md).


Reproduce the decoder feasibility study with:

```bash
bash deriv-gen/step-02-decoder-feasibility/scripts/run_study.sh
```

Its verified held-out metrics and frozen NO-GO decision are in
[`step-02-decoder-feasibility/RESULTS.md`](step-02-decoder-feasibility/RESULTS.md)
and
[`step-02-decoder-feasibility/DECISION.md`](step-02-decoder-feasibility/DECISION.md).

Reproduce the frozen candidate-search and latent-reranking study with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 apptainer exec --nv \
  --bind "$PWD:/repo" \
  --bind /nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers:/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers:ro \
  /nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif \
  bash /repo/deriv-gen/step-02b-candidate-reranking/scripts/run_study.sh /repo
```

Its verified fresh-panel metrics and GO decision are in
[`step-02b-candidate-reranking/RESULTS.md`](step-02b-candidate-reranking/RESULTS.md)
and
[`step-02b-candidate-reranking/DECISION.md`](step-02b-candidate-reranking/DECISION.md).

Reproduce the no-training chemical audit with:

```bash
bash deriv-gen/step-02c-chemical-characterization/scripts/run_study.sh
```

Its bounded chemical interpretation is in
[`step-02c-chemical-characterization/RESULTS.md`](step-02c-chemical-characterization/RESULTS.md)
and
[`step-02c-chemical-characterization/DECISION.md`](step-02c-chemical-characterization/DECISION.md).

Step 2d's GPU generation was completed through its registered SLURM workflow.
Its CPU-only post-completion SA extension can be reproduced from existing
candidate artifacts with:

```bash
bash deriv-gen/step-02d-generation-scaling/scripts/run_sa_extension.sh
```

Read the frozen scaling decision and additive SA interpretation in
[`step-02d-generation-scaling/RESULTS.md`](step-02d-generation-scaling/RESULTS.md)
and
[`step-02d-generation-scaling/DECISION.md`](step-02d-generation-scaling/DECISION.md).
The current strategy is [`STRATEGIC_DIRECTION.md`](STRATEGIC_DIRECTION.md), and
the cross-step operational handoff is [`CHECKPOINT.md`](CHECKPOINT.md). The
machine-verifiable decoder/sampler freeze is
[`shared/frozen-generation-v1/`](shared/frozen-generation-v1/).

## Artifact policy

Source code, protocols, manifests, completion seals, compact machine-readable
tables, decision records, and figures are versioned. Large generated model,
Parquet, and NumPy array artifacts remain on the project filesystem and are
excluded by the repository-wide ignore policy. Each step's `outputs/SHA256SUMS`
records the complete local run, while the scripts and frozen manifests provide
the reproducible path to regenerate omitted bulk artifacts. The shared frozen
generation contract additionally binds both excluded Step-2 decoder files by
SHA-256 and byte size.
