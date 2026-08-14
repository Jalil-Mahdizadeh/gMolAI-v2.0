# gMolAI derivative generation studies

This directory contains an additive, versioned investigation of molecular
derivative generation from the frozen promoted gMolAI representation. Nothing
in this directory changes the released encoder, calibrator, inference package,
or earlier benchmarks.

## Study layout

| Step | Directory | Status | Purpose |
|---:|---|---|---|
| 1 | `step-01-latent-geometry-retrieval/` | Completed | Test latent geometry, matched-pair directions, interpolation, and retrieval |
| 1b | `step-01b-scaled-space-selection/` | Completed and verified | Scale MMP mining to 1M train molecules and freeze the latent edit-control space |
| 2 | `step-02-decoder-feasibility/` | Completed and verified; faithful-inverse NO-GO | Train and control-test a decoder conditioned on the frozen released 384D representation |
| 2b | `step-02b-candidate-reranking/` | Completed and verified; GO | Test frozen-decoder search and target-blind frozen-gMolAI latent reranking on a fresh validation panel |
| 2c | `step-02c-chemical-characterization/` | Completed and verified | Chemically classify the frozen unperturbed candidate sets |
| 2d | `step-02d-generation-scaling/` | Completed and verified | Scale unperturbed candidate generation from 50 to 1,000 proposals and audit SA-score context |
| 3 | `step-03-controlled-candidates/` | Reserved | Generate, sanitize, re-encode, and rank controlled candidates |
| 4 | `step-04-native-decoder/` | Reserved | Evaluate a gMolAI-native conditional decoder if earlier gates justify it |

Steps 1 and 1b are retrieval-geometry feasibility studies, not decoding
claims. Step 2 trained only a new decoder and demonstrated strong use of the
released condition, but greedy decoding did not meet the frozen fidelity
gates. Step 2b froze that decoder and showed prospectively that the deficit was
primarily search-related: target-blind latent reranking reached 93.92% exact
identity on a fresh 10,000-molecule validation panel. Every completed step uses
immutable train/validation inputs; the locked internal test partition and all
endpoint labels are excluded.

Step 2c established that the frozen decoder's unperturbed candidate sets
already contain genuine non-seed analogues and one-cut MMP derivatives rather
than merely alternative SMILES. Step 2d then quantified their scaling,
locality, diversity, decoder-training novelty, and heuristic synthetic
accessibility through 1,000 proposals per seed. No latent perturbation,
MMP-direction editing, property optimization, or Step 3 generation has occurred.

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
The current cross-step handoff is [`CHECKPOINT.md`](CHECKPOINT.md).

## Artifact policy

Source code, protocols, manifests, completion seals, compact machine-readable
tables, decision records, and figures are versioned. Large generated model,
Parquet, and NumPy array artifacts remain on the project filesystem and are
excluded by the repository-wide ignore policy. Each step's `outputs/SHA256SUMS`
records the complete local run, while the scripts and frozen manifests provide
the reproducible path to regenerate omitted bulk artifacts.
