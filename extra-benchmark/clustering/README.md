# gMolAI minimal external clustering benchmark

This directory is a self-contained, frozen, inference-only manuscript benchmark. It compares gMolAI, Morgan, MolAI, MoLFormer, SMI-TED-Light, MolCLR-GIN, and KERMT v2 on ClassyFire-25 structural organization and QMugs quantum-property neighborhoods. Count Morgan and the 13 pretraining descriptors are labeled diagnostics only.

`PROTOCOL.md` explains the design; `protocol.json` is authoritative. Raw pinned inputs and provenance live under `inputs/`; immutable model outputs live under `artifacts/`; tables, figures, and their source data live under `outputs/`; verification records live under `audit/`; and all executable benchmark code lives under `scripts/`. `RESULTS.md` is generated only after the full audit passes.

The top-level runner is resume-safe and fail-closed: it verifies every checksum, prepares identities without model outputs, screens all model adapters, freezes common support, exports frozen vectors, evaluates the two prespecified endpoints, generates figures, and seals the output manifest.

From the repository root, reproduce or resume the complete workflow with:

```bash
sbatch extra-benchmark/clustering/run_clustering.sbatch
```

On an already allocated single-GPU node, the same file can be run with `bash`. The pinned raw inputs are retained in `inputs/raw`; no benchmark step trains or fine-tunes a model. Full installed-package inventories for every unique SIF are recorded under `audit/container_packages/`.

## Git and artifact policy

The local completed bundle retains the pinned raw datasets, prepared identity manifests, frozen embeddings, exact neighbor/cluster arrays, bootstrap arrays, adapter-screen payloads, and execution logs. These large or regenerable files are intentionally excluded from ordinary Git transport by this directory's `.gitignore`. The committed record contains the complete protocol and implementation, dataset/model provenance, audit summaries, publication tables and figures, and exact source data for every plotted figure. `outputs/SHA256SUMS` seals the full local bundle, including the locally retained files; a fresh clone must fetch the pinned inputs and run the resume-safe workflow before that full-bundle manifest can validate.
