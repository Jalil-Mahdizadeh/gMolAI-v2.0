# TDC ADMET frozen-representation benchmark

This completed broad-transfer gMolAI benchmark asks a narrower and more
defensible question than property-guided generation:

> Does the already-frozen gMolAI representation support competitive simple
> predictors across the complete TDC ADMET panel, relative to the same frozen
> comparator representations used in the earlier benchmark?

The study covers all 22 TDC ADMET endpoints from the DOI-backed 2026-03-24
snapshot. Neural encoders remain frozen. Only fold-local feature scaling and a
Ridge or logistic-regression probe are fitted. The fixed TDC test sets are never
used for model or hyperparameter selection.

The seven primary representations are gMolAI, Morgan radius-2, MolAI,
MoLFormer, SMI-TED-Light, MolCLR-GIN, and KERMT v2. A compact 13-property RDKit
descriptor panel is reported separately as a diagnostic control; it is excluded
from the primary seven-model rank summary.

This is a frozen-representation benchmark, not a TDC leaderboard submission:
the common panel applies the repository's molecular policy and requires support
from every encoder. Full source coverage and every exclusion are reported. The
five values per endpoint measure sensitivity to the official train/validation
scaffold split used for probe selection; they are not independent test-set
replicates or standard errors.

The complete 22-endpoint summary is accompanied by a preregistered 19-endpoint
sensitivity excluding BBB, Lipophilicity, and AqSolDB because a source-only audit
found exact or near-exact reuse of earlier representation-development evidence.
Every excluded sensitivity endpoint remains visible in the full result tables.

## Completed result

The benchmark completed on 14 August 2026 inside interactive Slurm allocation
`1255034` on n42. The all-model panel contains 43,730 unique identities and
78,131 labeled occurrences. SMI-TED-Light ranked first and gMolAI second in
both the complete 22-endpoint and predeclared 19-endpoint category-balanced
summaries. gMolAI had the most individual endpoint wins: 9/22 and 8/19,
respectively.

Read the complete endpoint table, limitations, coverage, selection-conditioning
analysis, and checksum audit in [RESULTS.md](RESULTS.md).

## Verification and reproduction

Verify the completed local artifacts without changing them:

```bash
PYTHONPATH=extra-benchmark/tdc-admet/scripts \
  python extra-benchmark/tdc-admet/scripts/verify_results.py
```

The protocol is frozen in [protocol.json](protocol.json) and explained in
[PROTOCOL.md](PROTOCOL.md). The end-to-end entry point is:

```bash
sbatch extra-benchmark/tdc-admet/run_tdc_admet.sbatch
```

On an allocated interactive GPU node it can also be run directly:

```bash
bash extra-benchmark/tdc-admet/run_tdc_admet.sbatch
```

The runner is restart-safe for deliberate reproduction: complete embeddings
are checksum-validated before reuse and partial or inconsistent outputs fail
closed. Because it recomputes and reseals compact endpoint outputs, use the
read-only verifier for routine integrity checks of this completed run.

The protocol records one diagnostic-only runtime amendment made after the
initial RDKit descriptor export failed and before any endpoint probe existed.
Exactly 12 undefined partial-charge descriptor values on six identities are
median-imputed within each training fold. The common panel and all seven
primary representations are unchanged.

## Interpretation boundary

This benchmark evaluates representation utility for ADMET prediction. It does
not score decoder-generated analogues, establish prospective ADMET improvement,
or reopen the closed derivative-generation workflow. Property-guided candidate
optimization remains explicitly future work.

The later
[external molecular-clustering benchmark](../clustering/RESULTS.md) is a
separate label-structure and retrieval study. It does not change this TDC
protocol, result, or interpretation boundary.
