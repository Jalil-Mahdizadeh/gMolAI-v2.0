# TDC ADMET frozen-representation benchmark

This is the final active gMolAI study before manuscript preparation. It asks a
narrower and more defensible question than property-guided generation:

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

## Reproduction

The protocol is frozen in [protocol.json](protocol.json) and explained in
[PROTOCOL.md](PROTOCOL.md). The end-to-end entry point is:

```bash
sbatch extra-benchmark/tdc-admet/run_tdc_admet.sbatch
```

On an allocated interactive GPU node it can also be run directly:

```bash
bash extra-benchmark/tdc-admet/run_tdc_admet.sbatch
```

The runner is restart-safe: complete embeddings are checksum-validated before
reuse and partial or inconsistent outputs fail closed.

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
