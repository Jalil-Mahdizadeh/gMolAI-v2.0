---
type: "query"
date: "2026-08-11T11:09:50.597045+00:00"
question: "one important question has emerged: at training step 15k, all parameters have been improved just FreeSolv. Reviewers will definetly ask why we did not promote 15k checkpoint instead."
contributor: "graphify"
outcome: "useful"
source_nodes: ["Promotion Integrity Gates", "_check_gate()", "_validate_promotion_quality", "Semantic Promotion Suite", "Validation Evidence Bundle"]
---

# Q: one important question has emerged: at training step 15k, all parameters have been improved just FreeSolv. Reviewers will definetly ask why we did not promote 15k checkpoint instead.

## Answer

Expanded from original query via graph vocabulary: [checkpoint, selection, promotion, gate, screening, downstream, validation, objective, metrics, seed, probe, transferable]. The repository supports retaining step 10k only as a deterministic fail-closed promotion decision, not as proof that 10k is statistically superior. The code enforces FreeSolv mean RMSE <=1.30; raw-hybrid screen means were 1.297036 at 10k and 1.303508 at 15k, so 10k passed and 15k failed. The paired 15k-minus-10k difference is +0.006472 kcal/mol (+0.499%); 6/10 paired outer splits favor 10k and 4/10 favor 15k. A naive paired 95% interval includes zero, and overlapping repeated scaffold splits preclude a formal independence-based inference. Therefore manuscript language should say numerically lower/operationally selected, not 10k optimum or transfer peaked. Also, only selected intrinsic diagnostics improved at 15k; the complete final 100k-calibration/50k-probe/five-dataset promotion panel was not run for 15k. The existing 15k probe is a smaller 10k-calibration/5k-validation screen. Strongest reviewer-proof additions are a full 15k qualification panel and seed43 10k-vs-15k FreeSolv replication, without consulting the already opened internal test. Git history contains the 1.30 gate and results in the same initial commit, so 'predeclared' needs external dated evidence; otherwise use 'fixed promotion threshold'.

## Outcome

- Signal: useful

## Source Nodes

- Promotion Integrity Gates
- _check_gate()
- _validate_promotion_quality
- Semantic Promotion Suite
- Validation Evidence Bundle