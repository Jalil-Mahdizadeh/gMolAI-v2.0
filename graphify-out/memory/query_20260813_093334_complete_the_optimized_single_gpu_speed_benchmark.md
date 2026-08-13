---
type: "query"
date: "2026-08-13T09:33:34.057044+00:00"
question: "Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation."
contributor: "graphify"
outcome: "useful"
source_nodes: ["speed_adapter.py", "OptimizedSmilesEncoder", "container_gpu()", "load_gmolai()", "atomic_write_json()"]
---

# Q: Complete the optimized single-GPU speed benchmark for gMolAI, Morgan, and competitor models at batch sizes 64, 128, 256, and 512 with the qualified worker count, then replace old speed artifacts and update documentation.

## Answer

Completed the definitive optimized speed benchmark on Arrhenius n54 in Slurm job 1230738, using one NVIDIA GH200 GPU, 48 CPUs, and batch sizes 64, 128, 256, and 512 for all seven encoders. The unified speed_adapter.py loads gMolAI through load_gmolai() and OptimizedSmilesEncoder (optimized_gine_v1), warms its 48-process RDKit pool outside the timer, and times full-panel canonical SMILES through ordered FP32 host vectors. gMolAI achieved 13,040.69, 22,901.45, 40,068.75, and 58,330.38 molecules/s, corresponding to 1.21x, 2.12x, 3.68x, and 5.36x Morgan and 3.59x, 4.96x, 7.76x, and 10.04x the fastest other representation-equivalent neural model. All output-equivalence gates passed except the known native KERMT v2 batch-padding nonconformance, which is explicitly marked record-only. finalize.py produced 28 validated conditions, compact JSON/CSV, latency CSV, PNG/PDF/SVG figures, raw provenance, COMPLETE state, and a 16-file SHA-256 ledger. All checksums and semantic contracts passed, the full repository test suite passed 59/59, stale speed results were permanently removed, and README/PROTOCOL/KERMT documentation was updated.

## Outcome

- Signal: useful

## Source Nodes

- speed_adapter.py
- OptimizedSmilesEncoder
- container_gpu()
- load_gmolai()
- atomic_write_json()