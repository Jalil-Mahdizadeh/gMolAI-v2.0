---
type: "query"
date: "2026-08-12T19:13:29.783066+00:00"
question: "Calculate the embedding time for each model and Morgan from the MoleculeNet plus HIV benchmark"
contributor: "graphify"
outcome: "useful"
source_nodes: ["export_embeddings()", "benchmark_io.py", "Morgan Fingerprint Baseline"]
---

# Q: Calculate the embedding time for each model and Morgan from the MoleculeNet plus HIV benchmark

## Answer

Expanded from the repository graph vocabulary via [benchmark, embedding, embeddings, export, model, morgan, runtime]. The sealed 45,504-row runtime CSV records load-plus-warm-up-plus-preprocessing-plus-inference-plus-export times: gMolAI 135.498 s, Morgan 6.801 s, MolAI epoch 6 12.233 s, MoLFormer 19.290 s, SMI-TED-Light 62.971 s, MolCLR-GIN 31.782 s, and KERMT v2 149.911 s. Morgan was CPU-based; neural models used one GPU. These are observed export timings, not isolated model-only latency.

## Outcome

- Signal: useful

## Source Nodes

- export_embeddings()
- benchmark_io.py
- Morgan Fingerprint Baseline