---
type: "query"
date: "2026-08-13T09:46:02.179242+00:00"
question: "is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["generate_embeddings.py", "build_smiles_encoder()", "OptimizedSmilesEncoder", ".encode()"]
---

# Q: is the use face folder 'inference' current? I mean is it using the speed-optimized pipeline instead of old slow one?

## Answer

Expanded from original query via graph vocabulary: [inference, generate, embeddings, optimized, smiles, encoder, fast, pipeline, canonical, representation, workers]. Yes, the live user-facing inference/generate_embeddings.py defaults to the speed-optimized pipeline. Its parser defaults --backend optimized, --batch-size 192, --node-budget 16384, and --workers auto. run_inference() calls build_smiles_encoder(), which returns OptimizedSmilesEncoder for that default; larger workloads are buffered across worker-sized pipeline windows so multiprocess RDKit packing is activated, while tiny inputs stay inline to avoid process overhead. The old PyG path remains available only when explicitly selecting --backend reference, and --backend verify returns optimized results after a bounded reference comparison. A direct default CLI smoke test in the pinned gMolAI SIF reported backend optimized_gine_v1, CUDA device 0, 48 workers, batch size 192, and node budget 16384. Important caveat: these changes are present in the live working tree but not committed to HEAD/main; generate_embeddings.py and inference/README.md are modified, while fast_graph.py, fast_inference.py, and their tests are untracked.

## Outcome

- Signal: useful

## Source Nodes

- generate_embeddings.py
- build_smiles_encoder()
- OptimizedSmilesEncoder
- .encode()