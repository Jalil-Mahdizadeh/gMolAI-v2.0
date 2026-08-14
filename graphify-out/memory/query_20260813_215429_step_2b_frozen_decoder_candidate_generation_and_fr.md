---
type: "query"
date: "2026-08-13T21:54:29.246937+00:00"
question: "Step 2b frozen-decoder candidate generation and frozen-gMolAI latent reranking on a fresh validation panel"
contributor: "graphify"
outcome: "useful"
source_nodes: ["OptimizedSmilesEncoder", "generate_embeddings.py", "Promoted gMolAI 384-Dimensional Molecular Vector", "Encoder/Reconstructor Capability Boundary"]
---

# Q: Step 2b frozen-decoder candidate generation and frozen-gMolAI latent reranking on a fresh validation panel

## Answer

Expanded from the repository vocabulary via [decoder, conditional, decode, embedding, encoder, inference, latent, molecular, reconstruction, sampling, similarity, validation]. Graph traversal identified OptimizedSmilesEncoder, generate_embeddings.py, the promoted 384-D vector contract, and the encoder/reconstructor capability boundary as the frozen paths to bind. Step 2b then froze checkpoint/calibrator/decoder, selected beam64_lp00 on train-development only, and achieved fresh-validation exact identity 93.92% from greedy 63.93% using target-blind released-gMolAI relative-L2 reranking; shuffled and nearest-wrong controls followed supplied sources at 93.92% and 95.13%. No retraining, test data, endpoints, perturbation, or derivative generation.

## Outcome

- Signal: useful

## Source Nodes

- OptimizedSmilesEncoder
- generate_embeddings.py
- Promoted gMolAI 384-Dimensional Molecular Vector
- Encoder/Reconstructor Capability Boundary