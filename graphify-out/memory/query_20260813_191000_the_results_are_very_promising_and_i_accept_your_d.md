---
type: "query"
date: "2026-08-13T19:10:00.725419+00:00"
question: "the results are very promising and I accept your decission. Proceed to the next step only: train and validate a molecular decoder conditioned on the frozen released gMolAI representation."
contributor: "graphify"
outcome: "useful"
source_nodes: ["GraphConditionedEdgeDecoder", "CanonicalMolecule", "canonicalize", "OptimizedSmilesEncoder"]
---

# Q: the results are very promising and I accept your decission. Proceed to the next step only: train and validate a molecular decoder conditioned on the frozen released gMolAI representation.

## Answer

Expanded from the original request via graph vocabulary: [decoder, smiles, tokenizer, sequence, canonical, stereochemistry, embeddings, calibrator, inference, validation, molecule]. Repository traversal showed that GraphConditionedEdgeDecoder is an edge predictor requiring node latents, not an inverse from the public 384-D vector; CanonicalMolecule/canonicalize and OptimizedSmilesEncoder define the chemistry and re-encoding contracts. Implemented and verified a decoder-only byte-SMILES Transformer on released_hybrid_w3. Held-out validation: valid 0.9752, exact identity 0.6390, scaffold 0.7470, Morgan 0.8495, median re-encoded cosine 1.0000. Shuffled and nearest-wrong conditions recovered their supplied molecules at 0.6392 and 0.6612 while original-target recovery was 0.0000 and 0.0007. Frozen decision: NO-GO for a faithful inverse, although condition dependence is demonstrated. Zero locked-test rows, zero endpoint labels, zero gMolAI parameters optimized, and no latent perturbation was performed.

## Outcome

- Signal: useful

## Source Nodes

- GraphConditionedEdgeDecoder
- CanonicalMolecule
- canonicalize
- OptimizedSmilesEncoder