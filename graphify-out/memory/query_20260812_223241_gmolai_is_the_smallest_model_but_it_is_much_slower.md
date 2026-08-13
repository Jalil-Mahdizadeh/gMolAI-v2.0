---
type: "query"
date: "2026-08-12T22:32:41.187954+00:00"
question: "gmolai is the smallest model but it is much slower compare to others. why is it?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["featurize_molecule()", "Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)", "PyTorch Geometric RDKit Inference Environment"]
---

# Q: gmolai is the smallest model but it is much slower compare to others. why is it?

## Answer

Expanded from original query via graph vocab: [benchmark, encoder, encoding, parameters, graph, geometric, gine, batch, gpu, inference, smiles, molecule]. The graph pointed to featurize_molecule and the four-layer GINE encoder. Repository verification shows that the measured path is end-to-end SMILES-to-host-vector encoding: serial RDKit parsing and atom/bond feature construction, PyG Data creation and collation, sparse GINE/scatter execution, three graph pooling operations, device transfers, host concatenation, and calibration. Parameter count measures model storage, not these overheads. The weak batch scaling and low peak allocation are consistent with an overhead-bound, under-filled GPU path; exact stage shares require profiling.

## Outcome

- Signal: useful

## Source Nodes

- featurize_molecule()
- Masked Graph VICReg Architecture (256 hidden, 128 node latent, 256 graph latent, four GINE layers)
- PyTorch Geometric RDKit Inference Environment