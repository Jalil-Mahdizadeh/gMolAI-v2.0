# Decoder design rationale

The repository's existing `GraphConditionedEdgeDecoder` in
`src/gmolai_retrain/model.py` is not a molecular inverse. It predicts edge
types from per-node latent states plus a graph context during representation
training. A released 384-D molecular vector does not contain or expose the
required node-state sequence, atom count, or atom identities, so reusing that
module would not provide a deployable decoder from the public representation.

The released data path already defines targets as canonical isomeric SMILES
through `src/gmolai_retrain/chem.py`. A conditional autoregressive SMILES
decoder is therefore the most practical inverse to test without changing
gMolAI. The fixed ASCII-byte tokenizer is lossless, has no validation OOV
problem, and can represent brackets, charges, ring closures, atom chirality,
and bond stereochemistry. Generated strings are still parsed and filtered by
the unchanged gMolAI chemistry policy.

The frozen 384-D released vector is projected into four cross-attention memory
tokens and an additive token-stream bias. Conditioning is therefore available
at every decoder layer and time step. Training includes a wrong-condition
margin, while held-out generation explicitly compares correct, shuffled, zero,
and nearest-wrong-molecule embeddings. These controls are essential because
teacher-forced SMILES likelihood alone could be achieved by an effectively
unconditional language model.

This design tests zero-perturbation inversion only. It does not add a decoder
head to gMolAI, alter its checkpoint or calibrator, train on endpoints, or
perform latent perturbation.
