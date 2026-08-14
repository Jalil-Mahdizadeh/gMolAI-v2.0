---
type: "query"
date: "2026-08-14T18:15:38.445888+00:00"
question: "There is a folder in the root named 'inference' which is the user-faced SMILES -> gMolAI embeddings hydride x3. I want to re-structure it so that: it takes a list of SMILES and generates the gMolAI release hybride x3 embedings in .npz or .npy format (you decide whic one). I also want some nice CLI flags for when user want to generate candidates SMILES from the corresponding embeddings, each in a separate CSV files (a single CSV for each seed molecule). The candidates must be valid and unique canonical SMILES with a column for Morgan/Tanimoto similarity values. Run the example_smiles.csv both embeddings and 1000 derivatives for each smiles. Ensure that the this user-faced worflow uses the correct ckeckpoints, sampling strategy, etc. rename '/inference/model/' to '/inference/models/' and place the correct decoder checkpoint there and validate all check points and calibrator. Update the README.md accordingly, particularly explain the CLI flags for both encode (embeddings generation) and decoder (candidate generation from embeddings). Do it professionaly and user friendly."
contributor: "graphify"
outcome: "useful"
source_nodes: ["gmolai.py", "run_encode()", "run_decode()", "ConditionalSmilesTransformer", "generate_embeddings.py", "Frozen generation baseline v1", "canonicalize()"]
---

# Q: There is a folder in the root named 'inference' which is the user-faced SMILES -> gMolAI embeddings hydride x3. I want to re-structure it so that: it takes a list of SMILES and generates the gMolAI release hybride x3 embedings in .npz or .npy format (you decide whic one). I also want some nice CLI flags for when user want to generate candidates SMILES from the corresponding embeddings, each in a separate CSV files (a single CSV for each seed molecule). The candidates must be valid and unique canonical SMILES with a column for Morgan/Tanimoto similarity values. Run the example_smiles.csv both embeddings and 1000 derivatives for each smiles. Ensure that the this user-faced worflow uses the correct ckeckpoints, sampling strategy, etc. rename '/inference/model/' to '/inference/models/' and place the correct decoder checkpoint there and validate all check points and calibrator. Update the README.md accordingly, particularly explain the CLI flags for both encode (embeddings generation) and decoder (candidate generation from embeddings). Do it professionaly and user friendly.

## Answer

Expanded from original query via graph vocabulary: [inference, embeddings, encoder, decoder, hybrid, checkpoint, calibrator, frozen, sampling, candidates, canonical, similarity]. The graph connected the legacy generate_embeddings.py encoder, frozen generation baseline, canonicalization policy, and decoder sources. Implemented gmolai.py with encode/decode/validate commands, a self-describing released_hybrid_w3 NPZ bundle, one valid unique canonical candidate CSV per seed, Morgan radius-2/2048-bit Tanimoto annotations, strict artifact hashes, the compact frozen decoder, and the immutable hybrid_b500_s500_t120 strategy. Renamed model to models with a historical compatibility symlink, documented all flags, ran the ten-seed example at 1,000 raw proposals per seed, retained 1,305 unique non-seed candidates, reproduced one archived Step-2d 1,000-proposal stream exactly, and passed artifact, output, frozen-contract, focused, and 65-test repository validation.

## Outcome

- Signal: useful

## Source Nodes

- gmolai.py
- run_encode()
- run_decode()
- ConditionalSmilesTransformer
- generate_embeddings.py
- Frozen generation baseline v1
- canonicalize()