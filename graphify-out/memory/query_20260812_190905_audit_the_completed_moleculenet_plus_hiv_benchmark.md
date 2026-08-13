---
type: "query"
date: "2026-08-12T19:09:05.742030+00:00"
question: "Audit the completed MoleculeNet plus HIV benchmark implementation and artifact flow"
contributor: "graphify"
outcome: "useful"
source_nodes: ["benchmark_moleculenet()", "_scaffold_splits()", "_split_identity_manifest()", "benchmark_io.py", "assemble_common.py", "finalize.py"]
---

# Q: Audit the completed MoleculeNet plus HIV benchmark implementation and artifact flow

## Answer

Expanded from the repository graph vocabulary via [benchmark, moleculenet, downstream, evaluation, scaffold, split, embeddings, coverage, protocol, finalize]. The existing graph links the repository's benchmark_moleculenet, _scaffold_splits, and _split_identity_manifest implementation to the benchmark-side benchmark_io.py, assemble_common.py, and finalize.py audit pattern. Direct artifact verification then confirmed that phase two reconstructs the repository preparation/split semantics, preserves inherited identity roles, validates common embeddings, and checksum-seals results.

## Outcome

- Signal: useful

## Source Nodes

- benchmark_moleculenet()
- _scaffold_splits()
- _split_identity_manifest()
- benchmark_io.py
- assemble_common.py
- finalize.py