---
type: "query"
date: "2026-08-11T21:57:25.846831+00:00"
question: "Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k."
contributor: "graphify"
outcome: "useful"
source_nodes: ["audit_downstream_checkpoint_exposure()", "_join_pretraining_rows()", "_scan_target_locations()", "_checkpoint_records()", "_rank_exposure()", "_seen_at_cycle_zero_cursor()", "command_audit_downstream_exposure()", "Manuscript rev4: exact downstream-molecule exposure audit"]
---

# Q: Perform a no-training audit of actual downstream-molecule exposure for every retained seed-42 checkpoint from 5k through 15k.

## Answer

Expanded from the repository vocabulary via [audit, checkpoint, cursor, shard, rank, identity, downstream, exposure, manuscript, artifact, overlap, dataset]. The exact audit is rooted at audit_downstream_checkpoint_exposure(), which prepares and collision-checks downstream canonical identities through _join_pretraining_rows(), resolves each training overlap to an exact graph location with _scan_target_locations(), restores all rank cursors with _checkpoint_records() and _rank_exposure(), and counts only locations satisfying the strict pre-cursor boundary in _seen_at_cycle_zero_cursor(). command_audit_downstream_exposure() exposes the workflow through the CLI, while the manuscript-rev4 evidence node documents its outputs and constraints.

## Outcome

- Signal: useful

## Source Nodes

- audit_downstream_checkpoint_exposure()
- _join_pretraining_rows()
- _scan_target_locations()
- _checkpoint_records()
- _rank_exposure()
- _seen_at_cycle_zero_cursor()
- command_audit_downstream_exposure()
- Manuscript rev4: exact downstream-molecule exposure audit