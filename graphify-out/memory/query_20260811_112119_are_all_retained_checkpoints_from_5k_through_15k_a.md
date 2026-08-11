---
type: "artifact_availability"
date: "2026-08-11T11:21:19.302498+00:00"
question: "Are all retained checkpoints from 5k through 15k available?"
contributor: "graphify"
outcome: "useful"
---

# Q: Are all retained checkpoints from 5k through 15k available?

## Answer

Yes. Both production runs, seed 42 and seed 43, contain step-000005000.pt, step-000007500.pt, step-000010000.pt, step-000012500.pt, and step-000015000.pt. Every file is 26,796,827 bytes, is a readable ZIP-format PyTorch checkpoint with 306 archive members, passes archive CRC testing with no bad member, and has a distinct SHA-256 digest. The checkpoint weights are available; this does not imply that every checkpoint has already undergone the same downstream promotion evaluation.

## Outcome

- Signal: useful