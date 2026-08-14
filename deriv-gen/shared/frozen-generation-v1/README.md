# Frozen generation baseline v1

This contract permanently pins the existing Step-2 decoder and the selected
Step-2d `hybrid_b500_s500_t120` sampling strategy as the derivative-generation
baseline. It is an operational handoff for a possible future study, not a new
experiment, result, or authorization to regenerate candidates.

## What is frozen

The canonical runtime decoder is
`step-02-decoder-feasibility/checkpoints/best.pt` (SHA-256 `bb962308...`). The
compact inference-only export is `checkpoints/decoder_inference.pt` (SHA-256
`8b4f8db...`). Both are bound byte-for-byte in [`contract.json`](contract.json),
along with the training/export seals and decoder implementation.

The generation policy remains `hybrid_b500_s500_t120`: 500 ordered beam
hypotheses and 500 sample-stream hypotheses are deterministically interleaved
to produce 1,000 raw proposal slots per seed. The sample stream begins with one
greedy proposal and then uses fixed-order draws from the temperature-1.2,
top-p-0.995 pool. Beam width, length penalty, maximum output length, RNG seed
derivation, ordering rules, implementation files, selection seal, decision,
and verification are all hash-bound by the contract.

The large `.pt` artifacts remain outside Git under the repository artifact
policy. Their exact local bytes are nevertheless part of the versioned
contract through SHA-256 and byte-size bindings.

## Change control

Do not retrain, overwrite, retune, or silently reinterpret this baseline. A
different decoder, temperature, top-p threshold, beam/sample allocation,
ordering rule, seed rule, output ceiling, or implementation is a different
baseline and requires a new versioned contract plus an explicit superseding
scientific decision. Completed Step-2 and Step-2d records remain unchanged.

## Verification

From the repository root, run:

```bash
python deriv-gen/shared/frozen-generation-v1/verify.py
```

The verifier is read-only. It hashes the two model artifacts and every bound
source/evidence file, then cross-checks the semantic fields against the
original Step-2 and Step-2d seals.
