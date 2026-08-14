# Decoder decision

## NO-GO

The decoder learned clear condition dependence, but it did not meet every predeclared fidelity criterion for a faithful inverse.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Final validation generation panel size | 10000.0000 | >= 10000.0000 | yes |
| Correct-condition valid-SMILES rate | 0.9752 | >= 0.9500 | yes |
| Correct-condition molecular identity recovery | 0.6390 | >= 0.8000 | no |
| Correct-condition scaffold recovery | 0.7470 | >= 0.9000 | no |
| Correct-condition all-row mean Morgan similarity | 0.8495 | >= 0.9000 | no |
| Median re-encoded cosine to correct condition | 1.0000 | >= 0.9500 | yes |
| Correct target-identity advantage over best wrong control | 0.6383 | >= 0.5000 | yes |
| Nearest-wrong supplied-source identity recovery | 0.6612 | >= 0.6000 | yes |
| Shuffled supplied-source identity recovery | 0.6392 | >= 0.6000 | yes |

The rule in `PROTOCOL.md` was frozen before final evaluation. Correct embeddings must reconstruct their targets strongly, while shuffled and nearest-wrong embeddings must redirect output toward the molecule supplying the condition. This separates a conditional inverse from an unconditional SMILES language model.

The NO-GO applies only to condition-dependent, zero-perturbation inversion of `released_hybrid_w3` on held-out validation molecules. No latent perturbation or derivative generation is included.
