# Design rationale

Beam search and stochastic sampling test complementary failure modes. A broad
beam asks whether the correct sequence has high decoder probability but loses
the greedy local decision. Fixed-seed top-p sampling asks whether it lies in a
wider plausible region that a probability-ranked beam misses. Both remain
strictly inference-only.

The deployable reranker uses released-gMolAI relative L2 to the supplied
condition. For a fixed query this has exactly the same ordering as L2, but the
reported value is normalized across molecules. Cosine is only a tie-breaker.
No target-derived structural quantity enters candidate generation, filtering,
ordering, or reranking.

The exact target may be used after ranking to calculate oracle Recall@k and
top-1 identity. Oracle recall diagnoses candidate coverage; reranked identity
diagnoses whether the frozen representation can identify a candidate once it
is available. Their gap separates generator search coverage from latent
selection failure.

The final panel excludes all original Step-2 generation molecules. Shuffled
and nearest-wrong controls are evaluated against both the original molecule
and the molecule supplying the condition. A true conditional inverse should
follow the latter under these controls.
