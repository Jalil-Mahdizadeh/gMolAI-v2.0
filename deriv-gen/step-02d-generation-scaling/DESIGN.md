# Step 2d design notes

- Development and final panels are chemically and procedurally separate.
- Strategy selection uses no final-validation result.
- All budgets share seeds and are literal prefixes of one frozen proposal stream.
- Raw proposal tables preserve invalid/rejected strings and duplicates.
- Candidate tables retain the first occurrence of each accepted canonical
  identity per seed/strategy.
- Useful-local chemistry is the union of one-cut MMP and non-empty
  same-scaffold relations; it is a descriptive definition, not a property claim.
- Novelty is defined solely against the 980,000 decoder-fit molecules.
- Pairwise diversity sampling is bounded and deterministic to avoid an
  infeasible billions-of-pairs calculation at budget 1,000.
