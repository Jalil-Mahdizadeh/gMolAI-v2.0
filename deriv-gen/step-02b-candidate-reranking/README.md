# Step 2b: candidate search plus frozen-latent reranking

Status: complete. Decision: **GO**.

The frozen Step-2 decoder was evaluated with the development-selected
`beam64_lp00` candidate policy. Final evaluation used a fresh deterministic
10,000-molecule validation panel disjoint from the original Step-2 generation
panel. Candidates were filtered by the unchanged chemistry policy, re-encoded
by frozen released gMolAI, and ranked only by supplied-condition latent
consistency.

Key outputs:

- `RESULTS.md`: concise results and scientific interpretation.
- `DECISION.md`: GO/NO-GO decision.
- `outputs/decision.json`: frozen gates and exact machine-readable values.
- `outputs/tables/`: development selection and final metrics by control/k.
- `outputs/raw/`: candidate-level and query-level reproducibility tables.
- `outputs/figures/`: presentation figures (not used for ranking).

No model was trained or modified, and no derivative generation was performed.
