# Step 2b results

## Outcome

**GO.** The frozen development-selected policy was `beam64_lp00`. Final
evaluation used 10,000 fresh validation molecules
per control, with zero overlap with the original Step-2 generation panel.

At k=50, the correct molecule was present for **93.92%** of correct
conditions (oracle Recall@50). Target-blind frozen-gMolAI reranking gave
**93.92%** exact identity@1, versus **63.93%** for greedy decoding on
the same fresh panel: an absolute gain of
**29.99%**.
Relative to the historical Step-2 63.90%, the gain is
**30.02%**.
When the target was present, latent reranking selected it with
**100.00%** efficiency.

The reranked correct-condition top-1 has
99.99% validity,
95.75% scaffold
recovery, and mean Morgan similarity
0.9743.

## Condition-use controls

- Shuffled conditions: source identity@1
  93.92%;
  original-target identity@1
  0.00%.
- Nearest-wrong conditions: source identity@1
  95.13%;
  original-target identity@1
  0.01%.

These candidates and their ranking therefore continue to follow the supplied
condition rather than the original target.

## Interpretation

The registered error classification is
`primarily_search_related_at_the_tested_candidate_budget`. Oracle coverage and deployable reranking are kept
separate throughout; no oracle target quantity was used to order candidates.

The frozen gate table is machine-readable in `outputs/decision.json`. The
decision on proceeding to MMP-perturbed decoding is **GO**. No MMP
perturbation or derivative generation was performed here.
