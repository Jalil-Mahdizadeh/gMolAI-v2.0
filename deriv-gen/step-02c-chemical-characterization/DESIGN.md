# Step 2c design and denominator notes

The primary unit is a fresh-validation seed (`n = 10,000`). The candidate-level
table has one row per retained Step-2b molecular identity, not one row per raw
beam hypothesis. A seed with no accepted candidate remains in the seed table
with zero counts.

The audit uses target chemistry only for post-generation characterization. No
candidate is generated or ranked here. MMP relations are assessed pairwise
between each seed and its retained candidates. Candidate-set diversity uses all
pairwise Morgan similarities within each seed; raw values plus seed offsets are
saved so global quantiles are reproducible.

Bemis-Murcko equality is meaningful only when the seed scaffold is non-empty.
For transparency, tables also include exact scaffold-key equality and a
separate both-acyclic category. Non-MMP graph changes are descriptor deltas;
without an atom mapping they are not presented as a unique edit path.
