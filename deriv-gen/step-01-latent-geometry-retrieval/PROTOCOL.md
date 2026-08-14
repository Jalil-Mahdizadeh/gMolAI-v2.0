# Final protocol: latent geometry and derivative retrieval

## Question

Does the promoted gMolAI representation contain sufficiently organized local
and directional chemical information to justify a later derivative generator?

This is a retrieval feasibility study. Retrieval supplies only known valid
molecules and therefore isolates embedding geometry from decoder behavior.

## Data separation

- Geometry, covariance, and MMP transformation directions are fitted only on a
  deterministic 100,000-molecule **pretraining-train** export.
- Queries and the candidate bank come from an independent deterministic
  50,000-molecule **pretraining-validation** export.
- Validation MMP queries must use an exact fragmentation core absent from the
  training examples for that transformation.
- The locked internal test partition is not read.
- No endpoint labels are used. The 13 standardized molecular descriptors stored
  with the representation probes are used only to quantify chemical movement.

## Coordinate spaces

The train payload contains raw `[graph_z, mean_node_z]` blocks. It is transformed
with the promoted train-only coordinate mean and population standard deviation.
The validation payload contains the released weighted vector
`[graph_std, 3 * mean_node_std]`; the factor of three is removed before analysis.

Three spaces are evaluated independently:

1. graph-256;
2. mean-node-128;
3. unweighted standardized hybrid-384.

Euclidean distance is used because the coordinates are train-standardized and
the perturbations are additive. Results are not described as decoding.

## Single-cut matched molecular pairs

RDKit MMP fragmentation uses one eligible cut. The larger fragment is treated
as the retained core. Conservative filters require a core of at least six heavy
atoms, a variable fragment of one to ten heavy atoms, at least 60% of parent
heavy atoms retained in the core, and bounded parent/substituent size changes.
Promiscuous core groups and per-core pair counts are capped deterministically.

A transformation is oriented by the canonical lexical order of its two
attachment-bearing substituent fragments. Following the pre-outcome support amendment documented in
`PROTOCOL_AMENDMENT_01.md`, it is eligible only when observed on at least two
distinct training cores. The trained displacement is the unit
vector of the mean normalized displacement, scaled by the median train-pair
distance for that transformation.

## Prospective support amendment

The initial five-core threshold admitted only one exact transformation. Before
inspecting any latent outcome or retrieval metric, it was reduced to two cores,
which admits 33 repeated train transformations; 21 have unseen-core validation
support, preserving 685 validation pairs before query capping. All other settings and feasibility gates are unchanged.
Directions with two-core support are exploratory; see
`PROTOCOL_AMENDMENT_01.md` for the support distribution and prior hashes.

## Analyses

1. **Global spectrum:** effective rank, participation ratio, and cumulative
   covariance spectrum in all three spaces.
2. **Local geometry:** train-neighbor distances and local covariance spectra for
   a scaffold-diverse validation panel.
3. **Distance-to-chemistry:** Morgan Tanimoto, scaffold retention, one-cut MMP
   relation, heavy-atom change, and descriptor RMS versus neighbor rank/distance,
   with random-pair controls.
4. **MMP consistency:** leave-one-example-out train alignment and unseen-core
   validation alignment against mismatched-transformation null directions.
5. **Derivative retrieval:** exact known validation derivative retrieval plus
   seed-related chemistry metrics for the methods below.

## Perturbation and retrieval controls

Blind methods use the transformation-specific median train MMP step length:

- isotropic Gaussian direction;
- global covariance-shaped direction;
- local covariance-shaped direction from 64 train neighbors;
- learned train-MMP direction.

`seed_nn` is the ordinary nearest-neighbor baseline. Oracle interpolation at
fractions 0.25, 0.50, 0.75, and 1.00 is explicitly labeled a geometry diagnostic,
not a deployable generation method. It tests whether movement along a real
seed-to-derivative segment behaves monotonically.

The validation seed itself is excluded from every candidate search. Exact target
recall is measured at 1, 10, and 50 among all 50,000 validation candidates.
Generic derivative quality is measured independently with seed Tanimoto,
scaffold retention, one-cut MMP relation, heavy-atom change, and descriptor
movement. Random methods use three deterministic replicates.

## Predeclared feasibility gates

- **G1 — local chemical organization:** in hybrid-384, nearest-neighbor mean
  Morgan Tanimoto is at least 1.5 times the random-pair mean and nearest-neighbor
  scaffold retention exceeds random by at least 0.02 absolute.
- **G2 — transferable MMP direction:** hybrid-384 median unseen-core validation
  cosine alignment exceeds the mismatched-direction null by at least 0.10 and is
  positive.
- **G3 — controlled retrieval:** hybrid-384 learned-MMP recall@10 exceeds the
  best blind perturbation by at least 0.02 absolute and by at least twofold when
  the blind recall is nonzero.
- **G4 — interpolation sanity:** hybrid-384 target recall@10 is nondecreasing at
  interpolation fractions 0.25, 0.50, 0.75, and 1.00, and is at least 0.99 at
  fraction 1.00.

Passing G1 alone supports chemically enriched retrieval. Passing G2 and G3
supports matched-pair displacement as the control signal for a later
seed-conditioned editor. Failure is retained and reported; thresholds are not
changed after seeing results.

## Statistical treatment

Aggregate retrieval intervals are deterministic 1,000-resample, query-level
bootstrap percentile intervals. MMP alignment is summarized across held-out
pairs and transformations. Point estimates, raw per-query records, negative
results, and empty-result cases are retained.

