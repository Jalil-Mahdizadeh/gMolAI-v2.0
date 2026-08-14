# Frozen protocol: scaled MMP mining and latent control-space selection

## Scientific boundary

This is an inference-only retrieval study. It must not retrain or modify gMolAI,
its checkpoint, calibrator, inference definition, or released 384-dimensional
vector. It does not train a decoder and does not claim novel-molecule generation.

The promoted seed-42 step-10,000 checkpoint and its immutable train-only
coordinate calibrator are read-only inputs. The locked test partition and all
endpoint labels are forbidden.

## Populations and separation

- Fit/mining population: a deterministic stratified sample of 1,000,000
  pretraining-**train** molecules, exported as raw graph-256 plus mean-node-128
  blocks with sampling seed 1,618,033.
- Validation population: the exact independent 50,000 pretraining-validation
  molecules used in Day 1, with the same identities and retrieval bank.
- MMP transformation directions are fitted only on train molecules.
- A validation pair is eligible only when its exact retained core was absent
  from every train observation used for that transformation.
- Train/validation molecule hashes must be disjoint.

## Candidate spaces

The immutable calibrator is applied once to raw blocks. No calibrator is refit.
The five Euclidean coordinate spaces are:

1. standardized graph block, `graph_256`;
2. standardized mean-node block, `mean_node_128`;
3. `[graph_std, mean_node_std]`, `hybrid_w1`;
4. `[graph_std, 3 * mean_node_std]`, `released_hybrid_w3`;
5. `[graph_std, 6 * mean_node_std]`, `hybrid_w6`.

Weights 1 and 6 are diagnostic metrics only. Weight 3 remains the released
gMolAI representation throughout.

## Scalable matched-molecular-pair mining

RDKit performs one-cut fragmentation with the Day-1 chemistry filters: retained
core at least six heavy atoms, substituent one to ten heavy atoms, retained core
at least 60% of the parent, substituent heavy-atom difference at most six, and
parent heavy-atom difference at most eight.

For scalability and statistical independence, fragments are reduced to one
deterministic molecule representative per `(core, substituent)` group. Mining
then records:

- eligible molecule-pair multiplicity before representative reduction;
- one independent `(core, transformation)` observation for direction fitting;
- distinct transformations and support at at least 2, 5, 10, and 20 train cores;
- transformations at each tier with unseen-core validation support.

No common-core group is silently discarded and support counts are calculated
before query caps. One core contributes at most one direction observation to a
given transformation, preventing pseudo-replication from stereoisomers or
duplicate fragment realizations.

## Direction and null controls

For every space and transformation, the train direction is the normalized mean
of unit displacement vectors across independent cores. Step length is the median
train displacement norm. Validation alignment is the cosine between the held-out
pair displacement and its fitted train direction.

Each held-out pair receives one deterministic mismatched-transformation null
drawn from the same train-support tier. The null transformation assignment,
validation identities, query selection, and random seeds are identical across
all five spaces.

Retrieval methods are:

- unperturbed seed nearest neighbor;
- isotropic random direction, three replicates;
- global-covariance direction, three replicates;
- local-covariance direction, three replicates;
- support-matched mismatched MMP direction;
- fitted MMP direction;
- oracle interpolation at 0.25, 0.50, 0.75, and 1.00, clearly labeled as a
  geometry diagnostic rather than a deployable method.

The seed is excluded from all searches against the common 50,000-molecule bank.

## Metrics

For every space and transformation, report:

- unseen-core alignment and mismatched-direction null;
- exact derivative recall@1, recall@10, and reciprocal rank within 50;
- top-1 scaffold retention;
- generic one-cut MMP relation to the seed;
- exact requested `(core, target substituent)` match;
- mean seed-to-retrieved Morgan Tanimoto.

Pair-weighted results are secondary. Primary summaries are unweighted macro
averages across transformations. Random replicates are averaged within a query
before transformation aggregation.

## Statistical analysis

Primary selection uses transformations supported by at least five independent
train cores. The at-least-ten and at-least-twenty cohorts are confirmatory when
they contain enough transformations with validation support.

The fixed retrieval panel is selected deterministically, prioritizes the
at-least-five-core cohort, is round-robin across transformations, and includes
at most 32 unseen-core queries per transformation (2,048 total). A primary
selection claim requires at least 10 transformations and 100 queries; otherwise
the outcome is explicitly labeled underpowered.

Uncertainty uses a paired hierarchical bootstrap with 2,000 replicates:
transformations are sampled with replacement, then held-out pairs are sampled
with replacement inside each sampled transformation. The same bootstrap draws
are used for every space. Tables include macro estimates, 95% percentile CIs,
paired differences from `released_hybrid_w3`, and performance versus train-core
support.

## Frozen control-space decision rule

A candidate is viable when, in the at-least-five-core cohort:

1. the 95% hierarchical-bootstrap CI lower bound for macro alignment gain over
   the mismatched null is greater than zero; and
2. at least 70% of transformations have positive mean alignment gain.

Among viable spaces, identify the Pareto set for macro recall@1, alignment gain,
exact requested-transform match, and fraction of transformations with positive
alignment gain. Prefer `released_hybrid_w3` when it is non-inferior to the best
Pareto candidate within all three fixed margins:

- recall@1: 0.02 absolute;
- alignment gain: 0.05 absolute;
- exact requested-transform match: 0.02 absolute.

If released weight 3 is inferior beyond a margin, choose the viable candidate
with highest macro recall@1; break differences within 0.02 by alignment gain,
then exact-transform match. The resulting choice is the **edit-control space**.
The decoder-conditioning representation remains the unchanged released ×3
hybrid in all cases.

## Required decisions

The report must state:

1. whether released ×3 improves over unweighted ×1;
2. whether mean-node-128 remains the strongest molecular-edit space;
3. whether directional transfer survives at 1M scale and at least 5–10 train
   cores;
4. which edit-control space should be frozen for decoder development and why.

