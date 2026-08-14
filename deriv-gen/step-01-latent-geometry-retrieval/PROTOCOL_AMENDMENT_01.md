# Protocol amendment 01: repeated-transformation support

Date: 2026-08-13
Status: adopted before inspecting any latent-geometry or retrieval outcome

## Trigger

The first deterministic MMP enumeration produced 14,343 training pairs across
14,306 exact transformations and 40,355 validation pairs. With the initially
planned minimum of five distinct training cores, only one transformation was
eligible (six training cores; 216 unseen-core validation pairs). The analysis
therefore stopped at its predeclared sufficiency check before retrieval results
were produced.

Support enumeration showed a discrete distribution: 214 shared transformations
had one training core, 20 had two, none had three through five, and one had six.
The exact-fragment transformation vocabulary is consequently much sparser than
anticipated for the fixed 100,000-molecule training export.

## Amendment

The minimum is reduced from five to two distinct training cores per exact
transformation. All other MMP filters, train/validation separation, unseen-core
holdout, query caps, controls, random seeds, feasibility gates, and statistical
procedures remain unchanged. This admits 33 repeated training transformations, of which 21
have unseen-core validation support, and 685 unseen-core validation pairs before
per-transformation query capping.

Directions supported by only two training cores are explicitly exploratory and
must not be described as robust medicinal-chemistry rules. Per-transformation
support and leave-one-core-out alignment remain in the outputs so the limitation
is auditable.

## Audit

Pre-amendment config SHA-256:
6bd896ea815c3ee02390e6dca82342d711e9aab5d1395a39cc22561c4005953d

Pre-amendment protocol SHA-256:
238870343f762f0b3c4f4d69a6e18d5537d9fc2bad072b98b668ccc7c165fe44

The threshold decision used only MMP support counts. No latent outcome or
retrieval metric was inspected to choose the revised value.
