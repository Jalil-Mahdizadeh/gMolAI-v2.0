# Manuscript rev5: evidence-source reorganization

This directory contains the publication-formatted fifth manuscript revision.
It was generated directly from the authoritative revision 4 DOCX and is an
editorial/structural revision only. No training was started or resumed, no
model was executed, no embedding was regenerated, and no checkpoint,
calibrator, representation definition, promotion threshold or result was
changed.

## Experimental chronology made explicit

The manuscript now presents the study in the following order:

`corpus/split -> training -> candidate checkpoints -> checkpoint-specific calibration -> validation -> external development/promotion gate -> freeze seed-42/10k -> locked internal test -> HIV confirmation -> seed-43 replication`

Methods Table 6 gives each evidence source, its population and actual usage,
whether it can update neural-network weights, whether it can influence
promotion, and its scientific purpose. It explicitly distinguishes the
221,148,895-graph training partition, the 943,038-graph validation partition,
the five-dataset development/promotion panel, the 1,088,766-graph locked
internal test partition, external confirmatory HIV and seed-43 replication.

## Old-to-new section mapping

| Revision 4 | Revision 5 | Editorial action |
|---|---|---|
| 1.1 | 1.1 | Retitled and lightly rewritten to state the complete chronology and freeze point. |
| 1.2-1.6 | 1.2-1.6 | Structure retained; terminology and cross-references normalized. |
| 1.7 | 1.7 | Retitled and clarified as candidate checkpoints, promotion freeze, post-selection confirmation and replication. |
| none | 1.8 | Added concise data/evidence-roles overview and Table 6. |
| 1.8-1.10 | 1.9-1.11 | Shifted by one without changing their substantive scope. |
| 2.1.1 plus exposure material from 2.1.7 | 2.1.1 | Consolidated corpus scale, realized split counts and exact checkpoint exposure. |
| Validation-derived material from 2.1.1, 2.1.2 and 2.1.5-2.1.6 | 2.1.2 | Collected all selected-representation validation evidence and stated every partition/sample size. |
| 2.1.3 | 2.1.3 | Retained the complete 5k-15k frozen retrospective promotion trajectory. |
| 2.1.4 | 2.1.4 | Condensed the released 384-D representation description. |
| 2.1.7 | 2.1.5 | Consolidated the five external development/promotion benchmarks, descriptor control, corpus membership and exact checkpoint exposure. |
| 2.1.8 | 2.1.6 | Isolated independent seed-43 replication from selection evidence. |
| Locked-test material from 2.1.9 | 2.1.7 | Created a highly visible protected internal-test subsection and retained the Morgan clustering reversal. |
| HIV material from 2.1.9 | 2.1.8 | Moved HIV into a separate external post-selection endpoint subsection. |
| 2.2 and 3 | 2.2 and 3 | Structure retained; bounded interpretation and cross-references aligned with the new evidence roles. |

## Interpretation-affecting wording changes

- Validation may influence checkpoint selection but never neural-network
  weights.
- BACE, BBBP, ESOL, FreeSolv and Lipophilicity are consistently called the
  selection-conditioned development/promotion panel; their outer test folds
  are nested-split folds, not the locked internal test partition.
- The promotion criteria are described conservatively as frozen criteria
  applied uniformly in the complete retrospective 5k-15k audit, not as
  prospectively preregistered.
- Pretraining-partition membership is distinguished from actual presentation to
  the model before a checkpoint. The seed-42/10k checkpoint consumed exactly
  57,504,265 unique training graphs.
- The locked internal test was opened only after the seed-42/10k checkpoint,
  calibrator and 384-D representation were frozen; none of its results could
  alter promotion.
- HIV is a separate external post-selection confirmation and was never part of
  the internal corpus split. Seed 43 is replication, not model selection.
- The standard Euclidean K-means analysis on L2-normalized inputs is explicitly
  not described as spherical K-means.
- The locked-test scaffold-clustering reversal is retained: Morgan outperformed
  gMolAI there even though gMolAI was stronger on validation. The manuscript
  does not claim universal superiority over Morgan or molecule-level novelty
  for downstream molecules consumed during pretraining.

## Tables, figures and validation

The five original Methods tables remain Tables 1-5. The new evidence-roles
table is Table 6. Revision 4 Tables 8, 6, 7, 9, 10 and 11 become revision 5
Tables 7-12 in the order in which their evidence is interpreted. Existing
table cell matrices and Office Math equation hashes are checked before and
after serialization. In-text table references are sequential from 1 through
12; the training/validation progress and promotion-trajectory images are
referenced as Figures 1 and 2.

The release DOCX reopened successfully, passed ZIP integrity and terminology
checks, and rendered to a 29-page, US-letter PDF with no PDF suspects. Every
rendered page was visually inspected for tables, equations, captions, wrapping
and page breaks. The final release raster was byte-identical on all 29 pages to
the fully inspected validation render.

## Rebuild

Use the isolated `manuscript` dependency environment or another environment
with `python-docx>=1.2`:

```text
python scripts/update_manuscript_rev5.py \
  --input artifacts/manuscript-rev4/gmolai-rev4.docx \
  --output artifacts/manuscript-rev5/gmolai-rev5.docx
```

The tracked manuscript is byte-identical to
`../manuscript/gmolai-rev5.docx`; revision 4 remains unchanged.
