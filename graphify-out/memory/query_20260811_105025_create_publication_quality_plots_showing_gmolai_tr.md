---
type: "codebase"
date: "2026-08-11T10:50:25.332778+00:00"
question: "Create publication-quality plots showing gMolAI training and validation progress for gmolai-rev1.docx."
contributor: "graphify"
outcome: "useful"
source_nodes: ["runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/metrics.jsonl", "runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050-seed43/metrics.jsonl", "configs/representation-pilot-mean-node-contrastive-001-desc050.yaml", "src/gmolai_retrain/train.py", "REVIEW_ARTIFACTS.md"]
---

# Q: Create publication-quality plots showing gMolAI training and validation progress for gmolai-rev1.docx.

## Answer

Created two publication-ready composite figures in ../manuscript/figures. Fig1 shows two-seed training loss, 50k-graph validation loss, reconstruction metrics, and latent effective rank from the seed42/seed43 metrics.jsonl logs. Fig2 shows the paired 10-split FreeSolv checkpoint screen at 5k, 7.5k, 10k, 12.5k, and 15k plus direction-aligned 10k-to-15k metric changes, documenting that the 10k checkpoint met the 1.30 RMSE promotion gate while later pretraining metrics continued improving. Exports: PDF/SVG and 600-dpi PNG/TIFF; source tables, generator, captions, hashes, and provenance manifest included. Independent audit matched 750 training records per seed, all 12 validation records, all 50 FreeSolv split values, and every derived change value.

## Outcome

- Signal: useful

## Source Nodes

- runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050/metrics.jsonl
- runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050-seed43/metrics.jsonl
- configs/representation-pilot-mean-node-contrastive-001-desc050.yaml
- src/gmolai_retrain/train.py
- REVIEW_ARTIFACTS.md