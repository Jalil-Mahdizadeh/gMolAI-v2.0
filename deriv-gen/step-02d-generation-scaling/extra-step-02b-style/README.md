# Extra Step 2b-style analysis of the Step 2d final library

This self-contained additive study applies the correct-condition search,
coverage, structural-fidelity, and frozen-latent reranking analyses from Step 2b
to the completed Step 2d final candidate streams at raw budgets 50, 100, 250,
500, and 1,000.

The original Step 2d directory is an immutable input. Runtime isolation mounts
the repository read-only and this folder read-write, so all generated inputs,
embeddings, tables, plots, logs, caches, state, and hashes stay here.

Run from the repository root on one GH200 GPU:

```bash
bash deriv-gen/step-02d-generation-scaling/extra-step-02b-style/run_analysis.sh
```

The frozen encoder invocation uses `--batch-size 512 --workers 48` and records
the effective settings in the embedding metadata. See `PROTOCOL.md` for exact
definitions. Completed results are written to `RESULTS.md` and `outputs/`.
