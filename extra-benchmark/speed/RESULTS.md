# Encoding-speed benchmark results

- **Panel:** 49,844 all-model common locked-test molecules
- **Host:** n54
- **GPU:** NVIDIA GH200 120GB (neural encoders only)
- **Slurm job:** 1230738
- **CPU allowance:** 48 CPUs; gMolAI uses 48 RDKit workers
- **Measurement:** one complete pass after one warm-up batch; no confidence intervals

| Encoder | Device | Batch 64 (mol/s) | Batch 128 (mol/s) | Batch 256 (mol/s) | Batch 512 (mol/s) | Output-equivalent? |
|---|---|---:|---:|---:|---:|:---:|
| gMolAI optimized | GPU | 13,040.69 | 22,901.45 | 40,068.75 | 58,330.38 | Yes |
| Morgan radius-2 | CPU | 10,736.06 | 10,781.57 | 10,885.82 | 10,889.23 | Yes |
| MolAI epoch 6 | GPU | 3,630.30 | 4,621.65 | 5,161.02 | 5,457.10 | Yes |
| MoLFormer | GPU | 2,162.66 | 3,645.33 | 5,101.70 | 5,808.19 | Yes |
| SMI-TED-Light | GPU | 577.22 | 1,005.90 | 1,530.65 | 2,118.45 | Yes |
| MolCLR-GIN | GPU | 1,455.41 | 1,522.37 | 1,602.64 | 1,693.53 | Yes |
| KERMT v2† | GPU | 315.87 | 340.77 | 373.03 | 392.63 | No |

The timer covers canonical SMILES already in RAM through complete ordered FP32 vectors in host RAM. It includes each model's required preprocessing and device transfers but excludes SIF/model loading, worker startup, warm-up, validation, hashing and disk serialization.

† Native KERMT v2 failed the frozen cross-batch output-equivalence gate (minimum cosine 0.982395; maximum relative-L2 delta 0.197321). Its batch-size points describe computational throughput for batch-dependent outputs, not scaling of one invariant molecular representation.

These are descriptive single-pass point measurements. Small differences must not be presented as established speed superiority, and Morgan's CPU result must not be described as GPU-forward performance.
