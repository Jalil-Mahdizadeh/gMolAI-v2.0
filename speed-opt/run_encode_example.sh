#!/usr/bin/env bash
set -Eeuo pipefail

repository_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain"
project_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI"
sif="${project_root}/containers/gmolai-pyg-25.09-arm64.sif"
cd "${repository_root}"

apptainer exec --cleanenv --nv \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
  --bind "${project_root}:${project_root}" \
  --pwd "${repository_root}" \
  "${sif}" \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python speed-opt/scripts/encode_smiles.py "$@"
