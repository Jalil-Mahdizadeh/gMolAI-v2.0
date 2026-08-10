#!/usr/bin/env bash
#SBATCH --job-name=gmolai-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "${CODE_DIR}/slurm/common.sh"
: "${GMOLAI_CHECKPOINT_NAME:?Set the immutable selected checkpoint, for example checkpoints/step-000010000.pt}"
checkpoint_name="${GMOLAI_CHECKPOINT_NAME}"
evaluation_arguments=(
  -m gmolai_retrain.cli --config "${CONFIG_PATH}" evaluate
  --checkpoint "${checkpoint_name}" --split test
)
if [[ -n "${GMOLAI_RUN_DIR:-}" ]]; then
  evaluation_arguments+=(--run-dir "${GMOLAI_RUN_DIR}")
fi
container_gpu torchrun --standalone --nproc_per_node=4 "${evaluation_arguments[@]}"
