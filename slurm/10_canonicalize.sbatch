#!/usr/bin/env bash
#SBATCH --job-name=gmolai-canonicalize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out

source "${CODE_DIR}/slurm/common.sh"
container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" canonicalize --task "${SLURM_ARRAY_TASK_ID}"
