#!/usr/bin/env bash
#SBATCH --job-name=gmolai-verify-inputs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

source "${CODE_DIR}/slurm/common.sh"
container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" verify-environment
container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" verify-inputs
