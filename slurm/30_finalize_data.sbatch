#!/usr/bin/env bash
#SBATCH --job-name=gmolai-finalize-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

source "${CODE_DIR}/slurm/common.sh"
container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" finalize-data
container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" fit-scaler
