#!/usr/bin/env bash
#SBATCH --job-name=gmolai-s2d
#SBATCH --account=naiss2025-3-10-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=deriv-gen/step-02d-generation-scaling/logs/slurm-%j.out
#SBATCH --error=deriv-gen/step-02d-generation-scaling/logs/slurm-%j.err

set -euo pipefail
repo_root=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain
cd "${repo_root}"
exec bash deriv-gen/step-02d-generation-scaling/scripts/run_study.sh
