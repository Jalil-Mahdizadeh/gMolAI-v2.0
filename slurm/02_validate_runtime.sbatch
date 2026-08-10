#!/usr/bin/env bash
#SBATCH --job-name=gmolai-validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "${CODE_DIR}/slurm/common.sh"

container_gpu python -c \
  "import torch,torch_geometric,pyarrow,rdkit,duckdb,pandas,scipy,sklearn; print(torch.cuda.is_available(), torch.__version__, torch_geometric.__version__, pyarrow.__version__, rdkit.__version__, duckdb.__version__, pandas.__version__, scipy.__version__, sklearn.__version__)"
container_gpu python -m pytest -q tests
container_gpu torchrun --standalone --nproc_per_node=2 tests/ddp_smoke.py
config_arguments=(-m gmolai_retrain.cli --config "${CONFIG_PATH}" check-config)
if [[ -n "${GMOLAI_PLAN_PATH:-}" ]]; then
  config_arguments+=(--plan "${GMOLAI_PLAN_PATH}")
fi
container_gpu python "${config_arguments[@]}"
