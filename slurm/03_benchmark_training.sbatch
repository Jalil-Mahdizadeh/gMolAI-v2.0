#!/usr/bin/env bash
#SBATCH --job-name=gmolai-benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "${CODE_DIR}/slurm/common.sh"

BENCHMARK_RUN_DIR="${CODE_DIR}/runs/benchmark-v4-${SLURM_JOB_ID}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-400}"
BENCHMARK_NODE_BUDGET="${BENCHMARK_NODE_BUDGET:-32768}"
BENCHMARK_GRAPH_BUDGET="${BENCHMARK_GRAPH_BUDGET:-2048}"
echo "benchmark_run_dir=${BENCHMARK_RUN_DIR}"
echo "gpu_dmon=${BENCHMARK_RUN_DIR}/gpu-dmon.txt"
echo "node_budget_per_gpu=${BENCHMARK_NODE_BUDGET} graph_budget_per_gpu=${BENCHMARK_GRAPH_BUDGET}"
container_gpu bash slurm/run_benchmark_in_container.sh \
  "${BENCHMARK_RUN_DIR}" "${CONFIG_PATH}" "${BENCHMARK_STEPS}" \
  "${BENCHMARK_NODE_BUDGET}" "${BENCHMARK_GRAPH_BUDGET}"
