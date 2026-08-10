#!/usr/bin/env bash
#SBATCH --job-name=gmolai-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

source "${CODE_DIR}/slurm/common.sh"

if [[ -n "${GMOLAI_RUN_DIR:-}" ]]; then
  [[ "${GMOLAI_RUN_DIR}" = /* ]] || { echo "GMOLAI_RUN_DIR must be absolute" >&2; exit 2; }
  RUN_DIR="${GMOLAI_RUN_DIR}"
else
  RUN_DIR=$(container_cpu python -c "from gmolai_retrain.config import load_config; print(load_config('${CONFIG_PATH}')['paths']['run_dir'])" | tail -n 1)
fi
mkdir -p "${RUN_DIR}"

request_checkpoint() {
  echo "Slurm wall-time signal received; requesting an atomic checkpoint" >&2
  touch "${RUN_DIR}/REQUEST_CHECKPOINT"
}
trap request_checkpoint USR1

set +e
training_arguments=(
  -m gmolai_retrain.cli --config "${CONFIG_PATH}" train
)
if [[ -n "${GMOLAI_PLAN_PATH:-}" ]]; then
  [[ "${GMOLAI_PLAN_PATH}" = /* ]] || { echo "GMOLAI_PLAN_PATH must be absolute" >&2; exit 2; }
  [[ -f "${GMOLAI_PLAN_PATH}" ]] || { echo "Missing GMOLAI_PLAN_PATH: ${GMOLAI_PLAN_PATH}" >&2; exit 2; }
  training_arguments+=(--plan "${GMOLAI_PLAN_PATH}")
fi
if [[ -n "${GMOLAI_RUN_DIR:-}" ]]; then
  training_arguments+=(--run-dir "${GMOLAI_RUN_DIR}")
fi
if [[ -n "${GMOLAI_NODE_BUDGET_PER_GPU:-}" ]]; then
  training_arguments+=(--node-budget "${GMOLAI_NODE_BUDGET_PER_GPU}")
fi
if [[ -n "${GMOLAI_GRAPH_BUDGET_PER_GPU:-}" ]]; then
  training_arguments+=(--graph-budget "${GMOLAI_GRAPH_BUDGET_PER_GPU}")
fi
container_gpu torchrun --standalone --nproc_per_node=4 "${training_arguments[@]}" &
training_pid=$!
while kill -0 "${training_pid}" 2>/dev/null; do
  wait "${training_pid}"
  status=$?
  if kill -0 "${training_pid}" 2>/dev/null; then
    continue
  fi
  break
done
set -e

if [[ ${status:-1} -eq 99 ]]; then
  echo "Checkpoint complete; requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit "${status:-1}"
