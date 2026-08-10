#!/usr/bin/env bash
set -euo pipefail

run_dir=${1:?run directory required}
config_path=${2:?config path required}
steps=${3:?step count required}
node_budget=${4:?node budget required}
graph_budget=${5:?graph budget required}

mkdir -p "${run_dir}"
nvidia-smi dmon -d 1 >"${run_dir}/gpu-dmon.txt" &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

torchrun --standalone --nproc_per_node=4 \
  -m gmolai_retrain.cli --config "${config_path}" benchmark-training \
  --steps "${steps}" --run-dir "${run_dir}" \
  --node-budget "${node_budget}" --graph-budget "${graph_budget}"
