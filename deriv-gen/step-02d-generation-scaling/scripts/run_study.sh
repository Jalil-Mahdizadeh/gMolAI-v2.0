#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
step_root=$(cd -- "${script_dir}/.." && pwd)
deriv_root=$(cd -- "${step_root}/.." && pwd)
repo_root=$(cd -- "${deriv_root}/.." && pwd)
container=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif
container_root=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers
container_step=/repo/deriv-gen/step-02d-generation-scaling
state_root="${step_root}/state"

case "${step_root}" in
  "${repo_root}/deriv-gen/step-02d-generation-scaling") ;;
  *) echo "Refusing unexpected Step 2d root: ${step_root}" >&2; exit 2 ;;
esac
test -f "${container}"
for name in register_study.py prepare_panels.py test_components.py generate_candidates.py assemble_phase.py analyze_phase.py freeze_strategy.py report_results.py verify_study.py; do
  test -f "${script_dir}/${name}"
done

mkdir -p \
  "${step_root}/inputs" \
  "${step_root}/prepared" \
  "${step_root}/intermediate" \
  "${step_root}/outputs/raw/development" \
  "${step_root}/outputs/raw/final" \
  "${step_root}/outputs/tables" \
  "${step_root}/outputs/figures" \
  "${step_root}/logs" \
  "${state_root}/home" \
  "${state_root}/cache" \
  "${state_root}/matplotlib" \
  "${state_root}/pycache" \
  "${state_root}/tmp" \
  "${state_root}/apptainer-cache" \
  "${state_root}/apptainer-tmp" \
  "${state_root}/duckdb_tmp"

export APPTAINER_CACHEDIR="${state_root}/apptainer-cache"
export APPTAINER_TMPDIR="${state_root}/apptainer-tmp"

base_command=(
  apptainer exec
  --cleanenv
  --home "${state_root}/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${step_root}:${container_step}:rw"
  --bind "${container_root}:${container_root}:ro"
  --pwd /repo
  --env PYTHONPATH=${container_step}/scripts:/repo/src:/repo/deriv-gen/step-02-decoder-feasibility/scripts:/repo/deriv-gen/step-02b-candidate-reranking/scripts:/repo/deriv-gen/step-02c-chemical-characterization/scripts:/repo/deriv-gen/step-01b-scaled-space-selection/scripts
  --env XDG_CACHE_HOME=${container_step}/state/cache
  --env MPLCONFIGDIR=${container_step}/state/matplotlib
  --env PYTHONPYCACHEPREFIX=${container_step}/state/pycache
  --env TMPDIR=${container_step}/state/tmp
  --env OMP_NUM_THREADS=64
  "${container}"
)
gpu_command=(
  apptainer exec
  --nv
  --cleanenv
  --home "${state_root}/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${step_root}:${container_step}:rw"
  --bind "${container_root}:${container_root}:ro"
  --pwd /repo
  --env PYTHONPATH=${container_step}/scripts:/repo/src:/repo/deriv-gen/step-02-decoder-feasibility/scripts:/repo/deriv-gen/step-02b-candidate-reranking/scripts
  --env XDG_CACHE_HOME=${container_step}/state/cache
  --env PYTHONPYCACHEPREFIX=${container_step}/state/pycache
  --env TMPDIR=${container_step}/state/tmp
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env OMP_NUM_THREADS=18
  "${container}"
)

if [[ ! -f "${state_root}/REGISTERED.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/register_study.py" --repo-root /repo --step-root "${container_step}"
fi
if [[ ! -f "${state_root}/PANELS_PREPARED.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/prepare_panels.py" --repo-root /repo --step-root "${container_step}"
fi
if [[ ! -f "${state_root}/COMPONENT_TESTS.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/test_components.py" --repo-root /repo --step-root "${container_step}"
fi

run_gpu_phase() {
  local phase=$1
  local upper
  upper=$(printf '%s' "${phase}" | tr '[:lower:]' '[:upper:]')
  if [[ ! -f "${state_root}/${upper}_GENERATION_COMPLETE.json" ]]; then
    srun --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=18 --cpu-bind=none --exact \
      bash -c '"${@}" python '"${container_step}"'/scripts/generate_candidates.py --repo-root /repo --step-root '"${container_step}"' --phase '"${phase}"' --shard-id "${SLURM_PROCID}" --num-shards 4' _ "${gpu_command[@]}"
    "${base_command[@]}" python "${container_step}/scripts/assemble_phase.py" --step-root "${container_step}" --phase "${phase}"
  fi
}

run_gpu_phase development
if [[ ! -f "${state_root}/DEVELOPMENT_ANALYSIS_COMPLETE.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/analyze_phase.py" --repo-root /repo --step-root "${container_step}" --phase development --workers 64
fi
if [[ ! -f "${state_root}/STRATEGY_FROZEN.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/freeze_strategy.py" --step-root "${container_step}"
fi
run_gpu_phase final
if [[ ! -f "${state_root}/FINAL_ANALYSIS_COMPLETE.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/analyze_phase.py" --repo-root /repo --step-root "${container_step}" --phase final --workers 64
fi
if [[ ! -f "${state_root}/REPORT_COMPLETE.json" ]]; then
  "${base_command[@]}" python "${container_step}/scripts/report_results.py" --step-root "${container_step}"
fi
"${base_command[@]}" python "${container_step}/scripts/verify_study.py" --repo-root /repo --step-root "${container_step}"
