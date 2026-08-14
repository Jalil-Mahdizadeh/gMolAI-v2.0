#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
step_root=$(cd -- "${script_dir}/.." && pwd)
deriv_root=$(cd -- "${step_root}/.." && pwd)
repo_root=$(cd -- "${deriv_root}/.." && pwd)
container=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif
container_root=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers
state_root="${step_root}/state"

case "${step_root}" in
  "${repo_root}/deriv-gen/step-02c-chemical-characterization") ;;
  *)
    echo "Refusing unexpected Step 2c root: ${step_root}" >&2
    exit 2
    ;;
esac

test -f "${container}"
test -f "${script_dir}/register_study.py"
test -f "${script_dir}/test_components.py"
test -f "${script_dir}/audit_candidates.py"
test -f "${script_dir}/report_results.py"
test -f "${script_dir}/verify_study.py"
test -f "${step_root}/config/protocol.json"

mkdir -p \
  "${step_root}/inputs" \
  "${step_root}/intermediate" \
  "${step_root}/outputs/raw" \
  "${step_root}/outputs/tables" \
  "${step_root}/outputs/figures" \
  "${state_root}/home" \
  "${state_root}/cache" \
  "${state_root}/matplotlib" \
  "${state_root}/pycache" \
  "${state_root}/tmp" \
  "${state_root}/apptainer-cache" \
  "${state_root}/apptainer-tmp"

export APPTAINER_CACHEDIR="${state_root}/apptainer-cache"
export APPTAINER_TMPDIR="${state_root}/apptainer-tmp"

container_command=(
  apptainer exec
  --cleanenv
  --home "${state_root}/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${step_root}:/repo/deriv-gen/step-02c-chemical-characterization:rw"
  --bind "${container_root}:${container_root}:ro"
  --pwd /repo
  --env PYTHONPATH=/repo/src:/repo/deriv-gen/step-02c-chemical-characterization/scripts:/repo/deriv-gen/step-01b-scaled-space-selection/scripts
  --env XDG_CACHE_HOME=/repo/deriv-gen/step-02c-chemical-characterization/state/cache
  --env MPLCONFIGDIR=/repo/deriv-gen/step-02c-chemical-characterization/state/matplotlib
  --env PYTHONPYCACHEPREFIX=/repo/deriv-gen/step-02c-chemical-characterization/state/pycache
  --env TMPDIR=/repo/deriv-gen/step-02c-chemical-characterization/state/tmp
  --env OMP_NUM_THREADS=48
  "${container}"
)

container_step_root=/repo/deriv-gen/step-02c-chemical-characterization

if [[ ! -f "${state_root}/REGISTERED.json" ]]; then
  "${container_command[@]}" python "${container_step_root}/scripts/register_study.py" \
    --repo-root /repo --step-root "${container_step_root}"
fi

if [[ ! -f "${state_root}/COMPONENT_TESTS.json" ]]; then
  "${container_command[@]}" python "${container_step_root}/scripts/test_components.py" \
    --repo-root /repo --step-root "${container_step_root}"
fi

if [[ ! -f "${state_root}/ANALYSIS_COMPLETE.json" ]]; then
  "${container_command[@]}" python "${container_step_root}/scripts/audit_candidates.py" \
    --repo-root /repo --step-root "${container_step_root}" --workers 48
fi

if [[ ! -f "${state_root}/REPORT_COMPLETE.json" ]]; then
  "${container_command[@]}" python "${container_step_root}/scripts/report_results.py" \
    --repo-root /repo --step-root "${container_step_root}"
fi

"${container_command[@]}" python "${container_step_root}/scripts/verify_study.py" \
  --repo-root /repo --step-root "${container_step_root}"
