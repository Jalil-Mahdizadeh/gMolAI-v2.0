#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
step_root=$(cd -- "${script_dir}/.." && pwd)
deriv_root=$(cd -- "${step_root}/.." && pwd)
repo_root=$(cd -- "${deriv_root}/.." && pwd)
container=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif
container_root=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers
container_step=/repo/deriv-gen/step-02d-generation-scaling

case "${step_root}" in
  "${repo_root}/deriv-gen/step-02d-generation-scaling") ;;
  *) echo "Refusing unexpected Step 2d root: ${step_root}" >&2; exit 2 ;;
esac
test -f "${container}"
test -f "${script_dir}/sa_extension.py"

mkdir -p "${step_root}/state/sa_tmp" "${step_root}/state/sa_pycache" "${step_root}/state/sa_matplotlib"

command=(
  apptainer exec
  --cleanenv
  --home "${step_root}/state/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${step_root}:${container_step}:rw"
  --bind "${container_root}:${container_root}:ro"
  --pwd /repo
  --env PYTHONPATH=${container_step}/scripts:/repo/src:/repo/deriv-gen/step-02-decoder-feasibility/scripts:/repo/deriv-gen/step-02b-candidate-reranking/scripts
  --env PYTHONPYCACHEPREFIX=${container_step}/state/sa_pycache
  --env MPLCONFIGDIR=${container_step}/state/sa_matplotlib
  --env TMPDIR=${container_step}/state/sa_tmp
  --env OMP_NUM_THREADS=1
  --env MKL_NUM_THREADS=1
  "${container}"
  python "${container_step}/scripts/sa_extension.py"
  --repo-root /repo
  --step-root "${container_step}"
)

"${command[@]}" register
"${command[@]}" component-test
"${command[@]}" analyze --workers "${SA_WORKERS:-64}"
"${command[@]}" report
"${command[@]}" verify
