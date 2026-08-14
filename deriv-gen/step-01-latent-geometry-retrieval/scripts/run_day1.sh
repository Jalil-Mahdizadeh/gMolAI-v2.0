#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
step_root=$(cd -- "${script_dir}/.." && pwd)
deriv_root=$(cd -- "${step_root}/.." && pwd)
repo_root=$(cd -- "${deriv_root}/.." && pwd)
container=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif
container_root=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers

case "${step_root}" in
  "${repo_root}/deriv-gen/step-01-latent-geometry-retrieval") ;;
  *)
    echo "Refusing unexpected study root: ${step_root}" >&2
    exit 2
    ;;
esac

test -f "${container}"
test -f "${script_dir}/day1_study.py"
test -f "${step_root}/config/protocol.json"
test -f "${step_root}/inputs/manifest.json"

mkdir -p \
  "${step_root}/state/cache" \
  "${step_root}/state/home" \
  "${step_root}/state/matplotlib" \
  "${step_root}/state/pycache" \
  "${step_root}/state/tmp"

log_path="${step_root}/state/run.log"

apptainer exec \
  --nv \
  --cleanenv \
  --home "${step_root}/state/home" \
  --bind "${repo_root}:/repo:ro" \
  --bind "${deriv_root}:/repo/deriv-gen:rw" \
  --bind "${container_root}:${container_root}:ro" \
  --pwd /repo \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env XDG_CACHE_HOME=/repo/deriv-gen/step-01-latent-geometry-retrieval/state/cache \
  --env MPLCONFIGDIR=/repo/deriv-gen/step-01-latent-geometry-retrieval/state/matplotlib \
  --env PYTHONPYCACHEPREFIX=/repo/deriv-gen/step-01-latent-geometry-retrieval/state/pycache \
  --env TMPDIR=/repo/deriv-gen/step-01-latent-geometry-retrieval/state/tmp \
  "${container}" \
  python /repo/deriv-gen/step-01-latent-geometry-retrieval/scripts/day1_study.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01-latent-geometry-retrieval \
  2>&1 | tee "${log_path}"

apptainer exec \
  --cleanenv \
  --home "${step_root}/state/home" \
  --bind "${repo_root}:/repo:ro" \
  --bind "${deriv_root}:/repo/deriv-gen:ro" \
  --bind "${container_root}:${container_root}:ro" \
  --pwd /repo \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "${container}" \
  python /repo/deriv-gen/step-01-latent-geometry-retrieval/scripts/verify_day1.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01-latent-geometry-retrieval

