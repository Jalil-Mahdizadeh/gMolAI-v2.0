#!/usr/bin/env bash
set -euo pipefail

STEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STEP_ROOT}/../.." && pwd)"
CONTAINER_IMAGE="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif"

case "${1:-}" in
  "")
    OVERWRITE=0
    ;;
  --overwrite)
    OVERWRITE=1
    ;;
  *)
    echo "Usage: $0 [--overwrite]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
  echo "Missing benchmark container: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

mkdir -p \
  "${STEP_ROOT}/state/home" \
  "${STEP_ROOT}/state/cache" \
  "${STEP_ROOT}/state/matplotlib" \
  "${STEP_ROOT}/state/tmp"

export APPTAINER_CACHEDIR="${STEP_ROOT}/state/cache/apptainer"
export APPTAINER_TMPDIR="${STEP_ROOT}/state/tmp"

apptainer exec --nv --cleanenv \
  --home "${STEP_ROOT}/state/home" \
  --bind "${REPO_ROOT}:/repo:ro" \
  --bind "${STEP_ROOT}:/step:rw" \
  --bind "${STEP_ROOT}/state/tmp:/tmp:rw" \
  --env XDG_CACHE_HOME=/step/state/cache \
  --env MPLCONFIGDIR=/step/state/matplotlib \
  --env TMPDIR=/step/state/tmp \
  --env PYTHONPYCACHEPREFIX=/step/state/pycache \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env GMOLAI_BENCHMARK_OVERWRITE="${OVERWRITE}" \
  --env GMOLAI_CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  "${CONTAINER_IMAGE}" \
  bash -lc '
    set -euo pipefail
    overwrite_args=()
    if [[ "${GMOLAI_BENCHMARK_OVERWRITE}" == "1" ]]; then
      overwrite_args=(--overwrite)
    fi
    python /step/scripts/prepare_inputs.py --repo-root /repo --step-root /step "${overwrite_args[@]}"
    python /step/scripts/run_benchmark.py --repo-root /repo --step-root /step "${overwrite_args[@]}"
    python /step/scripts/plot_results.py --step-root /step
    python /step/scripts/report_results.py --step-root /step
    python /step/scripts/verify_results.py --step-root /step
  '
