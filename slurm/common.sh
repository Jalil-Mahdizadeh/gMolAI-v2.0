#!/usr/bin/env bash
set -euo pipefail

: "${NAISS_PROJECT:?Set NAISS_PROJECT (source configs/arrhenius.env)}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT}"
: "${CODE_DIR:?Set CODE_DIR}"
: "${SIF_PATH:?Set SIF_PATH}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"

[[ -d "${PROJECT_ROOT}" ]] || { echo "Missing PROJECT_ROOT: ${PROJECT_ROOT}" >&2; exit 2; }
[[ -d "${CODE_DIR}" ]] || { echo "Missing CODE_DIR: ${CODE_DIR}" >&2; exit 2; }
[[ -f "${SIF_PATH}" ]] || { echo "Missing SIF_PATH: ${SIF_PATH}" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Missing CONFIG_PATH: ${CONFIG_PATH}" >&2; exit 2; }
command -v apptainer >/dev/null || { echo "apptainer is not available" >&2; exit 2; }

BIND_SPEC="${PROJECT_ROOT}:${PROJECT_ROOT}"
if [[ -n "${GMOLAI_EXTRA_BINDS:-}" ]]; then
  BIND_SPEC="${BIND_SPEC},${GMOLAI_EXTRA_BINDS}"
fi

container_cpu() {
  srun --nodes=1 --ntasks=1 --cpu-bind=none apptainer exec --cleanenv \
    --bind "${BIND_SPEC}" --pwd "${CODE_DIR}" "${SIF_PATH}" \
    env PYTHONPATH="${CODE_DIR}/src" PYTHONUNBUFFERED=1 "$@"
}

container_login() {
  apptainer exec --cleanenv \
    --bind "${BIND_SPEC}" --pwd "${CODE_DIR}" "${SIF_PATH}" \
    env PYTHONPATH="${CODE_DIR}/src" PYTHONUNBUFFERED=1 "$@"
}

container_gpu() {
  local step_gpus="${GMOLAI_STEP_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
  [[ "${step_gpus}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid GPU count for job step: ${step_gpus}" >&2
    return 2
  }
  # Arrhenius assigns GPU TRES separately to each srun step.  Without an
  # explicit request, a one-task step can expose only one device even when the
  # enclosing allocation owns two or four, causing torchrun rank>0 to fail
  # with an invalid device ordinal.
  srun --nodes=1 --ntasks=1 --cpu-bind=none --gpus="${step_gpus}" \
    apptainer exec --cleanenv --nv \
    --bind "${BIND_SPEC}" --pwd "${CODE_DIR}" "${SIF_PATH}" \
    env PYTHONPATH="${CODE_DIR}/src" PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN "$@"
}
