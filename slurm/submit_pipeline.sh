#!/usr/bin/env bash
set -euo pipefail

# The Arrhenius login nodes are amd64 while the GH200 image is arm64. This
# wrapper performs no container execution; it submits an arm64 orchestrator.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CODE_DIR_DEFAULT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ENV_FILE="${1:-${CODE_DIR_DEFAULT}/configs/arrhenius.env}"
[[ -f "${ENV_FILE}" ]] || { echo "Missing environment file: ${ENV_FILE}" >&2; exit 2; }
source "${ENV_FILE}"

: "${NAISS_PROJECT:?Set NAISS_PROJECT}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT}"
: "${CODE_DIR:?Set CODE_DIR}"
: "${SIF_PATH:?Set SIF_PATH}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"
[[ -f "${SIF_PATH}" ]] || { echo "Missing SIF: ${SIF_PATH}" >&2; exit 2; }
[[ -f "${CONFIG_PATH}" ]] || { echo "Missing config: ${CONFIG_PATH}" >&2; exit 2; }

export NAISS_PROJECT PROJECT_ROOT CODE_DIR SIF_PATH CONFIG_PATH
export APPTAINER_CACHEDIR GMOLAI_EXTRA_BINDS
export CANONICAL_MAX_CONCURRENT DEDUP_MAX_CONCURRENT FEATURIZE_MAX_CONCURRENT AUTO_SUBMIT_TRAIN

orchestrator_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL \
  "${CODE_DIR}/slurm/01_orchestrate_pipeline.sbatch")
printf 'pipeline_orchestrator=%s\n' "${orchestrator_job}"
printf 'Child job IDs will appear in slurm-gmolai-orchestrate-%s.out\n' "${orchestrator_job}"
