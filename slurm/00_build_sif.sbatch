#!/usr/bin/env bash
#SBATCH --job-name=gmolai-build-sif
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${CODE_DIR:?Export CODE_DIR before sbatch}"
: "${SIF_PATH:?Export SIF_PATH before sbatch}"
[[ -f "${CODE_DIR}/arrhenius.def" ]] || { echo "Missing arrhenius.def" >&2; exit 2; }
if [[ -e "${SIF_PATH}" ]]; then
  echo "Refusing to overwrite existing image: ${SIF_PATH}" >&2
  exit 2
fi
mkdir -p "$(dirname "${SIF_PATH}")"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/.apptainer-cache}"
export APPTAINER_TMPDIR="${SNIC_TMP:-${PROJECT_ROOT}/.apptainer-tmp}/apptainer-build-${SLURM_JOB_ID}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
cd "${CODE_DIR}"
# mksquashfs otherwise uses every allocated CPU and can exceed the Slurm memory
# cgroup while compressing the large NVIDIA PyG base image.
srun --nodes=1 --ntasks=1 apptainer build \
  --mksquashfs-args="-processors 2" "${SIF_PATH}" arrhenius.def
apptainer inspect "${SIF_PATH}"
