#!/usr/bin/env bash
#SBATCH --job-name=gmolai-promotion-trajectory
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%4
#SBATCH --output=slurm-gmolai-promotion-trajectory-%A_%a.out

set -euo pipefail

: "${CODE_DIR:?Set CODE_DIR (source configs/arrhenius.env before submission)}"
source "${CODE_DIR}/slurm/common.sh"

: "${GMOLAI_PROMOTION_RUN_DIR:?Set GMOLAI_PROMOTION_RUN_DIR}"
: "${GMOLAI_PROMOTION_SWEEP_ROOT:?Set GMOLAI_PROMOTION_SWEEP_ROOT}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) step="000005000" ;;
  1) step="000007500" ;;
  2) step="000010000" ;;
  3) step="000012500" ;;
  4) step="000015000" ;;
  *) echo "Unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

run_dir="${GMOLAI_PROMOTION_RUN_DIR}"
sweep_root="${GMOLAI_PROMOTION_SWEEP_ROOT}"
checkpoint="checkpoints/step-${step}.pt"
checkpoint_path="${run_dir}/${checkpoint}"
step_dir="${sweep_root}/step-${step}"

[[ -d "${run_dir}" ]] || { echo "Missing run directory: ${run_dir}" >&2; exit 2; }
[[ -f "${checkpoint_path}" ]] || { echo "Missing checkpoint: ${checkpoint_path}" >&2; exit 2; }
[[ "${sweep_root}" == "${run_dir}"/* ]] || {
  echo "Sweep root must be inside the production run directory" >&2
  exit 2
}
mkdir -p "${step_dir}"

calibration_source="${step_dir}/calibration-source-raw-hybrid-100k-seed271828.pt"
calibrator="${step_dir}/embedding-calibrator-100k-seed271828.pt"
probe_train="${step_dir}/probe-train-standardized-raw-hybrid-w3-10k-seed20260810.pt"
probe_validation="${step_dir}/probe-validation-standardized-raw-hybrid-w3-50k-seed20260810.pt"
probe_result="${step_dir}/representation-probe-standardized-raw-hybrid-w3-50k-sim5k.json"
downstream_result="${step_dir}/moleculenet-full-diagnostic-standardized-raw-hybrid-w3-10splits.json"
complete_marker="${step_dir}/COMPLETE"

if [[ -f "${complete_marker}" ]]; then
  echo "Step ${step} is already complete: ${step_dir}"
  exit 0
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

if [[ ! -f "${calibration_source}" ]]; then
  container_gpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" embed \
    --run-dir "${run_dir}" \
    --checkpoint "${checkpoint}" \
    --split train \
    --max-graphs 100000 \
    --sampling-seed 271828 \
    --embedding-definition raw_hybrid \
    --output "${calibration_source}"
fi

if [[ ! -f "${calibrator}" ]]; then
  container_cpu python -m gmolai_retrain.cli fit-embedding-calibrator \
    --embeddings "${calibration_source}" \
    --minimum-graphs 100000 \
    --output "${calibrator}"
fi

if [[ ! -f "${probe_train}" ]]; then
  container_gpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" embed \
    --run-dir "${run_dir}" \
    --checkpoint "${checkpoint}" \
    --split train \
    --max-graphs 10000 \
    --sampling-seed 20260810 \
    --embedding-definition standardized_raw_hybrid \
    --mean-node-weight 3 \
    --calibrator "${calibrator}" \
    --output "${probe_train}"
fi

if [[ ! -f "${probe_validation}" ]]; then
  container_gpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" embed \
    --run-dir "${run_dir}" \
    --checkpoint "${checkpoint}" \
    --split validation \
    --max-graphs 50000 \
    --sampling-seed 20260810 \
    --embedding-definition standardized_raw_hybrid \
    --mean-node-weight 3 \
    --calibrator "${calibrator}" \
    --output "${probe_validation}"
fi

if [[ ! -f "${probe_result}" ]]; then
  container_cpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" probe \
    --train-embeddings "${probe_train}" \
    --validation-embeddings "${probe_validation}" \
    --similarity-graphs 5000 \
    --output "${probe_result}"
fi

if [[ ! -f "${downstream_result}" ]]; then
  container_gpu python -m gmolai_retrain.cli --config "${CONFIG_PATH}" benchmark-downstream \
    --run-dir "${run_dir}" \
    --checkpoint "${checkpoint}" \
    --datasets-dir "${CODE_DIR}/work/downstream_benchmarks/moleculenet" \
    --scaffold-splits 10 \
    --embedding-definition standardized_raw_hybrid \
    --calibrator "${calibrator}" \
    --output "${downstream_result}"
fi

touch "${complete_marker}"
echo "Completed full promotion evidence for step ${step}: ${step_dir}"
