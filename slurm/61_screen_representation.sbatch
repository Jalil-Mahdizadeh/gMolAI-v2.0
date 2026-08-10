#!/usr/bin/env bash
#SBATCH --job-name=gmolai-rep-screen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=slurm-outs/rep-screen-%A_%a.out

source "${CODE_DIR}/slurm/common.sh"

case "${SLURM_ARRAY_TASK_ID}" in
  0) name="base"; variant="" ;;
  1) name="strong-vicreg"; variant="${CODE_DIR}/configs/representation-strong-vicreg.yaml" ;;
  2) name="128d"; variant="${CODE_DIR}/configs/representation-128d.yaml" ;;
  3) name="mask50"; variant="${CODE_DIR}/configs/representation-mask50.yaml" ;;
  4) name="no-vicreg"; variant="${CODE_DIR}/configs/representation-no-vicreg.yaml" ;;
  5) name="strong-128d"; variant="${CODE_DIR}/configs/representation-strong-128d.yaml" ;;
  6) name="invariance-only"; variant="${CODE_DIR}/configs/representation-invariance-only.yaml" ;;
  7) name="no-covariance"; variant="${CODE_DIR}/configs/representation-no-covariance.yaml" ;;
  8) name="gentle-vicreg"; variant="${CODE_DIR}/configs/representation-gentle-vicreg.yaml" ;;
  9) name="variance-only"; variant="${CODE_DIR}/configs/representation-variance-only.yaml" ;;
  10) name="projector"; variant="${CODE_DIR}/configs/representation-projector.yaml" ;;
  11) name="projector-strong"; variant="${CODE_DIR}/configs/representation-projector-strong.yaml" ;;
  12) name="projector-standard"; variant="${CODE_DIR}/configs/representation-projector-standard.yaml" ;;
  13) name="projector-standard-strong"; variant="${CODE_DIR}/configs/representation-projector-standard-strong.yaml" ;;
  14) name="projector-standard-strong-seed1337"; variant="${CODE_DIR}/configs/representation-projector-standard-strong-seed1337.yaml" ;;
  15) name="projector-standard-strong-seed2027"; variant="${CODE_DIR}/configs/representation-projector-standard-strong-seed2027.yaml" ;;
  16) name="contrastive-005"; variant="${CODE_DIR}/configs/representation-contrastive-005.yaml" ;;
  17) name="contrastive-010"; variant="${CODE_DIR}/configs/representation-contrastive-010.yaml" ;;
  18) name="contrastive-020"; variant="${CODE_DIR}/configs/representation-contrastive-020.yaml" ;;
  19) name="contrastive-detached-005"; variant="${CODE_DIR}/configs/representation-contrastive-detached-005.yaml" ;;
  20) name="contrastive-detached-010"; variant="${CODE_DIR}/configs/representation-contrastive-detached-010.yaml" ;;
  21) name="contrastive-002"; variant="${CODE_DIR}/configs/representation-contrastive-002.yaml" ;;
  *) echo "Unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

screen_root="${GMOLAI_SCREEN_ROOT:-${CODE_DIR}/runs/representation-screens-v1}"
run_dir="${screen_root}/${name}"
arguments=(
  -m gmolai_retrain.cli --config "${CONFIG_PATH}" train
  --plan "${CODE_DIR}/configs/representation-v1.yaml"
  --plan "${CODE_DIR}/configs/representation-screen.yaml"
  --run-dir "${run_dir}"
)
if [[ -n "${variant}" ]]; then
  arguments+=(--plan "${variant}")
fi
container_gpu python "${arguments[@]}"
