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
  "${repo_root}/deriv-gen/step-01b-scaled-space-selection") ;;
  *)
    echo "Refusing unexpected study root: ${step_root}" >&2
    exit 2
    ;;
esac

test -f "${container}"
test -f "${script_dir}/validate_export.py"
test -f "${script_dir}/prepare_populations.py"
test -f "${script_dir}/mine_mmp.py"
test -f "${script_dir}/analyze_spaces.py"
test -f "${script_dir}/verify_study.py"
test -f "${step_root}/config/protocol.json"
test -f "${step_root}/inputs/manifest.json"

mkdir -p \
  "${step_root}/exports" \
  "${step_root}/intermediate" \
  "${step_root}/outputs" \
  "${state_root}/cache" \
  "${state_root}/home" \
  "${state_root}/matplotlib" \
  "${state_root}/pycache" \
  "${state_root}/tmp" \
  "${state_root}/apptainer-cache" \
  "${state_root}/apptainer-tmp"

export APPTAINER_CACHEDIR="${state_root}/apptainer-cache"
export APPTAINER_TMPDIR="${state_root}/apptainer-tmp"
log_path="${state_root}/run.log"

container_rw=(
  apptainer exec
  --nv
  --cleanenv
  --home "${state_root}/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${deriv_root}:/repo/deriv-gen:rw"
  --bind "${repo_root}:${repo_root}:ro"
  --bind "${deriv_root}:${deriv_root}:rw"
  --bind "${container_root}:${container_root}:ro"
  --pwd /repo
  --env CUDA_VISIBLE_DEVICES=0
  --env PYTHONPATH=/repo/src:/repo/deriv-gen/step-01b-scaled-space-selection/scripts
  --env XDG_CACHE_HOME=/repo/deriv-gen/step-01b-scaled-space-selection/state/cache
  --env MPLCONFIGDIR=/repo/deriv-gen/step-01b-scaled-space-selection/state/matplotlib
  --env PYTHONPYCACHEPREFIX=/repo/deriv-gen/step-01b-scaled-space-selection/state/pycache
  --env TMPDIR=/repo/deriv-gen/step-01b-scaled-space-selection/state/tmp
  --env OMP_NUM_THREADS=48
  "${container}"
)

run_logged() {
  "${container_rw[@]}" "$@" 2>&1 | tee -a "${log_path}"
}

echo "[$(date --iso-8601=seconds)] scaled study launch" | tee -a "${log_path}"

if [[ ! -f "${state_root}/EXPORT_COMPLETE.json" ]]; then
  if [[ ! -f "${step_root}/exports/train_raw_hybrid_1m.pt" ]]; then
    run_logged \
      python -m gmolai_retrain.cli \
      --config "${repo_root}/configs/retrain.yaml" \
      embed \
      --run-dir /repo/runs/combined-zinc-pubchem-representation-pilot-mean-node-contrastive-001-desc050 \
      --checkpoint checkpoints/step-000010000.pt \
      --split train \
      --max-graphs 1000000 \
      --embedding-definition raw_hybrid \
      --sampling-seed 1618033 \
      --output /repo/deriv-gen/step-01b-scaled-space-selection/exports/train_raw_hybrid_1m.pt
  fi
  run_logged \
    python /repo/deriv-gen/step-01b-scaled-space-selection/scripts/validate_export.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01b-scaled-space-selection
fi

if [[ ! -f "${state_root}/FRAGMENTATION_COMPLETE.json" ]]; then
  run_logged \
    python /repo/deriv-gen/step-01b-scaled-space-selection/scripts/prepare_populations.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01b-scaled-space-selection \
    --workers 48
fi

if [[ ! -f "${state_root}/MMP_MINING_COMPLETE.json" ]]; then
  run_logged \
    python /repo/deriv-gen/step-01b-scaled-space-selection/scripts/mine_mmp.py \
    --step-root /repo/deriv-gen/step-01b-scaled-space-selection \
    --threads 48
fi

if [[ ! -f "${state_root}/COMPLETE.json" ]]; then
  run_logged \
    python /repo/deriv-gen/step-01b-scaled-space-selection/scripts/analyze_spaces.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01b-scaled-space-selection
fi

apptainer exec \
  --cleanenv \
  --home "${state_root}/home" \
  --bind "${repo_root}:/repo:ro" \
  --bind "${deriv_root}:/repo/deriv-gen:ro" \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${container_root}:${container_root}:ro" \
  --pwd /repo \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/repo/src:/repo/deriv-gen/step-01b-scaled-space-selection/scripts \
  "${container}" \
  python /repo/deriv-gen/step-01b-scaled-space-selection/scripts/verify_study.py \
    --repo-root /repo \
    --step-root /repo/deriv-gen/step-01b-scaled-space-selection \
  2>&1 | tee -a "${log_path}"

echo "[$(date --iso-8601=seconds)] scaled study verified" | tee -a "${log_path}"
