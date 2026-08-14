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
  "${repo_root}/deriv-gen/step-02-decoder-feasibility") ;;
  *)
    echo "Refusing unexpected study root: ${step_root}" >&2
    exit 2
    ;;
esac

if command -v apptainer >/dev/null 2>&1; then
  runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then
  runtime=singularity
else
  echo "Neither apptainer nor singularity is available" >&2
  exit 2
fi

test -f "${container}"
for script in test_components.py prepare_data.py train_decoder.py activate_extension.py select_decode.py export_decoder.py evaluate_decoder.py report_results.py verify_study.py; do
  test -f "${script_dir}/${script}"
done
test -f "${step_root}/config/protocol.json"
test -f "${step_root}/inputs/manifest.json"

mkdir -p   "${step_root}/prepared"   "${step_root}/checkpoints"   "${step_root}/outputs/raw"   "${step_root}/outputs/tables"   "${step_root}/outputs/figures"   "${step_root}/outputs/examples"   "${state_root}/cache"   "${state_root}/home"   "${state_root}/matplotlib"   "${state_root}/pycache"   "${state_root}/tmp"   "${state_root}/apptainer-cache"   "${state_root}/apptainer-tmp"

export APPTAINER_CACHEDIR="${state_root}/apptainer-cache"
export APPTAINER_TMPDIR="${state_root}/apptainer-tmp"
log_path="${state_root}/run.log"

container_command=(
  "${runtime}" exec
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
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env PYTHONPATH=/repo/src:/repo/deriv-gen/step-02-decoder-feasibility/scripts
  --env XDG_CACHE_HOME=/repo/deriv-gen/step-02-decoder-feasibility/state/cache
  --env MPLCONFIGDIR=/repo/deriv-gen/step-02-decoder-feasibility/state/matplotlib
  --env PYTHONPYCACHEPREFIX=/repo/deriv-gen/step-02-decoder-feasibility/state/pycache
  --env TMPDIR=/repo/deriv-gen/step-02-decoder-feasibility/state/tmp
  --env OMP_NUM_THREADS=48
  "${container}"
)

run_logged() {
  "${container_command[@]}" "$@" 2>&1 | tee -a "${log_path}"
}

echo "[$(date --iso-8601=seconds)] Step 2 decoder study launch" | tee -a "${log_path}"

run_logged python /repo/deriv-gen/step-02-decoder-feasibility/scripts/test_components.py

if [[ ! -f "${state_root}/PREPARED.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/prepare_data.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility
fi

if [[ ! -f "${state_root}/pilot/PILOT_COMPLETE.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/train_decoder.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility     --mode pilot     --pilot-rows 65536     --pilot-epochs 1
fi

if [[ ! -f "${state_root}/DEVELOPMENT_EXTENSION_DECISION.json" ]]; then
  if [[ ! -f "${state_root}/TRAINING_COMPLETE.json" ]]; then
    run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/train_decoder.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility     --mode full     --stop-after-epoch 12
  fi
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/activate_extension.py     --step-root /repo/deriv-gen/step-02-decoder-feasibility
fi

if [[ ! -f "${state_root}/TRAINING_COMPLETE.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/train_decoder.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility     --mode full
fi

if [[ ! -f "${state_root}/DECODE_SELECTION.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/select_decode.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility
fi

if [[ ! -f "${state_root}/DECODER_EXPORT.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/export_decoder.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility
fi

if [[ ! -f "${state_root}/EVALUATION_COMPLETE.json" ]]; then
  run_logged     python /repo/deriv-gen/step-02-decoder-feasibility/scripts/evaluate_decoder.py     --repo-root /repo     --step-root /repo/deriv-gen/step-02-decoder-feasibility
fi

run_logged   python /repo/deriv-gen/step-02-decoder-feasibility/scripts/report_results.py   --repo-root /repo   --step-root /repo/deriv-gen/step-02-decoder-feasibility

run_logged   python /repo/deriv-gen/step-02-decoder-feasibility/scripts/verify_study.py   --repo-root /repo   --step-root /repo/deriv-gen/step-02-decoder-feasibility

echo "[$(date --iso-8601=seconds)] Step 2 decoder study verified" | tee -a "${log_path}"
