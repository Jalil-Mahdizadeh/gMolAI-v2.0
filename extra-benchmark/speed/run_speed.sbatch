#!/usr/bin/env bash
#SBATCH --job-name=gmolai-speed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=extra-benchmark/speed/logs/slurm-%j.out

set -Eeuo pipefail
umask 027

repository_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain"
project_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI"
benchmark_dir="${repository_root}/extra-benchmark/speed"
script_dir="${benchmark_dir}/scripts"
panel="${benchmark_dir}/inputs/common_test.tsv"
base_sif="${project_root}/containers/gmolai-pyg-25.09-arm64.sif"

declare -A images=(
  [gmolai]="${base_sif}"
  [morgan]="${base_sif}"
  [molai]="${project_root}/containers/MolAI/molai-epoch6-arm64.sif"
  [molformer]="${project_root}/containers/MolFormer/molformer-xl-both-10pct-arm64.sif"
  [smi_ted]="${project_root}/containers/SMI-TED/smi-ted-light-arm64.sif"
  [molclr_gin]="${project_root}/containers/MolCLR-GIN/molclr-gin-arm64.sif"
  [kermt_v2]="${project_root}/containers/KERMT-v2/kermt-v2-arm64.sif"
)
models=(gmolai morgan molai molformer smi_ted molclr_gin kermt_v2)

cd "${repository_root}"
export APPTAINER_CACHEDIR="${project_root}/.apptainer-cache"

container_cpu() {
  local image="$1"
  shift
  local cpus="${SLURM_CPUS_PER_TASK:-48}"
  local job_id="${SLURM_JOB_ID:-}"
  apptainer exec --cleanenv \
    --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" \
    "${image}" \
    env \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      SLURM_CPUS_PER_TASK="${cpus}" \
      SLURM_JOB_ID="${job_id}" \
      "$@"
}

container_gpu() {
  local image="$1"
  shift
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  local cpus="${SLURM_CPUS_PER_TASK:-48}"
  local job_id="${SLURM_JOB_ID:-}"
  apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${visible}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" \
    "${image}" \
    env \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      SLURM_CPUS_PER_TASK="${cpus}" \
      SLURM_JOB_ID="${job_id}" \
      "$@"
}

update_state() {
  container_cpu "${base_sif}" python "${script_dir}/state_update.py" "$@"
}

record_failure() {
  local exit_code="$?"
  set +e
  update_state --stage failed --status failed \
    --message "Speed benchmark stopped with exit code ${exit_code}"
  exit "${exit_code}"
}
trap record_failure ERR

update_state --stage preflight --status running \
  --message "Verifying the frozen common panel, model sources, and SIF hashes"
container_cpu "${base_sif}" python "${script_dir}/preflight.py"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,power.draw \
    --format=csv
fi

for model in "${models[@]}"; do
  update_state --stage "encode_${model}" --status running \
    --message "Measuring ${model} once at batch sizes 64, 128, 256, and 512"
  output="${benchmark_dir}/outputs/raw/${model}.json"
  if [[ "${model}" == "morgan" ]]; then
    container_cpu "${images[${model}]}" \
      python "${script_dir}/speed_adapter.py" \
      --model "${model}" --input "${panel}" --output "${output}" \
      --batch-sizes 64 128 256 512
  else
    container_gpu "${images[${model}]}" \
      python "${script_dir}/speed_adapter.py" \
      --model "${model}" --input "${panel}" --output "${output}" \
      --batch-sizes 64 128 256 512
  fi
done

update_state --stage finalize --status running \
  --message "Validating all timings and producing compact tables and figures"
container_cpu "${base_sif}" python "${script_dir}/finalize.py"
update_state --stage complete --status complete \
  --message "Single-pass common locked-test speed benchmark completed"
trap - ERR

