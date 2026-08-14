#!/usr/bin/env bash
#SBATCH --job-name=gmolai-tdc-admet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=extra-benchmark/tdc-admet/logs/slurm-%j.out

set -Eeuo pipefail

repository_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain"
project_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI"
benchmark_dir="${repository_root}/extra-benchmark/tdc-admet"
script_dir="${benchmark_dir}/scripts"
qualified_script_dir="${repository_root}/extra-benchmark/test-partition/scripts"
base_sif="${project_root}/containers/gmolai-pyg-25.09-arm64.sif"

declare -A images=(
  [morgan]="${base_sif}"
  [molai]="${project_root}/containers/MolAI/molai-epoch6-arm64.sif"
  [molformer]="${project_root}/containers/MolFormer/molformer-xl-both-10pct-arm64.sif"
  [smi_ted]="${project_root}/containers/SMI-TED/smi-ted-light-arm64.sif"
  [molclr_gin]="${project_root}/containers/MolCLR-GIN/molclr-gin-arm64.sif"
  [kermt_v2]="${project_root}/containers/KERMT-v2/kermt-v2-arm64.sif"
)
declare -A batches=(
  [morgan]=4096
  [molai]=256
  [molformer]=128
  [smi_ted]=100
  [molclr_gin]=256
  [kermt_v2]=64
)
screen_models=(morgan molai molformer smi_ted molclr_gin kermt_v2)
primary_models=(gmolai morgan molai molformer smi_ted molclr_gin kermt_v2)
all_models=(gmolai morgan molai molformer smi_ted molclr_gin kermt_v2 descriptor_13)

cd "${repository_root}"
export APPTAINER_CACHEDIR="${project_root}/.apptainer-cache"

container_cpu() {
  apptainer exec --cleanenv \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}" \
    --bind "${project_root}:${project_root}" \
    --bind "/tmp:/tmp" \
    --pwd "${repository_root}" \
    "${base_sif}" \
    env PYTHONUNBUFFERED=1 PYTHONPATH="src:${script_dir}" "$@"
}

container_gpu() {
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${visible}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}" \
    --bind "${project_root}:${project_root}" \
    --bind "/tmp:/tmp" \
    --pwd "${repository_root}" \
    "${base_sif}" \
    env PYTHONUNBUFFERED=1 PYTHONPATH="src:${script_dir}" "$@"
}

model_cpu() {
  local image="$1"
  shift
  apptainer exec --cleanenv \
    --bind "${project_root}:${project_root}" \
    --bind "/tmp:/tmp" \
    --pwd "${repository_root}" \
    "${image}" \
    env PYTHONUNBUFFERED=1 "$@"
}

model_gpu() {
  local image="$1"
  shift
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${visible}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-interactive}" \
    --bind "${project_root}:${project_root}" \
    --bind "/tmp:/tmp" \
    --pwd "${repository_root}" \
    "${image}" \
    env PYTHONUNBUFFERED=1 "$@"
}

update_state() {
  container_cpu python "${script_dir}/state_update.py" "$@"
}

record_failure() {
  local exit_code="$?"
  update_state --stage failed --status failed \
    --message "Pipeline stopped with exit code ${exit_code}" || true
  exit "${exit_code}"
}
trap record_failure ERR

ensure_embedding() {
  local model="$1"
  local embedding="${benchmark_dir}/outputs/embeddings/${model}.npy"
  local metadata="${benchmark_dir}/outputs/embeddings/${model}.json"
  if [[ -f "${embedding}" && -f "${metadata}" ]]; then
    container_cpu python "${script_dir}/validate_embedding.py" \
      --model "${model}" --embedding "${embedding}" --metadata "${metadata}"
    return
  fi
  if [[ -e "${embedding}" || -L "${embedding}" || -e "${metadata}" || -L "${metadata}" ]]; then
    echo "Incomplete or unsafe existing output for ${model}" >&2
    return 2
  fi
  if [[ "${model}" == "gmolai" ]]; then
    container_gpu python "${script_dir}/gmolai_adapter.py" \
      --input "${benchmark_dir}/inputs/common_panel.tsv" \
      --output "${embedding}" --metadata "${metadata}"
  elif [[ "${model}" == "descriptor_13" ]]; then
    container_cpu python "${script_dir}/descriptor_adapter.py" \
      --input "${benchmark_dir}/inputs/common_panel.tsv" \
      --output "${embedding}" --metadata "${metadata}"
  elif [[ "${model}" == "morgan" ]]; then
    model_cpu "${images[${model}]}" python "${qualified_script_dir}/adapter.py" \
      --model "${model}" --mode encode \
      --input "${benchmark_dir}/inputs/common_panel.tsv" \
      --output "${embedding}" --metadata "${metadata}" \
      --batch-size "${batches[${model}]}"
  else
    model_gpu "${images[${model}]}" python "${qualified_script_dir}/adapter.py" \
      --model "${model}" --mode encode \
      --input "${benchmark_dir}/inputs/common_panel.tsv" \
      --output "${embedding}" --metadata "${metadata}" \
      --batch-size "${batches[${model}]}"
  fi
  container_cpu python "${script_dir}/validate_embedding.py" \
    --model "${model}" --embedding "${embedding}" --metadata "${metadata}"
}

mkdir -p "${benchmark_dir}/outputs/embeddings" "${benchmark_dir}/outputs/results"
update_state --stage preflight --status running \
  --message "Verifying frozen assets, containers, repository state, and one-GPU contract"
container_gpu python "${script_dir}/preflight.py"

update_state --stage data --status running \
  --message "Fetching and hash-verifying the DOI-backed 22-endpoint snapshot"
container_cpu python "${script_dir}/prepare.py"

update_state --stage coverage --status running \
  --message "Screening each unique canonical identity with qualified comparator preprocessing"
for model in "${screen_models[@]}"; do
  report="${benchmark_dir}/state/${model}-screen.json"
  if [[ ! -f "${report}" ]]; then
    model_cpu "${images[${model}]}" python "${qualified_script_dir}/adapter.py" \
      --model "${model}" --mode screen \
      --input "${benchmark_dir}/inputs/full_panel.tsv" \
      --output "${report}" --batch-size "${batches[${model}]}"
  fi
done
container_cpu python "${script_dir}/assemble_common.py"

update_state --stage representation --status running \
  --message "Exporting seven frozen representations and the diagnostic descriptor panel"
for model in "${all_models[@]}"; do
  ensure_embedding "${model}"
done

update_state --stage evaluation --status running \
  --message "Fitting paired shallow probes for all 22 endpoints and five scaffold-selection seeds"
evaluation_pids=()
for model in "${all_models[@]}"; do
  container_cpu env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python "${script_dir}/evaluate.py" --model "${model}" &
  evaluation_pids+=("$!")
done
evaluation_failures=0
for pid in "${evaluation_pids[@]}"; do
  if ! wait "${pid}"; then
    evaluation_failures=$((evaluation_failures + 1))
  fi
done
if (( evaluation_failures > 0 )); then
  echo "${evaluation_failures} evaluator process(es) failed" >&2
  exit 1
fi

update_state --stage integrity --status running \
  --message "Aggregating category-balanced ranks and checksum-sealing artifacts"
container_cpu python "${script_dir}/finalize.py"
update_state --stage complete --status complete \
  --message "TDC ADMET frozen-representation benchmark completed"
