#!/usr/bin/env bash
#SBATCH --job-name=gmolai-extra-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=extra-benchmark/test-partition/logs/slurm-%j.out

set -euo pipefail

repository_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain"
project_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI"
benchmark_dir="${repository_root}/extra-benchmark/test-partition"
script_dir="${benchmark_dir}/scripts"
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
models=(morgan molai molformer smi_ted molclr_gin kermt_v2)

cd "${repository_root}"
export APPTAINER_CACHEDIR="${project_root}/.apptainer-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

container_cpu() {
  local image="$1"
  shift
  apptainer exec --cleanenv \
    --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" \
    "${image}" \
    env PYTHONUNBUFFERED=1 "$@"
}

container_gpu() {
  local image="$1"
  shift
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  apptainer exec --cleanenv --nv \
    --env "CUDA_VISIBLE_DEVICES=${visible}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" \
    "${image}" \
    env PYTHONUNBUFFERED=1 "$@"
}

update_state() {
  container_cpu "${base_sif}" python "${script_dir}/state_update.py" "$@"
}

record_failure() {
  local exit_code="$?"
  update_state --stage failed --status failed \
    --message "Pipeline stopped with exit code ${exit_code}" || true
  exit "${exit_code}"
}
trap record_failure ERR

screen_panel() {
  local name="$1"
  local panel="$2"
  local model
  for model in "${models[@]}"; do
    container_cpu "${images[${model}]}" \
      python "${script_dir}/adapter.py" \
      --model "${model}" \
      --mode screen \
      --input "${panel}" \
      --output "${benchmark_dir}/state/${model}-${name}-screen.json" \
      --batch-size "${batches[${model}]}"
  done
}

ensure_embedding() {
  local model="$1"
  local name="$2"
  local panel="$3"
  local directory="$4"
  local embedding="${directory}/${model}-${name}.npy"
  local metadata="${directory}/${model}-${name}.json"
  mkdir -p "${directory}"
  if [[ -f "${embedding}" && -f "${metadata}" ]]; then
    container_cpu "${base_sif}" \
      python "${script_dir}/validate_adapter_output.py" \
      --model "${model}" --input "${panel}" \
      --embedding "${embedding}" --metadata "${metadata}"
    return
  fi
  if [[ -e "${embedding}" || -L "${embedding}" || -e "${metadata}" || -L "${metadata}" ]]; then
    echo "Incomplete or unsafe existing adapter output for ${model}-${name}" >&2
    return 2
  fi
  if [[ "${model}" == "morgan" ]]; then
    container_cpu "${images[${model}]}" \
      python "${script_dir}/adapter.py" \
      --model "${model}" --mode encode --input "${panel}" \
      --output "${embedding}" --metadata "${metadata}" \
      --batch-size "${batches[${model}]}"
  else
    container_gpu "${images[${model}]}" \
      python "${script_dir}/adapter.py" \
      --model "${model}" --mode encode --input "${panel}" \
      --output "${embedding}" --metadata "${metadata}" \
      --batch-size "${batches[${model}]}"
  fi
  container_cpu "${base_sif}" \
    python "${script_dir}/validate_adapter_output.py" \
    --model "${model}" --input "${panel}" \
    --embedding "${embedding}" --metadata "${metadata}"
}

update_state --stage preflight --status running --message "Verifying frozen artifacts and all SIF hashes"
container_cpu "${base_sif}" python "${script_dir}/preflight.py"

update_state --stage panels --status running --message "Reconstructing immutable canonical identity panels"
container_cpu "${base_sif}" python "${script_dir}/prepare_panels.py" --qualification-rows 128

update_state --stage validation_qualification --status running --message "Qualifying frozen adapters without locked-test tuning"
screen_panel validation "${benchmark_dir}/inputs/validation_qualification.tsv"
container_cpu "${base_sif}" python "${script_dir}/assemble_common.py" \
  --name validation --panel "${benchmark_dir}/inputs/validation_qualification.tsv"
for model in "${models[@]}"; do
  ensure_embedding "${model}" validation "${benchmark_dir}/inputs/common_validation.tsv" \
    "${benchmark_dir}/outputs/qualification"
done

update_state --stage locked_coverage --status running --message "Applying frozen coverage rules once to train-10k and locked-test-50k"
screen_panel train "${benchmark_dir}/inputs/train_10k.tsv"
screen_panel test "${benchmark_dir}/inputs/test_50k.tsv"
container_cpu "${base_sif}" python "${script_dir}/assemble_common.py" \
  --name train --panel "${benchmark_dir}/inputs/train_10k.tsv"
container_cpu "${base_sif}" python "${script_dir}/assemble_common.py" \
  --name test --panel "${benchmark_dir}/inputs/test_50k.tsv"

update_state --stage locked_encoding --status running --message "Exporting native FP32 representations on identical common identities"
for model in "${models[@]}"; do
  ensure_embedding "${model}" train "${benchmark_dir}/inputs/common_train.tsv" \
    "${benchmark_dir}/outputs/embeddings"
  ensure_embedding "${model}" test "${benchmark_dir}/inputs/common_test.tsv" \
    "${benchmark_dir}/outputs/embeddings"
done

update_state --stage payload_binding --status running --message "Binding native vectors to exact immutable identities"
container_cpu "${base_sif}" python "${script_dir}/package_payloads.py" --split train
container_cpu "${base_sif}" python "${script_dir}/package_payloads.py" --split test

update_state --stage common_probes --status running --message "Running unchanged common topology, geometry, neighbour, and clustering probes"
container_cpu "${base_sif}" python "${script_dir}/evaluate_payloads.py"

update_state --stage integrity --status running --message "Hashing outputs and sealing inference-only completion state"
container_cpu "${base_sif}" python "${script_dir}/finalize.py"
update_state --stage complete --status complete --message "Locked internal test-partition benchmark completed; MoleculeNet untouched"
