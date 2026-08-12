#!/usr/bin/env bash
#SBATCH --job-name=gmolai-extra-probes
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
models=(morgan molai molformer smi_ted molclr_gin kermt_v2)

cd "${repository_root}"
export APPTAINER_CACHEDIR="${project_root}/.apptainer-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

container_cpu() {
  apptainer exec --cleanenv \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" \
    --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" \
    "${base_sif}" \
    env PYTHONUNBUFFERED=1 "$@"
}

update_state() {
  container_cpu python "${script_dir}/state_update.py" "$@"
}

record_failure() {
  local exit_code="$?"
  update_state --stage recovery_failed --status failed \
    --message "Post-processing recovery stopped with exit code ${exit_code}" || true
  exit "${exit_code}"
}
trap record_failure ERR

update_state --stage recovery_validation --status running \
  --message "Validating frozen sources and existing native embeddings; no encoding"
container_cpu python "${script_dir}/preflight.py"
for model in "${models[@]}"; do
  for split in train test; do
    container_cpu python "${script_dir}/validate_adapter_output.py" \
      --model "${model}" \
      --input "${benchmark_dir}/inputs/common_${split}.tsv" \
      --embedding "${benchmark_dir}/outputs/embeddings/${model}-${split}.npy" \
      --metadata "${benchmark_dir}/outputs/embeddings/${model}-${split}.json"
  done
done

update_state --stage recovery_payload_binding --status running \
  --message "Restoring authoritative hybrid metadata in derived gMolAI payloads"
container_cpu python "${script_dir}/package_payloads.py" \
  --split train --repair-derived-metadata
container_cpu python "${script_dir}/package_payloads.py" \
  --split test --repair-derived-metadata

update_state --stage common_probes --status running \
  --message "Running topology, geometry, neighbour and clustering probes"
container_cpu python "${script_dir}/evaluate_payloads.py"

update_state --stage integrity --status running \
  --message "Hashing outputs and sealing inference-only completion state"
container_cpu python "${script_dir}/finalize.py"
update_state --stage complete --status complete \
  --message "Locked internal test-partition benchmark completed; MoleculeNet untouched"
