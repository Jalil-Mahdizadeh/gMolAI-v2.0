#!/usr/bin/env bash
#SBATCH --job-name=gmolai-clustering
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --output=extra-benchmark/clustering/logs/slurm-%j.out

set -euo pipefail

repository_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/gMolAI-retrain"
project_root="/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI"
benchmark_dir="${repository_root}/extra-benchmark/clustering"
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
  [morgan]=4096 [molai]=256 [molformer]=128 [smi_ted]=100
  [molclr_gin]=256 [kermt_v2]=64
)
screened_models=(morgan molai molformer smi_ted molclr_gin kermt_v2)
primary_models=(gmolai morgan molai molformer smi_ted molclr_gin kermt_v2)
all_representations=(gmolai morgan molai molformer smi_ted molclr_gin kermt_v2 morgan_count descriptor13)

cd "${repository_root}"
export APPTAINER_CACHEDIR="${project_root}/.apptainer-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
mkdir -p "${benchmark_dir}/artifacts/embeddings/classyfire" "${benchmark_dir}/artifacts/embeddings/qmugs"

container_cpu() {
  local image_path="$1"
  shift
  apptainer exec --cleanenv --bind "${project_root}:${project_root}" \
    --pwd "${repository_root}" "${image_path}" env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$@"
}

container_gpu() {
  local image_path="$1"
  shift
  local visible_gpu="${CUDA_VISIBLE_DEVICES:-0}"
  apptainer exec --cleanenv --nv --env "CUDA_VISIBLE_DEVICES=${visible_gpu}" \
    --env "CUBLAS_WORKSPACE_CONFIG=:4096:8" \
    --bind "${project_root}:${project_root}" --pwd "${repository_root}" \
    "${image_path}" env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 "$@"
}

echo "[preflight] verifying all frozen hashes"
container_cpu "${base_sif}" python "${script_dir}/preflight.py"
if [[ ! -f "${benchmark_dir}/audit/container_packages.json" ]]; then
  PYTHONDONTWRITEBYTECODE=1 python3 "${script_dir}/collect_container_versions.py"
fi

if [[ ! -f "${benchmark_dir}/audit/classyfire_preparation.json" ]]; then
  container_cpu "${base_sif}" python "${script_dir}/prepare_data.py" --dataset classyfire
fi
if [[ ! -f "${benchmark_dir}/audit/qmugs_preparation.json" ]]; then
  container_cpu "${base_sif}" python "${script_dir}/prepare_data.py" --dataset qmugs
fi

ensure_screen() {
  local model_name="$1"
  local panel_path="$2"
  local tag="$3"
  local report="${benchmark_dir}/artifacts/screens/${model_name}-${tag}.json"
  if [[ ! -f "${report}" ]]; then
    container_cpu "${images[${model_name}]}" python "${script_dir}/adapter.py" \
      --model "${model_name}" --mode screen --input "${panel_path}" \
      --output "${report}" --batch-size "${batches[${model_name}]}"
  fi
}

classy_candidates="${benchmark_dir}/inputs/prepared/classyfire_candidates.tsv"
for model_name in "${screened_models[@]}"; do
  ensure_screen "${model_name}" "${classy_candidates}" classyfire
done
if [[ ! -f "${benchmark_dir}/state/classyfire_common.json" ]]; then
  container_cpu "${base_sif}" python "${script_dir}/assemble_common.py" --benchmark classyfire
fi

if [[ ! -f "${benchmark_dir}/state/qmugs_common.json" ]]; then
  attempt_size=60000
  while true; do
    attempt_panel="${benchmark_dir}/inputs/prepared/qmugs_attempt_$(printf '%06d' "${attempt_size}").tsv"
    if [[ ! -f "${attempt_panel}" ]]; then
      container_cpu "${base_sif}" python "${script_dir}/make_qmugs_attempt.py" --size "${attempt_size}"
    fi
    tag="qmugs_$(printf '%06d' "${attempt_size}")"
    for model_name in "${screened_models[@]}"; do
      ensure_screen "${model_name}" "${attempt_panel}" "${tag}"
    done
    set +e
    container_cpu "${base_sif}" python "${script_dir}/assemble_common.py" \
      --benchmark qmugs --attempt-size "${attempt_size}"
    assemble_status=$?
    set -e
    if [[ "${assemble_status}" -eq 0 ]]; then
      break
    fi
    if [[ "${assemble_status}" -ne 3 ]]; then
      exit "${assemble_status}"
    fi
    attempt_size=$((attempt_size + 10000))
  done
fi

validate_embedding() {
  local model_name="$1"
  local benchmark_name="$2"
  local panel="${benchmark_dir}/inputs/prepared/${benchmark_name}_common.tsv"
  local directory="${benchmark_dir}/artifacts/embeddings/${benchmark_name}"
  container_cpu "${base_sif}" python "${script_dir}/validate_embedding.py" \
    --model "${model_name}" --benchmark "${benchmark_name}" --input "${panel}" \
    --embedding "${directory}/${model_name}.npy" --metadata "${directory}/${model_name}.json"
}

ensure_embedding() {
  local model_name="$1"
  local benchmark_name="$2"
  local panel="${benchmark_dir}/inputs/prepared/${benchmark_name}_common.tsv"
  local directory="${benchmark_dir}/artifacts/embeddings/${benchmark_name}"
  local matrix="${directory}/${model_name}.npy"
  local metadata="${directory}/${model_name}.json"
  if [[ -f "${matrix}" && -f "${metadata}" ]]; then
    validate_embedding "${model_name}" "${benchmark_name}"
    return
  fi
  if [[ -e "${matrix}" || -e "${metadata}" ]]; then
    echo "Incomplete embedding pair for ${model_name}/${benchmark_name}" >&2
    return 2
  fi
  if [[ "${model_name}" == gmolai ]]; then
    container_gpu "${base_sif}" python "${script_dir}/gmolai_adapter.py" \
      --input "${panel}" --output "${matrix}" --metadata "${metadata}"
  elif [[ "${model_name}" == morgan_count || "${model_name}" == descriptor13 ]]; then
    container_cpu "${base_sif}" python "${script_dir}/diagnostic_adapter.py" \
      --kind "${model_name}" --input "${panel}" --output "${matrix}" --metadata "${metadata}"
  elif [[ "${model_name}" == morgan ]]; then
    container_cpu "${base_sif}" python "${script_dir}/adapter.py" --model morgan \
      --mode encode --input "${panel}" --output "${matrix}" --metadata "${metadata}" \
      --batch-size "${batches[morgan]}"
  else
    container_gpu "${images[${model_name}]}" python "${script_dir}/adapter.py" \
      --model "${model_name}" --mode encode --input "${panel}" \
      --output "${matrix}" --metadata "${metadata}" --batch-size "${batches[${model_name}]}"
  fi
  validate_embedding "${model_name}" "${benchmark_name}"
}

for model_name in "${all_representations[@]}"; do
  ensure_embedding "${model_name}" classyfire
  ensure_embedding "${model_name}" qmugs
done
container_cpu "${base_sif}" python "${script_dir}/consolidate_coverage.py"

if [[ ! -f "${benchmark_dir}/audit/pretraining_exposure.json" ]]; then
  container_cpu "${base_sif}" python "${script_dir}/audit_exposure.py"
fi
if [[ ! -f "${benchmark_dir}/state/property_reference.json" ]]; then
  container_gpu "${base_sif}" python "${script_dir}/prepare_property_reference.py"
fi
for model_name in "${all_representations[@]}"; do
  if [[ ! -f "${benchmark_dir}/state/structural/${model_name}.json" ]]; then
    container_gpu "${base_sif}" python "${script_dir}/evaluate_structural_model.py" --model "${model_name}"
  fi
  if [[ ! -f "${benchmark_dir}/state/property/${model_name}.json" ]]; then
    container_gpu "${base_sif}" python "${script_dir}/evaluate_property_model.py" --model "${model_name}"
  fi
done

container_cpu "${base_sif}" python "${script_dir}/summarize_structural.py"
container_cpu "${base_sif}" python "${script_dir}/summarize_property.py"
container_cpu "${base_sif}" python "${script_dir}/make_pca_source.py"
container_cpu "${base_sif}" python "${script_dir}/make_figures.py"
container_cpu "${base_sif}" python "${script_dir}/render_results.py"
container_cpu "${base_sif}" python "${script_dir}/verify_results.py"
container_gpu "${base_sif}" python "${script_dir}/finalize.py"

echo "Benchmark complete: ${benchmark_dir}/state/COMPLETE.json"
