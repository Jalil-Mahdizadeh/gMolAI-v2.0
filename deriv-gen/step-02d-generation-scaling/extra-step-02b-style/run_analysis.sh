#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
analysis_root=$(cd -- "${script_dir}" && pwd -P)
repo_root=$(cd -- "${analysis_root}/../../.." && pwd -P)
expected_root="${repo_root}/deriv-gen/step-02d-generation-scaling/extra-step-02b-style"
container=/nobackup/proj/disk/theo-storage/personal/jalil/gMolAI/containers/gmolai-pyg-25.09-arm64.sif

if [[ "${analysis_root}" != "${expected_root}" ]]; then
  echo "Refusing unexpected analysis root: ${analysis_root}" >&2
  exit 2
fi
if [[ ! -f "${container}" ]]; then
  echo "Missing frozen runtime image: ${container}" >&2
  exit 2
fi
for required in \
  config/analysis.json \
  PROTOCOL.md \
  scripts/common.py \
  scripts/analysis_core.py \
  scripts/test_components.py \
  scripts/prepare_inputs.py \
  scripts/analyze_results.py \
  scripts/plot_results.py \
  scripts/verify_results.py; do
  if [[ ! -f "${analysis_root}/${required}" ]]; then
    echo "Missing analysis implementation: ${required}" >&2
    exit 2
  fi
done

mkdir -p \
  "${analysis_root}/inputs" \
  "${analysis_root}/intermediate" \
  "${analysis_root}/outputs/tables" \
  "${analysis_root}/outputs/plot-data" \
  "${analysis_root}/outputs/figures" \
  "${analysis_root}/logs" \
  "${analysis_root}/state/home" \
  "${analysis_root}/state/matplotlib" \
  "${analysis_root}/state/xdg-cache" \
  "${analysis_root}/state/tmp" \
  "${analysis_root}/state/pycache" \
  "${analysis_root}/state/apptainer-cache" \
  "${analysis_root}/state/apptainer-tmp" \
  "${analysis_root}/state/duckdb_tmp"

export APPTAINER_CACHEDIR="${analysis_root}/state/apptainer-cache"
export APPTAINER_TMPDIR="${analysis_root}/state/apptainer-tmp"
log_path="${analysis_root}/logs/run-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${log_path}") 2>&1

base_command=(
  apptainer exec
  --cleanenv
  --containall
  --home "${analysis_root}/state/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${analysis_root}:/analysis:rw"
  --pwd /repo
  --env PYTHONPATH=/analysis/scripts:/repo/src
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPYCACHEPREFIX=/analysis/state/pycache
  --env XDG_CACHE_HOME=/analysis/state/xdg-cache
  --env MPLCONFIGDIR=/analysis/state/matplotlib
  --env TMPDIR=/analysis/state/tmp
  --env OMP_NUM_THREADS=8
  "${container}"
)

gpu_command=(
  apptainer exec
  --nv
  --cleanenv
  --containall
  --home "${analysis_root}/state/home"
  --bind "${repo_root}:/repo:ro"
  --bind "${analysis_root}:/analysis:rw"
  --pwd /repo
  --env PYTHONPATH=/analysis/scripts:/repo/src
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONPYCACHEPREFIX=/analysis/state/pycache
  --env XDG_CACHE_HOME=/analysis/state/xdg-cache
  --env MPLCONFIGDIR=/analysis/state/matplotlib
  --env TMPDIR=/analysis/state/tmp
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env OMP_NUM_THREADS=8
  "${container}"
)

echo "Analysis root: ${analysis_root}"
echo "Repository mounted read-only at /repo; analysis mounted read-write at /analysis"

if [[ ! -f "${analysis_root}/state/COMPONENT_TESTS.json" ]]; then
  "${base_command[@]}" python /analysis/scripts/test_components.py --analysis-root /analysis
fi
"${base_command[@]}" python /analysis/scripts/prepare_inputs.py --repo-root /repo --analysis-root /analysis

embedding="${analysis_root}/intermediate/candidate_embeddings.npz"
embedding_metadata="${analysis_root}/intermediate/candidate_embeddings.metadata.json"
embedding_rejections="${analysis_root}/intermediate/candidate_embeddings.rejections.csv"
existing_embedding_outputs=0
for path in "${embedding}" "${embedding_metadata}" "${embedding_rejections}"; do
  if [[ -f "${path}" ]]; then
    existing_embedding_outputs=$((existing_embedding_outputs + 1))
  fi
done
if [[ "${existing_embedding_outputs}" -ne 0 && "${existing_embedding_outputs}" -ne 3 ]]; then
  echo "Partial re-encoding outputs exist; refusing to overwrite them" >&2
  exit 2
fi
if [[ "${existing_embedding_outputs}" -eq 0 ]]; then
  "${gpu_command[@]}" python /repo/inference/gmolai.py encode \
    --models-dir /repo/inference/models \
    --device cuda \
    --threads 8 \
    --input /analysis/inputs/encoder_input.csv \
    --output /analysis/intermediate/candidate_embeddings.npz \
    --smiles-column smiles \
    --id-column molecule_id \
    --backend optimized \
    --batch-size 512 \
    --node-budget 65536 \
    --workers 48 \
    --invalid-policy error
fi

"${base_command[@]}" python /analysis/scripts/analyze_results.py --repo-root /repo --analysis-root /analysis
"${base_command[@]}" python /analysis/scripts/plot_results.py --analysis-root /analysis
"${base_command[@]}" python /analysis/scripts/verify_results.py --repo-root /repo --analysis-root /analysis
echo "Verified analysis complete: ${analysis_root}/outputs/verification.json"
