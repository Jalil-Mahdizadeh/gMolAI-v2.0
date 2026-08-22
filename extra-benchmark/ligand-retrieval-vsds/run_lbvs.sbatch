#!/usr/bin/env bash
#SBATCH --job-name=gmolai-lbvs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=extra-benchmark/ligand-retrieval-vsds/logs/%x-%j.out

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${BENCHMARK_DIR}/../.." && pwd)"
PROJECT_ROOT="$(dirname "${REPOSITORY_ROOT}")"
CONTAINER_ROOT="${PROJECT_ROOT}/containers"
BASE_SIF="${CONTAINER_ROOT}/gmolai-pyg-25.09-arm64.sif"
GPU_VISIBLE="${CUDA_VISIBLE_DEVICES:-0}"
SOURCE_PANEL="${BENCHMARK_DIR}/inputs/prepared/molecule_candidates.tsv"
SCREEN_DIR="${BENCHMARK_DIR}/artifacts/screens"
EMBEDDING_DIR="${BENCHMARK_DIR}/embeddings/model-panels"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "${REPOSITORY_ROOT}"
mkdir -p "${BENCHMARK_DIR}/logs" "${SCREEN_DIR}" "${EMBEDDING_DIR}"

run_cpu() {
    apptainer exec --cleanenv \
        --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --pwd "${REPOSITORY_ROOT}" \
        "${BASE_SIF}" \
        env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
        "$@"
}

run_in_container() {
    local sif="$1"
    shift
    apptainer exec --cleanenv \
        --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --pwd "${REPOSITORY_ROOT}" \
        "${sif}" \
        env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
        "$@"
}

run_gpu() {
    local sif="$1"
    shift
    apptainer exec --cleanenv --nv \
        --env "CUDA_VISIBLE_DEVICES=${GPU_VISIBLE}" \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --pwd "${REPOSITORY_ROOT}" \
        "${sif}" \
        env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
        "$@"
}

stage_valid() {
    local manifest="$1"
    [[ -f "${manifest}" ]] && run_cpu python "${BENCHMARK_DIR}/scripts/stage_status.py" "${manifest}" >/dev/null 2>&1
}

refuse_if_present() {
    local later="$1"
    local message="$2"
    if [[ -e "${later}" ]]; then
        echo "Refusing to regenerate an invalid upstream stage after ${later}: ${message}" >&2
        exit 1
    fi
}

if stage_valid "${BENCHMARK_DIR}/state/COMPLETE.json"; then
    echo "Verified completed benchmark; nothing to rerun."
    exit 0
fi

if stage_valid "${BENCHMARK_DIR}/audits/preflight.json"; then
    echo "Reusing verified preflight."
else
    refuse_if_present "${BENCHMARK_DIR}/audits/data_preparation.json" "restore the frozen protocol/source environment first"
    run_cpu python "${BENCHMARK_DIR}/scripts/preflight.py"
fi

if stage_valid "${BENCHMARK_DIR}/audits/data_preparation.json"; then
    echo "Reusing verified data preparation."
else
    refuse_if_present "${BENCHMARK_DIR}/state/SCREENING_COMPLETE.json" "prepared identities may not change after screening"
    SOURCE_ROOT="${BENCHMARK_DIR}/inputs/raw/extracted/VSDS_vd/TrueDecoy set"
    if [[ ! -d "${SOURCE_ROOT}" ]]; then
        EXTRACT_TMP="$(mktemp -d "${BENCHMARK_DIR}/inputs/raw/extract.XXXXXX")"
        bsdtar -xf "${BENCHMARK_DIR}/inputs/raw/VSDS_vd-v3.rar" -C "${EXTRACT_TMP}"
        mkdir -p "${BENCHMARK_DIR}/inputs/raw/extracted"
        mv "${EXTRACT_TMP}/VSDS_vd" "${BENCHMARK_DIR}/inputs/raw/extracted/VSDS_vd"
        rmdir "${EXTRACT_TMP}"
    fi
    run_cpu python "${BENCHMARK_DIR}/scripts/prepare_data.py"
fi

if stage_valid "${BENCHMARK_DIR}/state/SCREENING_COMPLETE.json"; then
    echo "Reusing verified label-blind screens and model panels."
else
    refuse_if_present "${EMBEDDING_DIR}/gmolai.npy" "model panels may not change after representation export"
    if ! stage_valid "${SCREEN_DIR}/gmolai.json"; then
        run_cpu python "${BENCHMARK_DIR}/scripts/screen_gmolai.py" \
            --input "${SOURCE_PANEL}" --output "${SCREEN_DIR}/gmolai.json"
    fi
    declare -A SCREEN_SIFS=(
        [morgan]="${BASE_SIF}"
        [molai]="${CONTAINER_ROOT}/MolAI/molai-epoch6-arm64.sif"
        [molformer]="${CONTAINER_ROOT}/MolFormer/molformer-xl-both-10pct-arm64.sif"
        [smi_ted]="${CONTAINER_ROOT}/SMI-TED/smi-ted-light-arm64.sif"
        [molclr_gin]="${CONTAINER_ROOT}/MolCLR-GIN/molclr-gin-arm64.sif"
        [kermt_v2]="${CONTAINER_ROOT}/KERMT-v2/kermt-v2-arm64.sif"
    )
    declare -A BATCHES=(
        [morgan]=4096 [molai]=256 [molformer]=128 [smi_ted]=100 [molclr_gin]=256 [kermt_v2]=64
    )
    for model in morgan molai molformer smi_ted molclr_gin kermt_v2; do
        if ! stage_valid "${SCREEN_DIR}/${model}.json"; then
            run_in_container "${SCREEN_SIFS[${model}]}" \
                python "${BENCHMARK_DIR}/scripts/adapter.py" \
                --model "${model}" --mode screen --input "${SOURCE_PANEL}" \
                --output "${SCREEN_DIR}/${model}.json" --batch-size "${BATCHES[${model}]}"
        fi
    done
    run_cpu python "${BENCHMARK_DIR}/scripts/make_model_panels.py"
fi

declare -A MODEL_SIFS=(
    [gmolai]="${BASE_SIF}"
    [morgan]="${BASE_SIF}"
    [molai]="${CONTAINER_ROOT}/MolAI/molai-epoch6-arm64.sif"
    [molformer]="${CONTAINER_ROOT}/MolFormer/molformer-xl-both-10pct-arm64.sif"
    [smi_ted]="${CONTAINER_ROOT}/SMI-TED/smi-ted-light-arm64.sif"
    [molclr_gin]="${CONTAINER_ROOT}/MolCLR-GIN/molclr-gin-arm64.sif"
    [kermt_v2]="${CONTAINER_ROOT}/KERMT-v2/kermt-v2-arm64.sif"
)
declare -A MODEL_BATCHES=(
    [morgan]=4096 [molai]=256 [molformer]=128 [smi_ted]=100 [molclr_gin]=256 [kermt_v2]=64
)

for model in gmolai morgan molai molformer smi_ted molclr_gin kermt_v2; do
    matrix="${EMBEDDING_DIR}/${model}.npy"
    metadata="${EMBEDDING_DIR}/${model}.json"
    if [[ -e "${matrix}" || -e "${metadata}" ]]; then
        if [[ ! -f "${matrix}" || ! -f "${metadata}" ]]; then
            echo "Refusing incomplete representation pair for ${model}." >&2
            exit 1
        fi
        run_cpu python "${BENCHMARK_DIR}/scripts/validate_embedding.py" --model "${model}"
        continue
    fi
    panel="${BENCHMARK_DIR}/inputs/prepared/model_panels/${model}.tsv"
    if [[ "${model}" == "gmolai" ]]; then
        run_gpu "${MODEL_SIFS[${model}]}" \
            python "${BENCHMARK_DIR}/scripts/gmolai_adapter.py" \
            --input "${panel}" --output "${matrix}" --metadata "${metadata}"
    elif [[ "${model}" == "morgan" ]]; then
        run_cpu python "${BENCHMARK_DIR}/scripts/adapter.py" \
            --model "${model}" --mode encode --input "${panel}" \
            --output "${matrix}" --metadata "${metadata}" --batch-size "${MODEL_BATCHES[${model}]}"
    else
        run_gpu "${MODEL_SIFS[${model}]}" \
            python "${BENCHMARK_DIR}/scripts/adapter.py" \
            --model "${model}" --mode encode --input "${panel}" \
            --output "${matrix}" --metadata "${metadata}" --batch-size "${MODEL_BATCHES[${model}]}"
    fi
    run_cpu python "${BENCHMARK_DIR}/scripts/validate_embedding.py" --model "${model}"
done

if stage_valid "${BENCHMARK_DIR}/state/POPULATION_FROZEN.json"; then
    echo "Reusing verified common-support population freeze."
else
    refuse_if_present "${BENCHMARK_DIR}/state/ANCHORS_FROZEN.json" "population may not change after anchor generation"
    run_cpu python "${BENCHMARK_DIR}/scripts/freeze_population.py"
fi

if stage_valid "${BENCHMARK_DIR}/state/ANCHORS_FROZEN.json"; then
    echo "Reusing verified anchor schedule."
else
    refuse_if_present "${BENCHMARK_DIR}/state/RETRIEVAL_COMPLETE.json" "anchors may not change after retrieval"
    run_cpu python "${BENCHMARK_DIR}/scripts/freeze_anchors.py"
fi

if ! stage_valid "${BENCHMARK_DIR}/audits/pretraining_exposure.json"; then
    run_cpu python "${BENCHMARK_DIR}/scripts/audit_exposure.py"
fi

if stage_valid "${BENCHMARK_DIR}/state/RETRIEVAL_COMPLETE.json"; then
    echo "Reusing verified per-draw retrieval output."
else
    refuse_if_present "${BENCHMARK_DIR}/state/SUMMARY_COMPLETE.json" "per-draw results may not change after aggregation"
    run_cpu python "${BENCHMARK_DIR}/scripts/evaluate.py"
fi

if stage_valid "${BENCHMARK_DIR}/state/SUMMARY_COMPLETE.json"; then
    echo "Reusing verified target-level summaries."
else
    refuse_if_present "${BENCHMARK_DIR}/audits/figure_manifest.json" "summaries may not change after figure generation"
    run_cpu python "${BENCHMARK_DIR}/scripts/summarize.py"
fi

if ! stage_valid "${BENCHMARK_DIR}/audits/figure_manifest.json"; then
    run_cpu python "${BENCHMARK_DIR}/scripts/make_figures.py"
fi
if ! stage_valid "${BENCHMARK_DIR}/state/REPORT_COMPLETE.json"; then
    run_cpu python "${BENCHMARK_DIR}/scripts/render_results.py"
fi
run_cpu python "${BENCHMARK_DIR}/scripts/verify_results.py"
run_cpu python "${BENCHMARK_DIR}/scripts/finalize.py"
