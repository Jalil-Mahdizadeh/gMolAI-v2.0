#!/usr/bin/env bash
#SBATCH --job-name=gmolai-orchestrate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
source "${CODE_DIR}/slurm/common.sh"

# This job runs on an arm64 GH200 node, so the arm64 SIF can safely prepare the
# immutable task map. Child jobs inherit the exported deployment configuration.
container_login python -m gmolai_retrain.cli --config "${CONFIG_PATH}" prepare
PREPARED_JSON=$(container_login python -c "from gmolai_retrain.config import load_config; import pathlib; c=load_config('${CONFIG_PATH}'); print(pathlib.Path(c['paths']['work_dir'],'prepared.json'))" | tail -n 1)
CANONICAL_TASKS=$(container_login python -c "import json; print(json.load(open('${PREPARED_JSON}'))['canonicalize_task_count'])" | tail -n 1)
HASH_BUCKETS=$(container_login python -c "import json; print(json.load(open('${PREPARED_JSON}'))['hash_buckets'])" | tail -n 1)
(( CANONICAL_TASKS > 0 )) || { echo "No canonicalization tasks" >&2; exit 2; }
(( HASH_BUCKETS > 1 )) || { echo "Invalid hash bucket count" >&2; exit 2; }

export NAISS_PROJECT PROJECT_ROOT CODE_DIR SIF_PATH CONFIG_PATH GMOLAI_EXTRA_BINDS
verify_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL "${CODE_DIR}/slurm/05_verify_inputs.sbatch")
canonical_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${verify_job}" \
  --array="0-$((CANONICAL_TASKS-1))%${CANONICAL_MAX_CONCURRENT:-4}" "${CODE_DIR}/slurm/10_canonicalize.sbatch")
dedup_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${canonical_job}" \
  --array="0-$((HASH_BUCKETS-1))%${DEDUP_MAX_CONCURRENT:-4}" "${CODE_DIR}/slurm/20_deduplicate.sbatch")
data_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${dedup_job}" "${CODE_DIR}/slurm/30_finalize_data.sbatch")
feature_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${data_job}" \
  --array="0-$((HASH_BUCKETS-1))%${FEATURIZE_MAX_CONCURRENT:-4}" "${CODE_DIR}/slurm/40_featurize.sbatch")
graphs_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${feature_job}" "${CODE_DIR}/slurm/50_finalize_graphs.sbatch")
if [[ "${AUTO_SUBMIT_TRAIN:-0}" == "1" ]]; then
  train_job=$(sbatch --parsable -A "${NAISS_PROJECT}" --export=ALL --dependency="afterok:${graphs_job}" "${CODE_DIR}/slurm/60_train.sbatch")
else
  train_job="not-submitted"
fi

printf 'verify=%s\ncanonicalize=%s\ndeduplicate=%s\nfinalize_data=%s\nfeaturize=%s\nfinalize_graphs=%s\ntrain=%s\n' \
  "${verify_job}" "${canonical_job}" "${dedup_job}" "${data_job}" "${feature_job}" "${graphs_job}" "${train_job}"
if [[ "${train_job}" == "not-submitted" ]]; then
  echo "After reviewing work/dataset_manifest.json, work/descriptor_scaler.json, and work/graph_manifest.json:"
  echo "  sbatch -A '${NAISS_PROJECT}' --export=ALL --dependency='afterok:${graphs_job}' '${CODE_DIR}/slurm/60_train.sbatch'"
fi
