#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/repo}"
STEP_ROOT="${REPO_ROOT}/deriv-gen/step-02b-candidate-reranking"

python "${STEP_ROOT}/scripts/register_study.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/prepare_panels.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/test_components.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/evaluate_panel.py" --phase development --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/freeze_policy.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/evaluate_panel.py" --phase final --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/report_results.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
python "${STEP_ROOT}/scripts/verify_study.py" --repo-root "${REPO_ROOT}" --step-root "${STEP_ROOT}"
