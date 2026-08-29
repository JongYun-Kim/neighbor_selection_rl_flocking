#!/usr/bin/env bash
set -Eeuo pipefail

training_results_dir="${TRAINING_RESULTS_DIR:-/workspace/test_results}"
workflow_run_id="${WORKFLOW_RUN_ID:-dynamic-k-nn}"
# Selectable trainer entry point. The default is the unified trainer, whose
# recipe is chosen with FLOCK_PROFILE (default dknn = the ck848 original
# recipe, 8M steps); TRAIN_ENTRY overrides the script itself.
train_entry="${TRAIN_ENTRY:-train_unified.py}"
# Docker restarts this container on failure, and the host may reboot mid-run.
# Without resume the trainer would start a fresh Tune trial each time and lose
# every completed step; FLOCK_RESUME=1 makes it rejoin the existing trial from
# its last checkpoint (Tune AUTO+ERRORED). Export FLOCK_RESUME=0 to opt out.
export FLOCK_RESUME="${FLOCK_RESUME:-1}"
completion_marker="${training_results_dir}/.training_complete"

mkdir -p "${training_results_dir}"

if [ -f "${completion_marker}" ]; then
    echo "[training-service] run=${workflow_run_id} already completed; staying idle"
    exec sleep infinity
fi

echo "[training-service] run=${workflow_run_id} entry=${train_entry} results=${training_results_dir}"
set +e
python -u "${train_entry}"
training_exit_code=$?
set -e

if [ "${training_exit_code}" -ne 0 ]; then
    echo "[training-service] training exited with code ${training_exit_code}; Docker will restart the container" >&2
    exit "${training_exit_code}"
fi

temporary_marker="${completion_marker}.tmp"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${temporary_marker}"
mv "${temporary_marker}" "${completion_marker}"
echo "[training-service] training complete; staying idle"
exec sleep infinity
