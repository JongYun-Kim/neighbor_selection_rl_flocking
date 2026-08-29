#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

image_name="${IMAGE_NAME:-uom-neighbor-selection}"
container_name="${CONTAINER_NAME:-dynamic-k-nn-training}"
results_root="${RESULTS_ROOT:-${repo_root}/test_results}"
wandb_key_file="${WANDB_API_KEY_FILE_HOST:-${HOME}/.config/wandb/api_key}"
shm_size="${SHM_SIZE:-24g}"
gpu_request="${GPU_REQUEST:-all}"
# W&B is opt-in repo-wide (train_unified.py defaults it off); keep the same
# default here so a fresh clone starts without a key file.
wandb_enabled="${WANDB_ENABLED:-false}"

usage() {
    cat <<'EOF'
Usage: ./docker/run_train.sh <command> [options]

Commands:
  start [--build] [--run-id ID] [--dry-run]
      Start a durable background training container.
  status
      Show container state, result location, and latest Tune progress.
  logs [--no-follow]
      Follow training logs, or print the latest logs once.
  stop
      Explicitly stop the container. Docker will not restart an explicit stop.

Recipe (train_unified.py):
  FLOCK_PROFILE (dknn | pi_r | dknn_c2; default dknn), FLOCK_SEEDS,
  FLOCK_GPU, FLOCK_RESUME (default 1)

Environment overrides:
  IMAGE_NAME, CONTAINER_NAME, RESULTS_ROOT, WANDB_ENABLED (default false),
  WANDB_API_KEY_FILE_HOST,
  SHM_SIZE, GPU_REQUEST, BASE_ENV_SEED, TRAINING_SWARM_SIZE,
  NUM_ROLLOUT_WORKERS, NUM_ENVS_PER_WORKER, ROLLOUT_FRAGMENT_LENGTH,
  TOTAL_TRAINING_TIMESTEPS, MAX_TRAINING_TIME_S, TRAIN_BATCH_SIZE,
  SGD_MINIBATCH_SIZE, NUM_SGD_ITER, INITIAL_LR, FINAL_LR,
  CHECKPOINT_FREQ, KEEP_CHECKPOINTS_NUM, TUNE_EXPERIMENT_NAME, WANDB_ENABLED,
  WANDB_PROJECT, WANDB_RUN_NAME, TRAIN_ENTRY
EOF
}

absolute_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        realpath -m -- "${value}"
    else
        realpath -m -- "$(pwd -P)/${value}"
    fi
}

require_docker() {
    command -v docker >/dev/null 2>&1 || {
        echo "[run] docker CLI was not found" >&2
        exit 1
    }
}

show_status() {
    require_docker
    if ! docker container inspect "${container_name}" >/dev/null 2>&1; then
        echo "container=${container_name} status=not-created"
        return 1
    fi

    docker container inspect "${container_name}" \
        --format 'container={{.Name}} status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}} started={{.State.StartedAt}}'

    local run_id host_results
    run_id="$(docker container inspect "${container_name}" --format '{{index .Config.Labels "dynamic-k-nn.run-id"}}')"
    host_results="$(docker container inspect "${container_name}" --format '{{index .Config.Labels "dynamic-k-nn.results-dir"}}')"
    if [ -z "${run_id}" ] || [ -z "${host_results}" ]; then
        echo "Container ${container_name} was not created by this runner" >&2
        return 1
    fi
    echo "run_id=${run_id}"
    echo "results=${host_results}"

    if [ -f "${host_results}/.training_complete" ]; then
        echo "training=completed completed_at_utc=$(<"${host_results}/.training_complete")"
    else
        echo "training=incomplete"
    fi

    python3 - "${host_results}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
progress_files = sorted(root.glob("**/progress.csv"), key=lambda p: p.stat().st_mtime)
if not progress_files:
    raise SystemExit(0)
with progress_files[-1].open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(0)
row = rows[-1]
fields = (
    "training_iteration",
    "timesteps_total",
    "episode_reward_mean",
    "time_total_s",
    "time_this_iter_s",
)
print("latest_progress=" + " ".join(f"{key}={row.get(key, '')}" for key in fields))
PY
}

start_training() {
    local build_image=0
    local dry_run=0
    local run_id=""

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --build)
                build_image=1
                ;;
            --dry-run)
                dry_run=1
                ;;
            --run-id)
                shift
                [ "$#" -gt 0 ] || { echo "--run-id requires a value" >&2; exit 2; }
                run_id="$1"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown start option: $1" >&2
                exit 2
                ;;
        esac
        shift
    done

    run_id="${run_id:-dynamic-k-nn-$(date -u +%Y%m%dT%H%M%SZ)}"
    if ! [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
        echo "run ID must use only letters, digits, dot, underscore, and hyphen" >&2
        exit 2
    fi

    results_root="$(absolute_path "${results_root}")"
    wandb_key_file="$(absolute_path "${wandb_key_file}")"
    local run_results="${results_root}/${run_id}"

    local normalized_wandb="${wandb_enabled,,}"
    case "${normalized_wandb}" in
        1|true|yes|on)
            normalized_wandb=true
            ;;
        0|false|no|off)
            normalized_wandb=false
            ;;
        *)
            echo "WANDB_ENABLED must be a boolean, found: ${wandb_enabled}" >&2
            exit 2
            ;;
    esac

    local docker_args=(
        run
        --detach
        --init
        --gpus "${gpu_request}"
        --shm-size "${shm_size}"
        --restart unless-stopped
        --name "${container_name}"
        --workdir /workspace/source
        --label "dynamic-k-nn.run-id=${run_id}"
        --label "dynamic-k-nn.results-dir=${run_results}"
        --env START_SSHD=0
        --env "WORKFLOW_RUN_ID=${run_id}"
        --env TRAINING_RESULTS_DIR=/workspace/test_results
        --env "WANDB_ENABLED=${normalized_wandb}"
        --mount "type=bind,src=${repo_root},dst=/workspace/source,readonly"
        --mount "type=bind,src=${run_results},dst=/workspace/test_results"
    )

    if [ "${normalized_wandb}" = true ]; then
        docker_args+=(
            --env WANDB_API_KEY_FILE=/run/secrets/wandb_api_key
            --mount "type=bind,src=${wandb_key_file},dst=/run/secrets/wandb_api_key,readonly"
        )
    fi

    # train_unified.py reads its recipe from FLOCK_*; without these the
    # container silently falls back to the dknn defaults whatever the caller
    # asked for. The rest are the env-var knobs of the retired TRAIN_ENTRY.
    local override_names=(
        FLOCK_PROFILE FLOCK_SEEDS FLOCK_GPU FLOCK_RESUME
        BASE_ENV_SEED TRAINING_SWARM_SIZE NUM_ROLLOUT_WORKERS
        NUM_ENVS_PER_WORKER ROLLOUT_FRAGMENT_LENGTH TOTAL_TRAINING_TIMESTEPS
        MAX_TRAINING_TIME_S TRAIN_BATCH_SIZE SGD_MINIBATCH_SIZE NUM_SGD_ITER
        INITIAL_LR FINAL_LR CHECKPOINT_FREQ KEEP_CHECKPOINTS_NUM
        TUNE_EXPERIMENT_NAME WANDB_PROJECT WANDB_RUN_NAME TRAIN_ENTRY
    )
    local override_name
    for override_name in "${override_names[@]}"; do
        if [[ -v "${override_name}" ]]; then
            docker_args+=(--env "${override_name}=${!override_name}")
        fi
    done
    docker_args+=("${image_name}" bash docker/train_service.sh)

    if [ "${dry_run}" -eq 1 ]; then
        printf 'docker'
        printf ' %q' "${docker_args[@]}"
        printf '\n'
        return 0
    fi

    require_docker
    if docker container inspect "${container_name}" >/dev/null 2>&1; then
        echo "Container already exists: ${container_name}" >&2
        echo "Use status/logs/stop, or set a different CONTAINER_NAME." >&2
        exit 1
    fi
    if [ "${normalized_wandb}" = true ]; then
        if [ ! -s "${wandb_key_file}" ]; then
            echo "W&B API key file is missing or empty: ${wandb_key_file}" >&2
            exit 1
        fi
        if [ "$(stat -c '%a' "${wandb_key_file}")" != "600" ]; then
            echo "W&B API key file must have mode 600" >&2
            exit 1
        fi
    fi

    if [ "${build_image}" -eq 1 ] || ! docker image inspect "${image_name}" >/dev/null 2>&1; then
        IMAGE_NAME="${image_name}" "${script_dir}/build.sh"
    fi
    mkdir -p "${run_results}"

    docker "${docker_args[@]}"
    echo "[run] started container=${container_name} run_id=${run_id}"
    echo "[run] entry=${TRAIN_ENTRY:-train_unified.py}" \
         "profile=${FLOCK_PROFILE:-dknn} seeds=${FLOCK_SEEDS:-42}" \
         "gpus=${gpu_request} wandb=${normalized_wandb}"
    echo "[run] status: ./docker/run_train.sh status"
    echo "[run] logs:   ./docker/run_train.sh logs"
}

command="${1:-}"
if [ -z "${command}" ]; then
    usage
    exit 2
fi
shift

case "${command}" in
    start)
        start_training "$@"
        ;;
    status)
        [ "$#" -eq 0 ] || { echo "status takes no arguments" >&2; exit 2; }
        show_status
        ;;
    logs)
        require_docker
        if [ "${1:-}" = "--no-follow" ]; then
            [ "$#" -eq 1 ] || { echo "logs accepts only --no-follow" >&2; exit 2; }
            docker logs --tail 200 "${container_name}"
        elif [ "$#" -eq 0 ]; then
            docker logs --tail 200 --follow "${container_name}"
        else
            echo "logs accepts only --no-follow" >&2
            exit 2
        fi
        ;;
    stop)
        [ "$#" -eq 0 ] || { echo "stop takes no arguments" >&2; exit 2; }
        require_docker
        docker stop "${container_name}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: ${command}" >&2
        usage >&2
        exit 2
        ;;
esac
