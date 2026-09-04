#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

# Keep evaluation builds on a separate tag so ``--build`` cannot retag the
# image another collaborator's durable training runner expects to use.
image_name="${IMAGE_NAME:-uom-neighbor-selection-eval}"
checkpoint_root="${CHECKPOINT_ROOT:-${repo_root}/checkpoints}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/test_results/evaluation}"
device="${DEVICE:-cpu}"
gpu_request="${GPU_REQUEST:-}"

usage() {
    cat <<'EOF'
Usage:
  ./docker/run_eval.sh [--build] [--dry-run] <arguments after "python -m eval">

Examples:
  ./docker/run_eval.sh checkpoints --run-dir /workspace/checkpoints/my_run \
      --out /workspace/artifacts/checkpoints.json
  ./docker/run_eval.sh c2 --lane dev \
      --candidates /workspace/artifacts/checkpoints.json --run-id experiment-dev
  ./docker/run_eval.sh population --run-id population-best
  ./docker/run_eval.sh sensitivity run \
      --config eval/sensitivity/configs/oat.yaml --run-id sensitivity-oat

The first non-runner argument begins the eval command. Consequently, a
--dry-run before the command previews Docker without running it, while a
--dry-run after (for example) "population" is passed to the eval CLI.

Runner options:
  --build       Build IMAGE_NAME before the one-shot evaluation.
  --dry-run     Print shell-escaped build/run commands without changing state.
  -h, --help    Show this help.

Common eval commands:
  checkpoints, c2, population, sensitivity, validate, radii, heatmaps,
  control-effort

Environment overrides:
  IMAGE_NAME       Docker image (default: uom-neighbor-selection-eval)
  CHECKPOINT_ROOT  Host checkpoint tree, mounted read-only
                   (default model: dynamic_k_nn_best/checkpoint_000592)
  ARTIFACT_ROOT    Host artifact tree, mounted read-write
  CONTAINER_NAME   Optional container name; otherwise a unique name is made
  DEVICE           cpu (default) or cuda[:index]; CUDA population/sensitivity
                   commands require an explicit --batch-size
  GPU_REQUEST      Docker --gpus value; setting it explicitly requests a GPU
EOF
}

fail() {
    echo "[eval-run] $*" >&2
    exit 2
}

absolute_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        realpath -m -- "${value}"
    else
        realpath -m -- "$(pwd -P)/${value}"
    fi
}

validate_mount_path() {
    local label="$1"
    local value="$2"
    [[ "${value}" = /* ]] || fail "${label} did not resolve to an absolute path: ${value}"
    [[ "${value}" != / ]] || fail "${label} may not be the filesystem root"
    case "${value}" in
        *','*|*$'\n'*|*$'\r'*)
            fail "${label} contains a character unsupported by Docker --mount: ${value}"
            ;;
    esac
}

require_docker() {
    command -v docker >/dev/null 2>&1 || {
        echo "[eval-run] docker CLI was not found" >&2
        exit 1
    }
}

build_image=0
docker_dry_run=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --build)
            build_image=1
            shift
            ;;
        --dry-run)
            docker_dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

[ "$#" -gt 0 ] || {
    usage >&2
    exit 2
}
eval_args=("$@")

checkpoint_root="$(absolute_path "${checkpoint_root}")"
artifact_root="$(absolute_path "${artifact_root}")"
validate_mount_path "repository root" "${repo_root}"
validate_mount_path "CHECKPOINT_ROOT" "${checkpoint_root}"
validate_mount_path "ARTIFACT_ROOT" "${artifact_root}"

[[ -d "${repo_root}" ]] || fail "repository root is not a directory: ${repo_root}"
[[ -d "${checkpoint_root}" ]] || fail "CHECKPOINT_ROOT is not a directory: ${checkpoint_root}"
[[ "${artifact_root}" != "${repo_root}" ]] || \
    fail "ARTIFACT_ROOT may not be the repository root"
if [[ "${artifact_root}" == "${checkpoint_root}" \
      || "${artifact_root}" == "${checkpoint_root}/"* \
      || "${checkpoint_root}" == "${artifact_root}/"* ]]; then
    fail "ARTIFACT_ROOT and CHECKPOINT_ROOT must be disjoint"
fi

# A linked worktree's .git file points at a host-absolute gitdir that is not
# available inside the source-only bind mount. Capture provenance while the
# worktree metadata is still reachable and pass only the resulting scalars.
command -v git >/dev/null 2>&1 || fail "git CLI was not found"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum was not found"
git_commit="$(git --no-optional-locks -C "${repo_root}" rev-parse --verify HEAD)"
git_status="$(git --no-optional-locks -C "${repo_root}" status --porcelain)"
if [[ -n "${git_status}" ]]; then
    git_dirty=true
    git_diff_sha256="$({
        printf 'TRACKED\0'
        git --no-optional-locks -C "${repo_root}" diff --binary HEAD
        while IFS= read -r -d '' relative; do
            candidate="${repo_root}/${relative}"
            if [[ -L "${candidate}" ]]; then
                printf 'UNTRACKED\0%s\0L\0%s\0' \
                    "${relative}" "$(readlink -- "${candidate}")"
            elif [[ -f "${candidate}" ]]; then
                file_hash="$(sha256sum -- "${candidate}")"
                printf 'UNTRACKED\0%s\0F\0%s\0' \
                    "${relative}" "${file_hash%% *}"
            fi
        done < <(git --no-optional-locks -C "${repo_root}" \
            ls-files --others --exclude-standard -z)
    } | sha256sum)"
    git_diff_sha256="${git_diff_sha256%% *}"
else
    git_dirty=false
    git_diff_sha256=""
fi

trimmed_device="${device#"${device%%[![:space:]]*}"}"
trimmed_device="${trimmed_device%"${trimmed_device##*[![:space:]]}"}"
device="${trimmed_device,,}"
if [[ "${device}" == gpu ]]; then
    device=cuda
fi
if [[ "${device}" != cpu && ! "${device}" =~ ^cuda(:[0-9]+)?$ ]]; then
    fail "DEVICE must be cpu or cuda[:index], found: ${DEVICE:-cpu}"
fi

# An explicit evaluation --device cuda[:index] is also a GPU request. This
# keeps population and sensitivity commands self-contained while retaining CPU
# as the default.
eval_requests_gpu=0
for ((index = 0; index < ${#eval_args[@]}; index++)); do
    argument="${eval_args[index]}"
    argument_device=""
    if [[ "${argument}" == --device=* ]]; then
        argument_device="${argument#--device=}"
    elif [[ "${argument}" == --device && $((index + 1)) -lt ${#eval_args[@]} ]]; then
        argument_device="${eval_args[index + 1]}"
    fi
    argument_device="${argument_device#"${argument_device%%[![:space:]]*}"}"
    argument_device="${argument_device%"${argument_device##*[![:space:]]}"}"
    argument_device="${argument_device,,}"
    if [[ "${argument_device}" == gpu ]]; then
        argument_device=cuda
    fi
    if [[ "${argument_device}" =~ ^cuda(:[0-9]+)?$ ]]; then
        eval_requests_gpu=1
    fi
done

use_gpu=0
if [[ "${device}" =~ ^cuda(:[0-9]+)?$ || -n "${gpu_request}" \
      || "${eval_requests_gpu}" -eq 1 ]]; then
    use_gpu=1
    gpu_request="${gpu_request:-all}"
fi

if [[ -n "${CONTAINER_NAME:-}" ]]; then
    container_name="${CONTAINER_NAME}"
else
    container_name="dynamic-k-nn-eval-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
fi
if ! [[ "${container_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    fail "CONTAINER_NAME must use only letters, digits, dot, underscore, and hyphen"
fi

docker_args=(
    run
    --rm
    --init
    --name "${container_name}"
    --workdir /workspace/source
    --label dynamic-k-nn.workflow=evaluation
    --env CHECKPOINT_ROOT=/workspace/checkpoints
    --env ARTIFACT_ROOT=/workspace/artifacts
    --env "DEVICE=${device}"
    --env "EVAL_GIT_COMMIT=${git_commit}"
    --env "EVAL_GIT_DIRTY=${git_dirty}"
    --env "EVAL_GIT_DIFF_SHA256=${git_diff_sha256}"
    --mount "type=bind,src=${repo_root},dst=/workspace/source,readonly"
    --mount "type=bind,src=${checkpoint_root},dst=/workspace/checkpoints,readonly"
    --mount "type=bind,src=${artifact_root},dst=/workspace/artifacts"
)
if [ "${use_gpu}" -eq 1 ]; then
    docker_args+=(--gpus "${gpu_request}")
fi
docker_args+=("${image_name}" python -m eval "${eval_args[@]}")

if [ "${docker_dry_run}" -eq 1 ]; then
    if [ "${build_image}" -eq 1 ]; then
        printf 'IMAGE_NAME=%q %q\n' "${image_name}" "${script_dir}/build.sh"
    fi
    printf 'docker'
    printf ' %q' "${docker_args[@]}"
    printf '\n'
    exit 0
fi

require_docker
if docker container inspect "${container_name}" >/dev/null 2>&1; then
    echo "[eval-run] container already exists: ${container_name}" >&2
    exit 1
fi
if [ "${build_image}" -eq 1 ]; then
    IMAGE_NAME="${image_name}" "${script_dir}/build.sh"
elif ! docker image inspect "${image_name}" >/dev/null 2>&1; then
    echo "[eval-run] image does not exist: ${image_name}" >&2
    echo "[eval-run] rerun with --build or set IMAGE_NAME" >&2
    exit 1
fi

mkdir -p -- "${artifact_root}"
echo "[eval-run] image=${image_name} container=${container_name} device=${device}"
echo "[eval-run] checkpoints=${checkpoint_root} (read-only)"
echo "[eval-run] artifacts=${artifact_root} (read-write)"
exec docker "${docker_args[@]}"
