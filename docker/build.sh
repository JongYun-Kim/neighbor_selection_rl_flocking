#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${IMAGE_NAME:-uom-neighbor-selection}"
host_uid="${HOST_UID:-$(id -u)}"
host_gid="${HOST_GID:-$(id -g)}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[build] docker CLI was not found." >&2
  exit 1
fi

echo "[build] image=${image_name} uid=${host_uid} gid=${host_gid}"
exec docker build \
  --tag "${image_name}" \
  --build-arg "UID=${host_uid}" \
  --build-arg "GID=${host_gid}" \
  --file "${script_dir}/Dockerfile" \
  "${script_dir}"
