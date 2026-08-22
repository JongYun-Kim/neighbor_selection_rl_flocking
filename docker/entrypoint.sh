#!/bin/bash
set -Eeuo pipefail

if [ "${START_SSHD:-1}" = "1" ]; then
    # Keep the legacy interactive/SSH workflow available. Host-driven training
    # disables this daemon because the training process is the container job.
    [ -d /var/run/sshd ] || mkdir -p /var/run/sshd
    echo "[entrypoint] starting sshd..."
    /usr/sbin/sshd
else
    echo "[entrypoint] sshd disabled"
fi

# Just to check versions and stuffs at the entry
run_checks_and_shell() {
    echo "[entrypoint] running Python/CUDA checks as 'flocking'"

    sudo -H -u flocking bash -lc 'python - << "PY"
import torch, torchvision, torchaudio, ray, gym, numpy as np
import sys

print("=== Python & library versions ===")
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("ray", ray.__version__)
print("gym", gym.__version__)
print("numpy", np.__version__)

print("\n=== CUDA / GPU info ===")
cuda_available = torch.cuda.is_available()
print("torch.cuda.is_available:", cuda_available)

if cuda_available:
    device_count = torch.cuda.device_count()
    print("torch.cuda.device_count:", device_count)
    for i in range(device_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}, capability={torch.cuda.get_device_capability(i)}")
    print("current device index:", torch.cuda.current_device())
else:
    print("No CUDA device visible to PyTorch.")
PY'

    echo "[entrypoint] dropping into bash as 'flocking'"
    exec sudo -H -u flocking bash
}


# How it actually starts
if [ $# -eq 0 ]; then
    # no args → checks + flocking bash
    run_checks_and_shell
elif [ $# -eq 1 ] && [ "$1" = "bash" ]; then
    # single arg "bash" → also checks + flocking bash
    run_checks_and_shell
else
    # any other command → run as flocking
    echo "[entrypoint] executing as 'flocking': $*"
    # Preserve explicit Docker workflow metadata and runtime controls across
    # the root -> flocking user transition. Secrets are file-mounted rather
    # than placed in this environment.
    exec sudo -H -E -u flocking -- "$@"
fi
