#!/bin/bash
set -Eeuo pipefail

# The image runs as 'flocking' (Dockerfile `USER`), so this script only reports
# the environment on an interactive entry and hands every other invocation
# straight to the requested command. No privilege switching, no daemons.

# Just to check versions and stuffs at the entry
run_checks_and_shell() {
    echo "[entrypoint] running Python/CUDA checks as '$(id -un)'"

    # A broken import must not cost you the shell you need to debug it.
    python - << "PY" || echo "[entrypoint] WARNING: environment check failed" >&2
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
PY

    echo "[entrypoint] dropping into bash as '$(id -un)'"
    exec bash
}


# How it actually starts
if [ $# -eq 0 ]; then
    # no args → checks + interactive bash
    run_checks_and_shell
elif [ $# -eq 1 ] && [ "$1" = "bash" ]; then
    # single arg "bash" → also checks + interactive bash
    run_checks_and_shell
else
    # any other command (the training service, tests, one-off scripts) runs
    # verbatim as the container user.
    echo "[entrypoint] executing as '$(id -un)': $*"
    exec "$@"
fi
