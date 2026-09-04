"""Repository-local defaults shared by the maintained evaluation CLIs."""

from __future__ import annotations

import os
from pathlib import Path

from utils.paths import repo_path


CHECKPOINT_ROOT = Path(
    os.environ.get("CHECKPOINT_ROOT", repo_path("checkpoints"))
).expanduser()
DEFAULT_CHECKPOINT_PACKAGE = CHECKPOINT_ROOT / "dynamic_k_nn_best"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_PACKAGE / "checkpoint_000592"
DEFAULT_CHECKPOINT_LABEL = "dynamic_k_nn_best"
