"""Reproducibility metadata shared by maintained evaluation suites."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np


def _command(repo_root: Path, *args: str, binary: bool = False):
    return subprocess.check_output(
        args,
        cwd=str(repo_root),
        text=not binary,
        stderr=subprocess.DEVNULL,
    )


def dirty_payload_sha256(repo_root: Path) -> str:
    """Hash the tracked HEAD diff plus every untracked, non-ignored file."""
    repo_root = Path(repo_root)
    digest = hashlib.sha256()
    digest.update(b"TRACKED\0")
    digest.update(_command(
        repo_root, "git", "diff", "--binary", "HEAD", binary=True))
    untracked = _command(
        repo_root, "git", "ls-files", "--others", "--exclude-standard", "-z",
        binary=True,
    ).split(b"\0")
    for encoded in sorted(item for item in untracked if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = repo_root / relative
        digest.update(b"UNTRACKED\0" + encoded + b"\0")
        if path.is_symlink():
            digest.update(b"L\0" + os.readlink(path).encode(
                "utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"F\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"MISSING\0")
        digest.update(b"\0")
    return digest.hexdigest()


def git_snapshot(repo_root: Path) -> Dict[str, object]:
    """Capture commit and a dirty payload hash, including untracked code."""
    if os.environ.get("EVAL_GIT_COMMIT"):
        dirty_text = os.environ.get("EVAL_GIT_DIRTY", "")
        return {
            "commit": os.environ["EVAL_GIT_COMMIT"],
            "dirty": dirty_text.lower() in ("1", "true", "yes"),
            "dirty_diff_sha256": os.environ.get("EVAL_GIT_DIFF_SHA256") or None,
        }
    try:
        commit = _command(repo_root, "git", "rev-parse", "HEAD").strip()
        status = _command(repo_root, "git", "status", "--porcelain")
        return {
            "commit": commit,
            "dirty": bool(status),
            "dirty_diff_sha256": (
                dirty_payload_sha256(repo_root) if status else None),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_diff_sha256": None}


def runtime_snapshot() -> Dict[str, object]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
        result["torch"] = torch.__version__
    except ImportError:
        result["torch"] = None
    return result


__all__ = ["dirty_payload_sha256", "git_snapshot", "runtime_snapshot"]
