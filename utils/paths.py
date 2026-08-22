"""Repo-root resolution for scripts that historically hardcoded /workspace.

Resolution order:
  1. FLOCK_ROOT environment variable, when set.
  2. The directory containing this package's parent (i.e. the repo root that
     this file was imported from).

In the canonical deployment the repo sits at /workspace, so both resolutions
agree there; a clone at any other path works with no configuration.
"""
import os


def repo_root() -> str:
    return os.environ.get("FLOCK_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    return os.path.join(repo_root(), *parts)
