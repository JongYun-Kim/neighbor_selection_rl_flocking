"""Small, shared building blocks for maintained evaluation runners."""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from eval.artifacts import (
    atomic_open,
    make_staging_directory,
    run_fingerprint,
    sha256_file,
    sha256_tree,
)
from eval.policies import resolve_dynamic_checkpoint


def parse_int_range(value: str) -> List[int]:
    """Parse ``A-B`` or a comma-separated integer list."""
    value = str(value).strip()
    if "," in value:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif "-" in value:
        left, right = (int(item) for item in value.split("-", 1))
        if right < left:
            raise ValueError("range end precedes range start")
        result = list(range(left, right + 1))
    else:
        result = [int(value)]
    if not result or len(result) != len(set(result)):
        raise ValueError("seed list must be non-empty and unique")
    return result


def derive_action_seed(seed: int, num_agents: int) -> int:
    value = f"distance-pointer-action:{int(seed)}:{int(num_agents)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)


def pure_acs_pointer(env) -> np.ndarray:
    """Choose the farthest active agent so ACS sees the complete graph."""
    active = np.flatnonzero(env.state["padding_mask"])
    distances = env.rel_state["rel_agent_dists"]
    pointer = np.zeros(env.num_agents_max, dtype=np.int64)
    for ego in active:
        candidates = active[active != ego]
        pointer[ego] = (
            ego if not candidates.size
            else candidates[int(np.argmax(distances[ego, candidates]))]
        )
    return pointer


def state_hash(state: np.ndarray) -> str:
    value = np.ascontiguousarray(state)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_identity(checkpoint_path: Path) -> Dict[str, object]:
    source = resolve_dynamic_checkpoint(checkpoint_path)
    checkpoint = Path(source["checkpoint"])
    config_file = Path(source["config_file"])
    if source["kind"] == "rllib":
        state_file = Path(source["policy_state"])
        checkpoint_id = checkpoint.name
    else:
        state_file = Path(source["state_dict"])
        checkpoint_id = Path(source["archive_root"]).name
    tree_hash = sha256_tree(checkpoint)
    state_digest = sha256_file(state_file)
    config_hash = sha256_file(config_file)
    model_config_hash = run_fingerprint(source["model_config"])
    package_hash = run_fingerprint({
        "checkpoint_tree_sha256": tree_hash,
        "state_file_sha256": state_digest,
        "config_file_sha256": config_hash,
        "model_config_sha256": model_config_hash,
        "obs_position_scale": source["obs_position_scale"],
        "source_kind": source["kind"],
    })
    return {
        "id": checkpoint_id,
        "source": str(Path(checkpoint_path).expanduser().resolve()),
        "source_kind": source["kind"],
        "tree_sha256": tree_hash,
        "state_file": str(state_file),
        "state_file_sha256": state_digest,
        "config_file": str(config_file),
        "config_file_sha256": config_hash,
        "model_config_sha256": model_config_hash,
        "package_sha256": package_hash,
        "obs_position_scale": source["obs_position_scale"],
    }


def quarantine(path: Path, bundle: Path) -> Path:
    relative = path.relative_to(bundle)
    target = bundle / "quarantine" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        suffix = datetime.now().strftime("%H%M%S%f")
        target = target.with_name(f"{target.stem}-{suffix}{target.suffix}")
    shutil.move(str(path), str(target))
    return target


def archive_checkpoint(bundle: Path, checkpoint_path: Path,
                       checkpoint_info: Dict[str, object],
                       repair_invalid: bool = False) -> str:
    """Atomically archive and verify a complete inference load package."""
    source = resolve_dynamic_checkpoint(checkpoint_path)
    archive_root = bundle / "checkpoint"
    if archive_root.exists():
        try:
            archived = checkpoint_identity(archive_root)
            if archived["package_sha256"] != checkpoint_info["package_sha256"]:
                raise ValueError("archived finalist package hash differs from source")
            return archive_root.relative_to(bundle).as_posix()
        except Exception:
            if not repair_invalid:
                raise
            quarantine(archive_root, bundle)

    bundle.mkdir(parents=True, exist_ok=True)
    staging = make_staging_directory(bundle, ".tmp-checkpoint.")
    try:
        archived_checkpoint = staging / "checkpoint"
        source_checkpoint = Path(source["checkpoint"])
        if source_checkpoint.is_dir():
            shutil.copytree(str(source_checkpoint), str(archived_checkpoint))
        else:
            shutil.copy2(str(source_checkpoint), str(archived_checkpoint))
        if source["kind"] == "rllib":
            shutil.copy2(str(source["config_file"]), str(staging / "params.json"))
        else:
            source_root = Path(source["archive_root"])
            for name in ("metadata.json", "model_state_dict.pt", "checksums.json"):
                item = source_root / name
                if item.is_file():
                    shutil.copy2(str(item), str(staging / name))
        archived = checkpoint_identity(staging)
        if archived["package_sha256"] != checkpoint_info["package_sha256"]:
            raise RuntimeError("finalist checkpoint package failed checksum validation")
        os.replace(staging, archive_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return archive_root.relative_to(bundle).as_posix()


def write_csv_atomic(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(list(rows)).to_csv(stream, index=False)


__all__ = [
    "archive_checkpoint",
    "checkpoint_identity",
    "derive_action_seed",
    "parse_int_range",
    "pure_acs_pointer",
    "quarantine",
    "state_hash",
    "utc_now",
    "write_csv_atomic",
]
