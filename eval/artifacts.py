"""Versioned evaluation artifacts, atomic I/O, and legacy read compatibility.

The criterion-of-record evaluator historically stored a compact ``T+1`` set
of scalar series, while the earlier Dynamic-k evaluator stored complete state
and action traces under different key names.  This module gives new code one
read interface without changing the meaning of either historical format.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


FULL_SCHEMA_VERSION = "main-c2-full-2.0"

SCALAR_SERIES = (
    "s_ent", "v_ent", "s_ent_env", "v_ent_env", "phi", "reward",
    "nnd_mean", "nnd_max", "min_pair", "diam", "radius",
    "n_comp_sel", "n_comp_r0", "churn", "deg_mean",
)

FULL_REQUIRED = (
    "agent_states", "pointer_actions", "binary_actions", "control_inputs",
)

CANONICAL_FULL_REQUIRED = FULL_REQUIRED + SCALAR_SERIES + (
    "deg_agents", "t_fire", "success", "J",
)


class EpisodeView(Mapping):
    """Mapping-like, normalized view of one episode NPZ.

    Arrays are copied before the underlying ``np.load`` handle is closed.  The
    ``legacy`` flag means only that aliases were required; it does not make an
    old 1000-step trajectory a valid failure observation under the 6000-step
    main C2 protocol.
    """

    def __init__(self, data: Dict[str, Any], path: Path, schema: str,
                 legacy: bool, full: bool):
        self._data = data
        self.path = Path(path)
        self.schema_version = str(schema)
        self.legacy = bool(legacy)
        self.full = bool(full)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._data["meta"]


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(_scalar_string(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _copy_archive(path: Path) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _legacy_full_to_view(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map dev NPZ 1.1 names to canonical read names, additively."""
    data = dict(raw)
    aliases = {
        "spatial_entropy": "s_ent",
        "velocity_entropy": "v_ent",
        "proximity_components": "n_comp_r0",
    }
    for old, new in aliases.items():
        if new not in data and old in data:
            data[new] = np.asarray(data[old]).copy()

    # The old full format has T post-step samples.  Canonical series have the
    # initial state at index zero.  Derive initial metrics from the stored state
    # so plotting alignment remains explicit and correct.
    if "agent_states" in data:
        states = np.asarray(data["agent_states"], dtype=np.float64)
        if states.ndim == 3 and states.shape[0] >= 1:
            p0, v0 = states[0, :, :2], states[0, :, 2:4]
            initial = {
                "s_ent": float(np.sqrt(p0.var(axis=0).sum())),
                "v_ent": float(np.sqrt(v0.var(axis=0).sum())),
            }
            speed = np.linalg.norm(v0, axis=1)
            unit = v0 / np.maximum(speed, 1e-12)[:, None]
            initial["phi"] = float(np.linalg.norm(unit.mean(axis=0)))
            horizon = states.shape[0] - 1
            for name in ("s_ent", "v_ent", "phi", "n_comp_r0"):
                if name not in data:
                    continue
                values = np.asarray(data[name])
                if values.shape == (horizon,):
                    first = initial.get(name, values[0])
                    data[name] = np.concatenate(
                        [np.asarray([first], dtype=values.dtype), values]
                    )

    if "reward" not in data and "original_rewards" in data:
        rewards = np.asarray(data["original_rewards"])
        data["reward"] = np.concatenate(
            [np.asarray([np.nan], dtype=rewards.dtype), rewards]
        )

    meta = {}
    for key in ("episode_index", "seed", "action_seed", "num_agents",
                "horizon", "policy_variant", "checkpoint_kind", "action_mode"):
        if key in data:
            value = np.asarray(data[key])
            meta[key] = value.item() if value.shape == () else value.tolist()
    meta["source_schema"] = _scalar_string(data.get("schema_version", "legacy-full"))
    meta["protocol_id"] = "legacy-40-window"
    meta["official_c2_complete_horizon"] = False
    data["meta"] = meta
    return data


def load_episode(path: os.PathLike) -> EpisodeView:
    """Load new full, current-main sparse, or historical dev NPZ.

    Historical convergence fields remain present under their original keys,
    but are never aliased to the main C2 result.
    """
    path = Path(path)
    raw = _copy_archive(path)
    schema = _scalar_string(raw.get("schema_version", "main-sparse-1"))

    if schema == FULL_SCHEMA_VERSION:
        data = dict(raw)
        data["meta"] = _json_object(data.get("meta", {}))
        return EpisodeView(data, path, schema, legacy=False, full=True)

    if "agent_states" in raw and "pointer_actions" in raw:
        data = _legacy_full_to_view(raw)
        return EpisodeView(data, path, schema, legacy=True, full=True)

    # Current criterion-of-record sparse artifact.
    data = dict(raw)
    data["meta"] = _json_object(data.get("meta", {}))
    return EpisodeView(data, path, schema, legacy=False, full=False)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def json_compatible(value: Any) -> Any:
    """Return a strict-JSON representation, mapping non-finite scalars to null.

    Numeric arrays in NPZ artifacts retain their IEEE NaNs.  Embedded metadata
    is JSON, however, so missing statistics (for example ``J`` on a failed
    episode) are represented explicitly as JSON ``null``.
    """
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: os.PathLike) -> str:
    """Hash one file or tree, including entry types and symlink targets."""
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*"),
                       key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        if item.is_symlink():
            digest.update(b"L\0" + relative + b"\0"
                          + os.readlink(item).encode("utf-8") + b"\0")
        elif item.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif item.is_file():
            digest.update(b"F\0" + relative + b"\0")
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def run_fingerprint(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def _atomic_temporary_file(path: Path):
    """Create an adjacent temporary file while respecting the process umask.

    ``tempfile.mkstemp`` deliberately forces mode 0600.  Evaluation artifacts
    are commonly shared through a group-writable results directory, so that
    default silently made new NPZ/JSON files unreadable to collaborators even
    when the caller's umask was 0002.  ``os.open(..., 0o666)`` retains exclusive
    creation while allowing the kernel's umask/default ACL policy to decide the
    effective mode.  Replacing an existing file preserves its current mode,
    matching ordinary in-place writers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _ in range(100):
        temporary = path.with_name(
            ".tmp-{}.{}".format(path.name, secrets.token_hex(8))
        )
        try:
            descriptor = os.open(str(temporary), flags, 0o666)
        except FileExistsError:
            continue
        try:
            if existing_mode is not None:
                os.fchmod(descriptor, existing_mode)
        except BaseException:
            os.close(descriptor)
            os.unlink(temporary)
            raise
        return descriptor, temporary
    raise FileExistsError("could not allocate atomic temporary file for {}".format(path))


@contextmanager
def atomic_open(path: os.PathLike, mode: str, *, encoding=None, newline=None):
    """Yield an adjacent atomic output stream with normal shared-file modes."""
    if mode not in ("w", "wb"):
        raise ValueError("atomic_open supports only 'w' and 'wb'")
    path = Path(path)
    descriptor, temporary = _atomic_temporary_file(path)
    kwargs = {}
    if mode == "w":
        kwargs.update(encoding=encoding or "utf-8", newline=newline)
    try:
        try:
            stream = os.fdopen(descriptor, mode, **kwargs)
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def make_staging_directory(parent: os.PathLike, prefix: str) -> Path:
    """Create a unique directory using normal umask/default-ACL semantics."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        candidate = parent / "{}{}".format(prefix, secrets.token_hex(8))
        try:
            os.mkdir(candidate, 0o777)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(
        "could not allocate staging directory under {}".format(parent)
    )


def atomic_write_json(path: os.PathLike, value: Any) -> None:
    with atomic_open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def atomic_save_npz(path: os.PathLike, payload: Dict[str, Any]) -> None:
    with atomic_open(path, "wb") as stream:
        np.savez_compressed(stream, **payload)


def validate_full_episode(view: EpisodeView, expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Structurally validate canonical or historical full trajectories."""
    if not view.full:
        raise ValueError(f"full trajectory fields are unavailable: {view.path}")
    required = (CANONICAL_FULL_REQUIRED
                if view.schema_version == FULL_SCHEMA_VERSION else FULL_REQUIRED)
    missing = [key for key in required if key not in view]
    if missing:
        raise ValueError(f"missing full episode keys {missing}: {view.path}")

    states = np.asarray(view["agent_states"])
    pointer = np.asarray(view["pointer_actions"])
    binary = np.asarray(view["binary_actions"])
    controls = np.asarray(view["control_inputs"])
    if states.ndim != 3 or states.shape[2] != 5:
        raise ValueError(f"agent_states must have shape (T+1,N,5): {view.path}")
    horizon, n_agents = states.shape[0] - 1, states.shape[1]
    expected_shapes = {
        "pointer_actions": (horizon, n_agents),
        "binary_actions": (horizon, n_agents, n_agents),
        "control_inputs": (horizon, n_agents),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(view[key]).shape != shape:
            raise ValueError(f"{key} shape {np.asarray(view[key]).shape} != {shape}")
    if not np.issubdtype(pointer.dtype, np.integer):
        raise ValueError(f"pointer actions must have integer dtype: {view.path}")
    if np.any(pointer < 0) or np.any(pointer >= n_agents):
        raise ValueError(f"pointer action out of range: {view.path}")
    if binary.dtype != np.bool_ and not np.all((binary == 0) | (binary == 1)):
        raise ValueError(f"binary actions must contain only 0/1: {view.path}")
    if not np.all(np.diagonal(binary.astype(bool), axis1=1, axis2=2)):
        raise ValueError(f"binary action lacks self loops: {view.path}")
    for key in ("agent_states", "control_inputs"):
        if not np.isfinite(np.asarray(view[key])).all():
            raise ValueError(f"non-finite {key}: {view.path}")

    series_names = SCALAR_SERIES if view.schema_version == FULL_SCHEMA_VERSION else (
        "s_ent", "v_ent", "phi", "n_comp_r0", "reward")
    for name in series_names:
        if name in view and np.asarray(view[name]).shape != (horizon + 1,):
            raise ValueError(f"{name} must have T+1 values: {view.path}")
    if "deg_agents" in view and np.asarray(view["deg_agents"]).shape != (
            horizon, n_agents):
        raise ValueError(f"deg_agents must have shape (T,N): {view.path}")
    for name in ("t_fire", "success", "J"):
        if name in view and np.asarray(view[name]).shape != ():
            raise ValueError(f"{name} must be scalar: {view.path}")
    if view.schema_version == FULL_SCHEMA_VERSION:
        for name in ("s_ent", "v_ent", "phi", "nnd_mean", "nnd_max",
                     "min_pair", "diam", "radius", "n_comp_r0"):
            if not np.isfinite(np.asarray(view[name])).all():
                raise ValueError(f"non-finite canonical {name}: {view.path}")
        for name in ("s_ent_env", "v_ent_env", "reward"):
            values = np.asarray(view[name])
            if not np.isnan(values[0]) or not np.isfinite(values[1:]).all():
                raise ValueError(
                    f"canonical {name} must be NaN only at index zero: {view.path}")

    meta = dict(view.meta)
    if expected:
        for key, value in expected.items():
            actual = meta.get(key)
            if actual != value:
                raise ValueError(
                    f"episode metadata mismatch for {key}: {actual!r} != {value!r}"
                )
    return {"path": str(view.path), "horizon": horizon, "n_agents": n_agents,
            "seed": meta.get("seed"), "policy": meta.get("policy"),
            "schema_version": view.schema_version}
