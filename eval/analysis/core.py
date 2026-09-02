"""Validated, artifact-agnostic primitives for dynamic-k neighbor analysis.

The canonical input is the mapping returned by :func:`eval.artifacts.load_episode`.
For the archived development runs we also accept the historical NPZ aliases used by
the original visualization scripts.  All ranking is performed independently at
each action time; agent identities are therefore deliberately not preserved along
the rank axis.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


_ALIASES = {
    "agent_states": ("states", "state_history"),
    "pointer_actions": ("pointer", "actions", "action_indices"),
    "binary_actions": ("binary", "binary_action", "neighbor_masks"),
    "phi": ("polarization", "order_parameter"),
    "s_ent": ("spatial_entropy", "position_entropy"),
    "v_ent": ("velocity_entropy",),
    "meta": ("metadata",),
}


def _mapping_lookup(
    episode: Mapping[str, Any],
    canonical: str,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    for key in (canonical,) + _ALIASES.get(canonical, ()):
        try:
            return episode[key]
        except (KeyError, TypeError):
            continue
    if required:
        raise KeyError(
            "episode is missing {!r} (accepted aliases: {})".format(
                canonical, ", ".join(_ALIASES.get(canonical, ())) or "none"
            )
        )
    return default


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


def _normalise_meta(episode: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _mapping_lookup(episode, "meta", required=False, default={})
    raw = _scalar(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("episode meta is not valid JSON") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("episode meta must be a mapping or a JSON object")
    meta = dict(raw)

    # Old full-episode NPZs stored these scalar fields at archive top level.
    scalar_keys = (
        "action_mode",
        "checkpoint_kind",
        "dt",
        "dt_seconds",
        "horizon",
        "n_agents",
        "num_agents",
        "policy_variant",
        "r0",
        "seed",
    )
    for key in scalar_keys:
        if key not in meta:
            try:
                meta[key] = _scalar(episode[key])
            except (KeyError, TypeError):
                pass
    return meta


def _integer_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.issubdtype(raw.dtype, np.floating) or not np.all(
            np.isfinite(raw)
        ):
            raise ValueError("{} must contain integer indices".format(name))
        rounded = np.rint(raw)
        if not np.array_equal(raw, rounded):
            raise ValueError("{} must contain integer indices".format(name))
        raw = rounded
    return np.asarray(raw, dtype=np.int64)


def _boolean_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_:
        if not np.all(np.logical_or(raw == 0, raw == 1)):
            raise ValueError("{} must contain only boolean/0/1 values".format(name))
    return np.asarray(raw, dtype=bool)


def wrap_angle(values: np.ndarray) -> np.ndarray:
    """Wrap angles to the half-open interval ``[-pi, pi)``."""

    return (np.asarray(values) + np.pi) % (2.0 * np.pi) - np.pi


def compute_cutoff_radii(
    states: np.ndarray, pointer_actions: np.ndarray
) -> np.ndarray:
    """Return every ego's selected pointer distance at each action time.

    ``states[t]`` is used with ``pointer_actions[t]``.  In particular this does
    not make the common off-by-one mistake of measuring the pointer after the
    environment has already advanced to ``states[t + 1]``.
    """

    states = np.asarray(states, dtype=np.float64)
    pointer = _integer_array(pointer_actions, "pointer_actions")
    if states.ndim != 3 or states.shape[2] < 2:
        raise ValueError("states must have shape (T+1, N, >=2)")
    if pointer.ndim != 2:
        raise ValueError("pointer_actions must have shape (T, N)")
    horizon, num_agents = pointer.shape
    if states.shape[:2] != (horizon + 1, num_agents):
        raise ValueError("state/pointer shapes are inconsistent")
    if np.any(pointer < 0) or np.any(pointer >= num_agents):
        raise ValueError("pointer action contains an out-of-range agent index")
    positions = states[:-1, :, :2]
    target_positions = positions[np.arange(horizon)[:, None], pointer]
    return np.linalg.norm(target_positions - positions, axis=2)


def pointer_actions_to_mask(
    states: np.ndarray, pointer_actions: np.ndarray
) -> np.ndarray:
    """Reconstruct dynamic-k masks exactly as the environment does.

    The environment stores relative positions in a float32 buffer before taking
    their norm.  Reproducing that conversion is essential for rare near-ties.
    A self pointer is a special action selecting only self, even when another
    agent is exactly co-located.
    """

    states = np.asarray(states, dtype=np.float64)
    pointer = _integer_array(pointer_actions, "pointer_actions")
    if pointer.ndim != 2:
        raise ValueError("pointer_actions must have shape (T, N)")
    horizon, num_agents = pointer.shape
    if states.ndim != 3 or states.shape[:2] != (horizon + 1, num_agents):
        raise ValueError("state/pointer shapes are inconsistent")
    if states.shape[2] < 2:
        raise ValueError("states must contain x/y positions")
    if np.any(pointer < 0) or np.any(pointer >= num_agents):
        raise ValueError("pointer action contains an out-of-range agent index")

    positions = states[:-1, :, :2]
    relative = np.asarray(
        positions[:, None, :, :] - positions[:, :, None, :], dtype=np.float32
    )
    distances = np.linalg.norm(relative, axis=3)
    time_index = np.arange(horizon)[:, None]
    ego_index = np.arange(num_agents)[None, :]
    cutoff = distances[time_index, ego_index, pointer]
    expected = distances <= cutoff[:, :, None]

    self_rows = np.where(pointer == ego_index)
    if self_rows[0].size:
        expected[self_rows[0], self_rows[1], :] = False
        expected[self_rows[0], self_rows[1], self_rows[1]] = True
    return expected


def centered_positions(states: np.ndarray) -> np.ndarray:
    """Translate every state so its instantaneous swarm centroid is the origin."""

    positions = np.asarray(states, dtype=np.float64)[..., :2]
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("states must have shape (T, N, >=2)")
    return positions - np.mean(positions, axis=1, keepdims=True)


def centroid_distance_ranking(
    centered: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid distance and stable nearest-to-farthest agent order."""

    centered = np.asarray(centered, dtype=np.float64)
    if centered.ndim != 3 or centered.shape[2] != 2:
        raise ValueError("centered positions must have shape (T, N, 2)")
    distances = np.linalg.norm(centered, axis=2)
    return distances, np.argsort(distances, axis=1, kind="stable")


def mean_heading_deviation_ranking(
    states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return circular swarm heading, absolute deviation, and closest order."""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[2] < 5:
        raise ValueError("states must have shape (T, N, >=5)")
    headings = states[:, :, 4]
    if not np.all(np.isfinite(headings)):
        raise ValueError("heading values must be finite")
    mean_heading = np.arctan2(
        np.mean(np.sin(headings), axis=1),
        np.mean(np.cos(headings), axis=1),
    )
    deviation = np.abs(wrap_angle(headings - mean_heading[:, None]))
    order = np.argsort(deviation, axis=1, kind="stable")
    return mean_heading, deviation, order


# Backward-friendly name used by the development visualization.
mean_heading_ranking = mean_heading_deviation_ranking


def state_polarization(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[2] < 5:
        raise ValueError("states must have shape (T, N, >=5)")
    headings = states[..., 4]
    return np.hypot(
        np.mean(np.cos(headings), axis=1),
        np.mean(np.sin(headings), axis=1),
    )


def state_entropy_series(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position and velocity dispersion at every state boundary."""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[0] < 2 or states.shape[2] < 4:
        raise ValueError("states must have shape (T+1, N, >=4)")
    if states.shape[1] < 1 or not np.all(np.isfinite(states[..., :4])):
        raise ValueError("states must contain finite physical values")
    position = np.sqrt(np.sum(np.var(states[..., :2], axis=1), axis=1))
    velocity = np.sqrt(np.sum(np.var(states[..., 2:4], axis=1), axis=1))
    return position, velocity


def _aligned_recorded_series(
    recorded: np.ndarray,
    derived: np.ndarray,
    name: str,
) -> Tuple[np.ndarray, float]:
    original = np.asarray(recorded)
    values = np.asarray(original, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("{} must be a finite one-dimensional series".format(name))
    horizon = derived.size - 1
    if values.shape == (horizon,):
        target = derived[1:]
        aligned = np.concatenate((derived[:1], values))
    elif values.shape == (horizon + 1,):
        target = derived
        aligned = values
    else:
        raise ValueError(
            "{} must have shape (T,) or (T+1,), got {}".format(name, values.shape)
        )
    error = float(np.max(np.abs(values - target)))
    tolerance = 2e-6 if original.dtype.itemsize <= 4 else 1e-10
    if not np.allclose(values, target, rtol=tolerance, atol=tolerance):
        raise ValueError(
            "{} differs from the physical states (max abs error {:.3e})".format(
                name, error
            )
        )
    return aligned, error


@dataclass(frozen=True)
class EpisodeData:
    """Validated in-memory view shared by all analysis entry points."""

    states: np.ndarray
    pointer_actions: np.ndarray
    binary_actions: np.ndarray
    phi: np.ndarray
    position_entropy: np.ndarray
    velocity_entropy: np.ndarray
    meta: Dict[str, Any]
    source: Optional[Path]
    metric_errors: Dict[str, float]

    @property
    def horizon(self) -> int:
        return int(self.pointer_actions.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.pointer_actions.shape[1])


def coerce_episode(
    episode: Mapping[str, Any],
    *,
    source: Optional[Path] = None,
    require_metrics: bool = True,
) -> EpisodeData:
    """Normalize a canonical ``EpisodeView`` or legacy mapping.

    Stored metrics may use either canonical state-boundary alignment ``T+1`` or
    the old post-step-only alignment ``T``.  The returned series always contain
    ``T+1`` state-boundary values.
    """

    states = np.asarray(_mapping_lookup(episode, "agent_states"), dtype=np.float64)
    pointer = _integer_array(
        _mapping_lookup(episode, "pointer_actions"), "pointer_actions"
    )
    binary = _boolean_array(
        _mapping_lookup(episode, "binary_actions"), "binary_actions"
    )
    if states.ndim != 3 or states.shape[2] < 5:
        raise ValueError("agent_states must have shape (T+1, N, >=5)")
    if pointer.ndim != 2:
        raise ValueError("pointer_actions must have shape (T, N)")
    horizon, num_agents = pointer.shape
    if states.shape[:2] != (horizon + 1, num_agents):
        raise ValueError("agent_states and pointer_actions shapes are inconsistent")
    if binary.shape != (horizon, num_agents, num_agents):
        raise ValueError(
            "binary_actions must have shape (T, N, N), got {}".format(binary.shape)
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("agent_states contains a non-finite value")

    meta = _normalise_meta(episode)
    meta_n = meta.get("num_agents", meta.get("n_agents"))
    if meta_n is not None and int(meta_n) != num_agents:
        raise ValueError("meta num_agents disagrees with agent_states")
    meta_horizon = meta.get("horizon", meta.get("max_steps"))
    if meta_horizon is not None and int(meta_horizon) != horizon:
        raise ValueError("meta horizon disagrees with pointer_actions")
    if np.any(pointer < 0) or np.any(pointer >= num_agents):
        raise ValueError("pointer action contains an out-of-range agent index")

    derived_phi = state_polarization(states)
    derived_position, derived_velocity = state_entropy_series(states)
    errors: Dict[str, float] = {}

    recorded_phi = _mapping_lookup(
        episode, "phi", required=require_metrics, default=derived_phi
    )
    phi, errors["phi_max_abs_error"] = _aligned_recorded_series(
        recorded_phi, derived_phi, "phi"
    )
    recorded_position = _mapping_lookup(
        episode, "s_ent", required=require_metrics, default=derived_position
    )
    position_entropy, errors["position_entropy_max_abs_error"] = (
        _aligned_recorded_series(recorded_position, derived_position, "s_ent")
    )
    recorded_velocity = _mapping_lookup(
        episode, "v_ent", required=require_metrics, default=derived_velocity
    )
    velocity_entropy, errors["velocity_entropy_max_abs_error"] = (
        _aligned_recorded_series(recorded_velocity, derived_velocity, "v_ent")
    )
    return EpisodeData(
        states=states,
        pointer_actions=pointer,
        binary_actions=binary,
        phi=phi,
        position_entropy=position_entropy,
        velocity_entropy=velocity_entropy,
        meta=meta,
        source=None if source is None else Path(source),
        metric_errors=errors,
    )


def resolve_dt(episode: EpisodeData, override: Optional[float] = None) -> float:
    """Resolve physical seconds per step from CLI override or artifact metadata."""

    metadata_value: Any = episode.meta.get(
        "dt", episode.meta.get("dt_seconds")
    )
    if metadata_value is None:
        env_meta = episode.meta.get("env")
        if isinstance(env_meta, Mapping):
            metadata_value = env_meta.get("dt", env_meta.get("dt_seconds"))
    candidate: Any = override if override is not None else metadata_value
    if candidate is None:
        raise ValueError("episode metadata has no dt; supply an explicit dt override")
    dt = float(candidate)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if metadata_value is not None:
        recorded = float(metadata_value)
        if not np.isfinite(recorded) or recorded <= 0.0:
            raise ValueError("episode metadata dt must be positive and finite")
        if override is not None and not np.isclose(
            dt, recorded, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "dt override {} disagrees with episode metadata {}".format(
                    dt, recorded
                )
            )
    return dt


def require_deterministic_episode(episode: EpisodeData) -> None:
    """Reject stochastic rollouts instead of silently mixing populations."""

    mode = episode.meta.get("action_mode", episode.meta.get("policy"))
    if mode is None and "deterministic" in episode.meta:
        mode = "deterministic" if bool(episode.meta["deterministic"]) else "stochastic"
    if mode is None:
        raise ValueError("episode metadata has no deterministic action-mode marker")
    if str(mode).strip().lower() not in {"deterministic", "det", "greedy"}:
        raise ValueError("episode action_mode is not deterministic: {!r}".format(mode))


def validate_pointer_mask(
    episode: EpisodeData, *, raise_on_mismatch: bool = True
) -> Dict[str, Any]:
    """Cross-check pointer actions against the persisted binary neighbor masks."""

    expected = pointer_actions_to_mask(episode.states, episode.pointer_actions)
    mismatch = np.argwhere(expected != episode.binary_actions)
    report: Dict[str, Any] = {
        "mismatch_count": int(mismatch.shape[0]),
        "first_mismatch": None,
    }
    if mismatch.size:
        first = tuple(int(value) for value in mismatch[0])
        report["first_mismatch"] = list(first)
        if raise_on_mismatch:
            raise ValueError(
                "pointer-derived mask disagrees with binary_actions at {} "
                "({} mismatched cells)".format(first, mismatch.shape[0])
            )
    return report


def rank_episode_cutoff_radii(
    states: np.ndarray, pointer_actions: np.ndarray
) -> Dict[str, np.ndarray]:
    """Rank one episode's cutoff radii by position and circular heading context."""

    radii = compute_cutoff_radii(states, pointer_actions)
    centered = centered_positions(states)[:-1]
    centroid_distances, centroid_order = centroid_distance_ranking(centered)
    mean_heading, heading_deviation, heading_order = (
        mean_heading_deviation_ranking(np.asarray(states)[:-1])
    )
    centroid_ranked_distances = np.take_along_axis(
        centroid_distances, centroid_order, axis=1
    )
    heading_ranked_deviation = np.take_along_axis(
        heading_deviation, heading_order, axis=1
    )
    if np.any(np.diff(centroid_ranked_distances, axis=1) < -1e-12):
        raise ValueError("centroid-distance rank is not monotone")
    if np.any(np.diff(heading_ranked_deviation, axis=1) < -1e-12):
        raise ValueError("mean-heading-deviation rank is not monotone")
    return {
        "cutoff_radii": radii,
        "centroid_order": centroid_order,
        "heading_order": heading_order,
        "mean_heading": mean_heading,
        "centroid_ranked_radii": np.take_along_axis(
            radii, centroid_order, axis=1
        ),
        "heading_ranked_radii": np.take_along_axis(
            radii, heading_order, axis=1
        ),
        "centroid_ranked_distances": centroid_ranked_distances,
        "heading_ranked_deviation": heading_ranked_deviation,
    }


def time_window_steps(
    horizon: int, dt: float, max_time_seconds: Optional[float] = None
) -> int:
    """Return action-time cells ending exactly at ``max_time_seconds``."""

    horizon = int(horizon)
    dt = float(dt)
    if horizon < 1 or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("horizon and dt must be positive")
    if max_time_seconds is None:
        return horizon
    requested = float(max_time_seconds)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("max_time_seconds must be positive and finite")
    raw_steps = requested / dt
    steps = int(round(raw_steps))
    if not np.isclose(raw_steps, steps, rtol=0.0, atol=1e-9):
        raise ValueError(
            "max_time_seconds={} does not align with dt={}".format(requested, dt)
        )
    if steps < 1 or steps > horizon:
        raise ValueError(
            "max_time_seconds={} is outside (0, {}]".format(
                requested, horizon * dt
            )
        )
    return steps
