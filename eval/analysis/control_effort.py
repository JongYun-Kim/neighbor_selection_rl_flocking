"""Main-C2 control-effort analysis for versioned full episode bundles.

The analysis accepts both ``main-c2-full-2.0`` episodes and the historical
full Dynamic-k archives normalized by :func:`eval.artifacts.load_episode`.
Convergence is always recomputed with the main criterion of record; archived
legacy convergence labels are never reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval.artifacts import atomic_open, atomic_write_json, load_episode
from eval.protocol import MAIN_C2_V1, judge_episode_c2

from .bundle import resolve_analysis_output


PRE_C2_WINDOW = 100
POST_C2_WINDOW = 100
EVENT_OFFSETS = np.arange(-200, 301, dtype=np.int32)

SUCCESS_METRICS = (
    "t_fire",
    "l1_integral_to_c2",
    "env_l1_cost_to_c2",
    "l2_energy_to_c2",
    "l2_norm_to_c2",
    "mean_abs_to_c2",
    "rms_to_c2",
    "pre100_mean_abs",
    "post100_mean_abs",
    "pre100_mean_square",
    "post100_mean_square",
    "pre100_rms",
    "post100_rms",
)
HORIZON_METRICS = (
    "l1_integral_to_c2_or_horizon",
    "env_l1_cost_to_c2_or_horizon",
    "l2_energy_to_c2_or_horizon",
    "l2_norm_to_c2_or_horizon",
)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if np.asarray(values).size else float("nan")


def compute_effort_metrics(control_inputs: np.ndarray, t_fire: int,
                           dt: float, speed: float) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Compute per-episode L1/L2 metrics from post-step heading-rate controls.

    ``t_fire`` is the index in a T+1 state series.  Thus the control that creates
    state ``t_fire`` has index ``t_fire - 1`` and the inclusive effort-to-C2
    interval is ``control_inputs[:t_fire]``.
    """
    control = np.asarray(control_inputs, dtype=np.float64)
    if control.ndim != 2 or min(control.shape) <= 0:
        raise ValueError("control_inputs must have non-empty shape (T, N)")
    if not np.isfinite(control).all():
        raise ValueError("control_inputs contains non-finite values")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("episode dt must be a positive finite value")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("episode speed must be a positive finite value")

    horizon = int(control.shape[0])
    success = int(t_fire) >= 0
    if success and int(t_fire) > horizon:
        raise ValueError("C2 firing index exceeds the control horizon")

    l1_rate = np.mean(np.abs(control), axis=1)
    l2_rate = np.mean(np.square(control), axis=1)
    stop = int(t_fire) if success else horizon
    event_index = max(0, int(t_fire) - 1) if success else horizon
    pre = slice(max(0, event_index - PRE_C2_WINDOW), event_index)
    post = slice(event_index, min(horizon, event_index + POST_C2_WINDOW))

    l1_integral = float(dt * np.sum(l1_rate[:stop]))
    l2_energy = float(dt * np.sum(l2_rate[:stop]))
    success_value = lambda value: value if success else float("nan")
    metrics = {
        "l1_integral_to_c2": success_value(l1_integral),
        "env_l1_cost_to_c2": success_value(float(speed * l1_integral)),
        "l2_energy_to_c2": success_value(l2_energy),
        "l2_norm_to_c2": success_value(float(np.sqrt(l2_energy))),
        "mean_abs_to_c2": success_value(_mean_or_nan(l1_rate[:stop])),
        "rms_to_c2": success_value(
            float(np.sqrt(np.mean(l2_rate[:stop]))) if stop else float("nan")
        ),
        "l1_integral_to_c2_or_horizon": l1_integral,
        "env_l1_cost_to_c2_or_horizon": float(speed * l1_integral),
        "l2_energy_to_c2_or_horizon": l2_energy,
        "l2_norm_to_c2_or_horizon": float(np.sqrt(l2_energy)),
        "pre100_mean_abs": success_value(_mean_or_nan(l1_rate[pre])),
        "post100_mean_abs": success_value(_mean_or_nan(l1_rate[post])),
        "pre100_mean_square": success_value(_mean_or_nan(l2_rate[pre])),
        "post100_mean_square": success_value(_mean_or_nan(l2_rate[post])),
        "pre100_rms": success_value(
            float(np.sqrt(np.mean(l2_rate[pre])))
            if l2_rate[pre].size else float("nan")
        ),
        "post100_rms": success_value(
            float(np.sqrt(np.mean(l2_rate[post])))
            if l2_rate[post].size else float("nan")
        ),
        "full_mean_abs": float(np.mean(l1_rate)),
        "full_mean_square": float(np.mean(l2_rate)),
    }
    temporal = {
        "l1_rate": l1_rate,
        "l2_rate": l2_rate,
        "l1_cost_cumulative": speed * dt * np.cumsum(l1_rate),
        "l2_energy_cumulative": dt * np.cumsum(l2_rate),
    }
    return metrics, temporal


def event_aligned(values: np.ndarray, t_fire: int,
                  offsets: np.ndarray = EVENT_OFFSETS) -> np.ndarray:
    """Align a post-step control-rate series to the action reaching C2."""
    values = np.asarray(values, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int32)
    aligned = np.full(offsets.shape, np.nan, dtype=np.float64)
    if int(t_fire) < 0:
        return aligned
    event_index = max(0, int(t_fire) - 1)
    indices = event_index + offsets
    valid = (indices >= 0) & (indices < values.shape[0])
    aligned[valid] = values[indices[valid]]
    return aligned


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        return value
    item = array.item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _view_value(view, names: Sequence[str]) -> Any:
    for name in names:
        if name in view.meta and view.meta[name] is not None:
            return view.meta[name]
        if name in view:
            value = np.asarray(view[name])
            if value.shape == ():
                return _scalar(value)
    return None


def _walk_named_values(value: Any, names: set) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names and not isinstance(child, (Mapping, list, tuple)):
                yield child
            yield from _walk_named_values(child, names)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_named_values(child, names)


def _manifest_value(manifest: Mapping[str, Any], names: Sequence[str],
                    n_agents: Optional[int] = None) -> Any:
    spec = manifest.get("spec", {}) if isinstance(manifest, Mapping) else {}
    if isinstance(spec, Mapping):
        for name in names:
            if name in spec:
                return spec[name]

    # Historical manifests keep the effective env config per population size.
    evaluation = manifest.get("evaluation", {}) if isinstance(manifest, Mapping) else {}
    configs = evaluation.get("config_by_num_agents", {}) if isinstance(evaluation, Mapping) else {}
    if n_agents is not None and isinstance(configs, Mapping):
        selected = configs.get(str(int(n_agents)), configs.get(int(n_agents)))
        candidates = list(_walk_named_values(selected, set(names)))
        if candidates:
            return candidates[0]

    candidates = list(_walk_named_values(manifest, set(names)))
    if not candidates:
        return None
    numeric = []
    for item in candidates:
        try:
            numeric.append(float(item))
        except (TypeError, ValueError):
            pass
    if numeric and all(value == numeric[0] for value in numeric[1:]):
        return candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _required_physical_value(view, manifest: Mapping[str, Any], name: str,
                             aliases: Sequence[str], n_agents: int) -> float:
    episode_value = _view_value(view, aliases)
    manifest_value = _manifest_value(manifest, aliases, n_agents=n_agents)
    if episode_value is not None and manifest_value is not None and not np.isclose(
            float(episode_value), float(manifest_value), rtol=0.0, atol=1e-12):
        raise ValueError(
            f"episode/manifest {name} mismatch: {episode_value!r} != "
            f"{manifest_value!r}: {view.path}")
    value = episode_value if episode_value is not None else manifest_value
    if value is None:
        raise ValueError(
            f"{name} is absent from episode metadata and bundle manifest: {view.path}"
        )
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"invalid {name}={value!r}: {view.path}")
    return number


def _component_count(positions: np.ndarray, r0: float) -> int:
    positions = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = distances < float(r0)
    n_agents = positions.shape[0]
    seen = np.zeros(n_agents, dtype=bool)
    count = 0
    for start in range(n_agents):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for neighbor in np.flatnonzero(adjacency[node] & ~seen):
                seen[neighbor] = True
                stack.append(int(neighbor))
    return count


def _state_metrics(states: np.ndarray, r0: float,
                   need_components: bool) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    positions = states[:, :, :2].astype(np.float64, copy=False)
    velocities = states[:, :, 2:4].astype(np.float64, copy=False)
    spatial = np.sqrt(np.var(positions, axis=1).sum(axis=1))
    speeds = np.linalg.norm(velocities, axis=2)
    units = velocities / np.maximum(speeds, 1e-12)[:, :, None]
    phi = np.linalg.norm(np.mean(units, axis=1), axis=1)
    components = None
    if need_components:
        components = np.asarray(
            [_component_count(frame, r0) for frame in positions], dtype=np.int16)
    return phi, spatial, components


def _main_c2_series(view, states: np.ndarray, r0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = states.shape[0]

    def archived(name: str) -> Optional[np.ndarray]:
        if name not in view:
            return None
        values = np.asarray(view[name]).copy()
        return values if values.shape == (expected,) else None

    phi = archived("phi")
    spatial = archived("s_ent")
    components = archived("n_comp_r0")
    derived_phi, derived_spatial, derived_components = _state_metrics(
        states, r0, need_components=components is None)
    if phi is None:
        phi = derived_phi
    if spatial is None:
        spatial = derived_spatial
    if components is None:
        components = derived_components

    # Legacy post-step series were promoted to T+1 by repeating their first
    # component count.  Recompute index zero from the actual initial state.
    phi = np.asarray(phi, dtype=np.float64)
    spatial = np.asarray(spatial, dtype=np.float64)
    components = np.asarray(components)
    phi[0] = derived_phi[0]
    spatial[0] = derived_spatial[0]
    components[0] = _component_count(states[0, :, :2], r0)
    if not (np.isfinite(phi).all() and np.isfinite(spatial).all()
            and np.isfinite(components).all()):
        raise ValueError(f"non-finite C2 state series: {view.path}")
    return phi, spatial, components


def _identity_from_view(view, states: np.ndarray) -> Tuple[int, str, int]:
    n_agents = int(states.shape[1])
    recorded_n = _view_value(view, ("N", "num_agents"))
    if recorded_n is not None and int(recorded_n) != n_agents:
        raise ValueError(f"episode N metadata disagrees with state shape: {view.path}")

    policy = _view_value(view, ("policy", "policy_variant", "variant"))
    if policy is None:
        policy = view.path.parent.name
    seed = _view_value(view, ("seed",))
    if seed is None:
        match = re.search(r"(?:seed_|_s)(\d+)", view.path.stem)
        if match:
            seed = int(match.group(1))
    if seed is None:
        raise ValueError(f"episode seed is unavailable: {view.path}")
    return n_agents, str(policy), int(seed)


def analyze_episode(path: os.PathLike, manifest: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load and analyze one full episode, returning row and temporal series."""
    manifest = manifest or {}
    view = load_episode(path)
    if not view.full:
        raise ValueError(f"control effort requires a full artifact: {view.path}")
    states = np.asarray(view["agent_states"], dtype=np.float64)
    controls = np.asarray(view["control_inputs"], dtype=np.float64)
    if states.ndim != 3 or states.shape[2] < 4:
        raise ValueError(f"agent_states must have shape (T+1,N,>=4): {view.path}")
    if controls.shape != (states.shape[0] - 1, states.shape[1]):
        raise ValueError(f"control/state shape mismatch: {view.path}")

    n_agents, policy, seed = _identity_from_view(view, states)
    dt = _required_physical_value(view, manifest, "dt", ("dt",), n_agents)
    speed = _required_physical_value(view, manifest, "speed", ("speed",), n_agents)
    r0 = _required_physical_value(
        view, manifest, "r0", ("r0", "proximity_radius"), n_agents)
    phi, spatial, components = _main_c2_series(view, states, r0)

    if "reward" not in view:
        raise ValueError(f"main C2 J requires reward/original_rewards: {view.path}")
    reward = np.asarray(view["reward"])
    if reward.shape == (controls.shape[0],):
        reward = np.concatenate([np.asarray([np.nan], dtype=reward.dtype), reward])
    if reward.shape != (states.shape[0],):
        raise ValueError(f"reward must have T+1 values: {view.path}")
    judgment = judge_episode_c2(phi, spatial, components, reward)
    effort, temporal = compute_effort_metrics(
        controls, judgment["t_fire"], dt=dt, speed=speed)

    expected_horizon = _manifest_value(manifest, ("horizon",), n_agents=n_agents)
    complete_horizon = (
        bool(view.meta.get("official_c2_complete_horizon", not view.legacy))
        and controls.shape[0] >= 6000
        and (expected_horizon is None or controls.shape[0] >= int(expected_horizon))
    )
    initial_hash = hashlib.sha256(
        np.ascontiguousarray(states[0]).view(np.uint8)).hexdigest()
    row = {
        "path": str(view.path),
        "schema_version": view.schema_version,
        "legacy": int(view.legacy),
        "N": n_agents,
        "policy": policy,
        "seed": seed,
        "horizon": int(controls.shape[0]),
        "dt": dt,
        "speed": speed,
        "r0": r0,
        "initial_state_sha256": initial_hash,
        "official_c2_complete_horizon": int(complete_horizon),
        **judgment,
        **effort,
    }
    return {
        "row": row,
        "l1_rate": temporal["l1_rate"],
        "l2_rate": temporal["l2_rate"],
    }


def _is_pure_acs(policy: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(policy).lower())
    return normalized in {"pureacs", "acs", "fullyconnected", "fc"}


def _finite_delta(learned: Any, baseline: Any) -> Tuple[float, float]:
    try:
        learned_value, baseline_value = float(learned), float(baseline)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (np.isfinite(learned_value) and np.isfinite(baseline_value)):
        return float("nan"), float("nan")
    delta = learned_value - baseline_value
    percent = (
        100.0 * delta / baseline_value
        if baseline_value != 0 else float("nan")
    )
    return delta, percent


def paired_vs_pure_acs(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build one matched-seed row per learned policy episode."""
    baselines: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    identities = set()
    for row in records:
        identity = (int(row["N"]), str(row["policy"]), int(row["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate episode identity: {identity}")
        identities.add(identity)
        if _is_pure_acs(row["policy"]):
            key = (int(row["N"]), int(row["seed"]))
            if key in baselines:
                raise ValueError(f"multiple PureACS episodes for N/seed {key}")
            baselines[key] = row

    output = []
    for learned in records:
        if _is_pure_acs(learned["policy"]):
            continue
        key = (int(learned["N"]), int(learned["seed"]))
        if key not in baselines:
            raise ValueError(f"missing paired PureACS episode for N/seed {key}")
        baseline = baselines[key]
        if learned["initial_state_sha256"] != baseline["initial_state_sha256"]:
            raise ValueError(
                f"initial states are not paired for N={key[0]}, seed={key[1]}"
            )
        pair = {
            "N": key[0],
            "policy": learned["policy"],
            "pure_acs_policy": baseline["policy"],
            "seed": key[1],
            "learned_success": int(learned["success"]),
            "pure_acs_success": int(baseline["success"]),
            "both_success": int(bool(learned["success"] and baseline["success"])),
        }
        for metric in SUCCESS_METRICS + HORIZON_METRICS:
            learned_value, baseline_value = learned[metric], baseline[metric]
            if metric in SUCCESS_METRICS and not pair["both_success"]:
                delta, percent = float("nan"), float("nan")
            else:
                delta, percent = _finite_delta(learned_value, baseline_value)
            pair[f"learned_{metric}"] = learned_value
            pair[f"pure_acs_{metric}"] = baseline_value
            pair[f"delta_{metric}"] = delta
            pair[f"percent_change_{metric}"] = percent
        output.append(pair)
    return sorted(output, key=lambda row: (row["N"], str(row["policy"]), row["seed"]))


def summarize_event_aligned(analyses: Sequence[Mapping[str, Any]],
                            offsets: np.ndarray = EVENT_OFFSETS) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for item in analyses:
        if int(item["row"]["success"]):
            groups[(int(item["row"]["N"]), str(item["row"]["policy"]))].append(item)

    rows = []
    for (n_agents, policy), items in sorted(groups.items()):
        dt_values = {float(item["row"]["dt"]) for item in items}
        if len(dt_values) != 1:
            raise ValueError(f"event alignment requires one dt per N/policy: {(n_agents, policy)}")
        dt = next(iter(dt_values))
        for metric in ("l1_rate", "l2_rate"):
            matrix = np.stack([
                event_aligned(item[metric], item["row"]["t_fire"], offsets)
                for item in items
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q25, median, q75 = np.nanpercentile(matrix, [25, 50, 75], axis=0)
                mean = np.nanmean(matrix, axis=0)
            counts = np.sum(np.isfinite(matrix), axis=0)
            for index, offset in enumerate(offsets):
                rows.append({
                    "N": n_agents,
                    "policy": policy,
                    "metric": metric,
                    "offset_steps": int(offset),
                    "offset_seconds": float(offset * dt),
                    "n": int(counts[index]),
                    "mean": float(mean[index]),
                    "q25": float(q25[index]),
                    "median": float(median[index]),
                    "q75": float(q75[index]),
                })
    return rows


def _episode_paths_from_csv(path: Path, root: Path) -> List[Path]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        path_column = next(
            (name for name in ("path", "episode_path", "artifact_path", "raw_path")
             if name in names), None)
        if path_column is None:
            raise ValueError(f"episodes CSV has no artifact path column: {path}")
        output = []
        for row in reader:
            item = Path(row[path_column])
            output.append(item if item.is_absolute() else root / item)
    return output


def discover_bundle_episodes(bundle: os.PathLike) -> Tuple[Path, Dict[str, Any], List[Path]]:
    """Discover episodes, preferring a bundle manifest and episodes CSV."""
    supplied = Path(bundle).resolve()
    manifest_path = supplied if supplied.is_file() else supplied / "manifest.json"
    root = manifest_path.parent if manifest_path.is_file() else supplied
    manifest: Dict[str, Any] = {}
    paths: List[Path] = []
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("status") is not None and manifest.get("status") != "completed":
            raise ValueError(
                f"bundle status is {manifest.get('status')!r}, not 'completed': {root}")
        csv_path = next(
            (candidate for candidate in
             (root / "episodes.csv", root / "summaries" / "episodes.csv")
             if candidate.is_file()), None)
        if csv_path is not None:
            paths = _episode_paths_from_csv(csv_path, root)
        else:
            episode_root = root / "episodes"
            if episode_root.is_dir():
                paths = sorted(episode_root.rglob("*.npz"))
    else:
        paths = sorted(root.rglob("*.npz"))

    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError(f"no episode NPZ files found under {root}")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"bundle references missing episode: {missing[0]}")
    return root, manifest, unique


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
                      fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = list(fieldnames or ())
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _policy_order(policies: Iterable[str]) -> List[str]:
    def key(policy: str):
        normalized = str(policy).lower()
        if _is_pure_acs(policy):
            return (0, normalized)
        if "determin" in normalized:
            return (1, normalized)
        if "stoch" in normalized:
            return (2, normalized)
        return (3, normalized)
    return sorted(set(str(policy) for policy in policies), key=key)


def create_static_plots(records: Sequence[Mapping[str, Any]],
                        pairs: Sequence[Mapping[str, Any]],
                        event_rows: Sequence[Mapping[str, Any]],
                        output_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    policies = _policy_order(row["policy"] for row in records)
    populations = sorted(set(int(row["N"]) for row in records))
    colors = {policy: plt.get_cmap("tab10")(index % 10)
              for index, policy in enumerate(policies)}
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for axis, metric, title in (
        (axes[0], "env_l1_cost_to_c2", "Environment L1 cost to main C2"),
        (axes[1], "l2_energy_to_c2", "Quadratic L2 energy to main C2"),
    ):
        positions, labels, data, facecolors = [], [], [], []
        cursor = 1
        for n_agents in populations:
            for policy in policies:
                values = np.asarray([
                    row[metric] for row in records
                    if int(row["N"]) == n_agents and row["policy"] == policy
                    and np.isfinite(float(row[metric]))
                ], dtype=np.float64)
                if values.size:
                    positions.append(cursor)
                    labels.append(f"N{n_agents}\n{policy}")
                    data.append(values)
                    facecolors.append(colors[policy])
                cursor += 1
            cursor += 1
        if data:
            boxes = axis.boxplot(data, positions=positions, widths=0.7,
                                 patch_artist=True, showfliers=False)
            for box, color in zip(boxes["boxes"], facecolors):
                box.set_facecolor(color)
                box.set_alpha(0.7)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "effort_to_main_c2.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    learned_policies = _policy_order(row["policy"] for row in pairs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    for axis, metric, title in (
        (axes[0], "env_l1_cost_to_c2", "Paired L1 change vs PureACS"),
        (axes[1], "l2_energy_to_c2", "Paired L2 change vs PureACS"),
    ):
        positions, labels, data = [], [], []
        for n_index, n_agents in enumerate(populations):
            for p_index, policy in enumerate(learned_policies):
                values = np.asarray([
                    row[f"percent_change_{metric}"] for row in pairs
                    if int(row["N"]) == n_agents and row["policy"] == policy
                    and int(row["both_success"])
                    and np.isfinite(float(row[f"percent_change_{metric}"]))
                ], dtype=np.float64)
                if values.size:
                    positions.append(n_index * (len(learned_policies) + 1) + p_index + 1)
                    labels.append(f"N{n_agents}\n{policy}")
                    data.append(values)
        if data:
            axis.boxplot(data, positions=positions, widths=0.7, showfliers=False)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("percent change")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "paired_vs_pure_acs.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, pre_metric, post_metric, title in (
        (axes[0], "pre100_mean_abs", "post100_mean_abs",
         "Mean |u| around main C2"),
        (axes[1], "pre100_mean_square", "post100_mean_square",
         "Mean u² around main C2"),
    ):
        positions, labels = [], []
        cursor = 0
        for n_agents in populations:
            for policy in policies:
                selected = [row for row in records
                            if int(row["N"]) == n_agents
                            and row["policy"] == policy and int(row["success"])]
                if not selected:
                    cursor += 1
                    continue
                pre = np.asarray([row[pre_metric] for row in selected], dtype=np.float64)
                post = np.asarray([row[post_metric] for row in selected], dtype=np.float64)
                finite = np.isfinite(pre) & np.isfinite(post)
                if np.any(finite):
                    x = np.asarray([cursor, cursor + 0.35])
                    for left, right in zip(pre[finite], post[finite]):
                        axis.plot(x, [left, right], color=colors[policy], alpha=0.15,
                                  linewidth=0.8)
                    axis.plot(x, [np.mean(pre[finite]), np.mean(post[finite])],
                              color=colors[policy], marker="o", linewidth=2.0)
                    positions.extend(x.tolist())
                    labels.extend([f"N{n_agents}\n{policy}\npre",
                                   f"N{n_agents}\n{policy}\npost"])
                cursor += 1
            cursor += 1
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "pre_post_main_c2.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)

    if event_rows:
        fig, axes = plt.subplots(2, len(populations),
                                 figsize=(5.2 * len(populations), 8), squeeze=False)
        for col, n_agents in enumerate(populations):
            for row_index, metric in enumerate(("l1_rate", "l2_rate")):
                axis = axes[row_index, col]
                for policy in policies:
                    rows = [row for row in event_rows
                            if int(row["N"]) == n_agents
                            and row["policy"] == policy and row["metric"] == metric]
                    if not rows:
                        continue
                    x = np.asarray([row["offset_seconds"] for row in rows])
                    median = np.asarray([row["median"] for row in rows])
                    q25 = np.asarray([row["q25"] for row in rows])
                    q75 = np.asarray([row["q75"] for row in rows])
                    axis.plot(x, median, label=policy, color=colors[policy])
                    axis.fill_between(x, q25, q75, color=colors[policy], alpha=0.16)
                axis.axvline(0.0, color="black", linewidth=0.8)
                axis.set_title(f"N={n_agents}, {metric}")
                axis.set_xlabel("physical time from main C2 (s)")
                axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "event_aligned_control.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def create_simple_animations(analyses: Sequence[Mapping[str, Any]], output_dir: Path,
                             stride: int, fps: int) -> List[Path]:
    """Render one matched-seed position MP4 per population when ffmpeg exists."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if not animation.writers.is_available("ffmpeg"):
        warnings.warn("--animations requested but matplotlib ffmpeg writer is unavailable")
        return []
    if stride < 1 or fps < 1:
        raise ValueError("animation stride and fps must be positive")

    by_key = {(item["row"]["N"], item["row"]["policy"], item["row"]["seed"]): item
              for item in analyses}
    policies = _policy_order(item["row"]["policy"] for item in analyses)
    outputs = []
    animation_dir = output_dir / "animations"
    animation_dir.mkdir(parents=True, exist_ok=True)
    for n_agents in sorted(set(int(item["row"]["N"]) for item in analyses)):
        seeds = sorted(set(int(item["row"]["seed"]) for item in analyses
                           if int(item["row"]["N"]) == n_agents))
        seed = next((candidate for candidate in seeds
                     if all((n_agents, policy, candidate) in by_key for policy in policies)), None)
        if seed is None:
            continue
        items = [by_key[(n_agents, policy, seed)] for policy in policies]
        views = [load_episode(item["row"]["path"]) for item in items]
        states = [np.asarray(view["agent_states"], dtype=np.float64) for view in views]
        horizon = min(array.shape[0] - 1 for array in states)
        frame_indices = np.arange(0, horizon + 1, stride, dtype=int)
        all_positions = np.concatenate([array[:horizon + 1, :, :2].reshape(-1, 2)
                                        for array in states], axis=0)
        lo, hi = all_positions.min(axis=0), all_positions.max(axis=0)
        padding = max(float(np.max(hi - lo)) * 0.05, 1.0)
        fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4.8), squeeze=False)
        scatters = []
        for axis, policy, state in zip(axes[0], policies, states):
            scatter = axis.scatter(state[0, :, 0], state[0, :, 1], s=24)
            scatters.append(scatter)
            axis.set_xlim(lo[0] - padding, hi[0] + padding)
            axis.set_ylim(lo[1] - padding, hi[1] + padding)
            axis.set_aspect("equal")
            axis.set_title(policy)

        def update(frame_number):
            state_index = int(frame_indices[frame_number])
            for axis, scatter, state, item in zip(axes[0], scatters, states, items):
                scatter.set_offsets(state[state_index, :, :2])
                reached = int(item["row"]["t_fire"]) >= 0 and state_index >= int(item["row"]["t_fire"])
                axis.set_xlabel(
                    f"t={state_index * float(item['row']['dt']):.1f}s"
                    + (" · C2" if reached else ""))
            return scatters

        movie = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
        path = animation_dir / f"N{n_agents}_seed_{seed:05d}.mp4"
        movie.save(path, writer=animation.FFMpegWriter(fps=fps), dpi=120)
        plt.close(fig)
        outputs.append(path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze main-C2 L1/L2 control effort in a full episode bundle")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="bundle directory or its manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=("default for canonical bundles: "
                              "<bundle>/analysis/control_effort; required "
                              "outside legacy/noncanonical inputs"))
    parser.add_argument("--animations", action="store_true",
                        help="also render one simple matched-seed MP4 per N")
    parser.add_argument("--animation-stride", type=int, default=10)
    parser.add_argument("--animation-fps", type=int, default=15)
    return parser


def run_from_bundle(run: os.PathLike, output: Optional[os.PathLike] = None,
                    animations: bool = False, *, animation_stride: int = 10,
                    animation_fps: int = 15) -> Dict[str, Any]:
    """Analyze a population bundle and return JSON-serializable outputs.

    This is the programmatic entry point used by ``python -m eval
    control-effort``.  ``dt``, ``speed``, and ``r0`` are resolved from episode
    metadata first and the bundle manifest second; this function supplies no
    physical defaults.
    """
    root, manifest, episode_paths = discover_bundle_episodes(run)
    output_dir = resolve_analysis_output(
        root, manifest, output, "analysis", "control_effort")

    analyses = [analyze_episode(path, manifest) for path in episode_paths]
    records = sorted((item["row"] for item in analyses),
                     key=lambda row: (row["N"], str(row["policy"]), row["seed"]))
    pairs = paired_vs_pure_acs(records)
    event_rows = summarize_event_aligned(analyses)

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv = output_dir / "episode_control_effort.csv"
    pairs_csv = output_dir / "paired_vs_pure_acs.csv"
    event_csv = output_dir / "event_aligned.csv"
    pair_columns = [
        "N", "policy", "pure_acs_policy", "seed", "learned_success",
        "pure_acs_success", "both_success",
    ]
    for metric in SUCCESS_METRICS + HORIZON_METRICS:
        pair_columns.extend([
            f"learned_{metric}", f"pure_acs_{metric}", f"delta_{metric}",
            f"percent_change_{metric}",
        ])
    event_columns = [
        "N", "policy", "metric", "offset_steps", "offset_seconds", "n",
        "mean", "q25", "median", "q75",
    ]
    persisted_records = []
    for record in records:
        persisted = dict(record)
        persisted["path"] = os.path.relpath(record["path"], root)
        persisted_records.append(persisted)
    _atomic_write_csv(episodes_csv, persisted_records)
    _atomic_write_csv(pairs_csv, pairs, pair_columns)
    _atomic_write_csv(event_csv, event_rows, event_columns)
    plots = create_static_plots(records, pairs, event_rows, output_dir)
    animation_paths = (
        create_simple_animations(
            analyses, output_dir, stride=animation_stride, fps=animation_fps)
        if animations else []
    )
    absolute_outputs = {
        "episodes_csv": str(episodes_csv),
        "paired_csv": str(pairs_csv),
        "event_aligned_csv": str(event_csv),
        "plots": [str(path) for path in plots],
        "animations": [str(path) for path in animation_paths],
    }
    persisted_outputs = {
        "episodes_csv": os.path.relpath(episodes_csv, output_dir),
        "paired_csv": os.path.relpath(pairs_csv, output_dir),
        "event_aligned_csv": os.path.relpath(event_csv, output_dir),
        "plots": [os.path.relpath(path, output_dir) for path in plots],
        "animations": [os.path.relpath(path, output_dir) for path in animation_paths],
    }
    analysis_manifest = {
        "schema_version": "main-c2-control-effort-1.0",
        "path_base": "control_effort_manifest_directory",
        "source_bundle": os.path.relpath(root, output_dir),
        "source_run_id": manifest.get("run_id"),
        "source_fingerprint": manifest.get("fingerprint"),
        "protocol": {
            "id": "main_c2_v1",
            "phi_goal": MAIN_C2_V1.phi_goal,
            "alignment_window": MAIN_C2_V1.alignment_window,
            "stability_window": MAIN_C2_V1.stability_window,
            "spatial_band_epsilon": MAIN_C2_V1.spatial_band_epsilon,
        },
        "episode_count": len(records),
        "paired_count": len(pairs),
        "outputs": persisted_outputs,
    }
    analysis_manifest_path = output_dir / "control_effort_manifest.json"
    atomic_write_json(analysis_manifest_path, analysis_manifest)
    return {
        "schema_version": analysis_manifest["schema_version"],
        "episodes": len(records),
        "pairs": len(pairs),
        "plots": len(plots),
        "animations": len(animation_paths),
        "output_dir": str(output_dir),
        "manifest": str(analysis_manifest_path),
        "outputs": absolute_outputs,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_from_bundle(
        args.bundle, output=args.output_dir, animations=args.animations,
        animation_stride=args.animation_stride, animation_fps=args.animation_fps)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
