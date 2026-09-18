"""Robustness x control effort for the sparse sensitivity lane.

The sensitivity runner stores compact per-step control diagnostics --
``mean_abs_control`` (the agent-mean |u| of one policy interval),
``max_abs_control`` and ``control_saturation_fraction`` -- instead of the full
``control_inputs`` matrix the population lane keeps.  That is enough for the L1,
peak and saturation effort axes, and this module derives them post hoc for
every episode of a sensitivity bundle, per setting and aligned on the C2 event.

``mean_abs_control`` equals :func:`eval.analysis.control_effort.compute_effort_metrics`'s
``l1_rate`` **only** while ``substeps_per_policy_interval == 1``: the runner
averages |u| over substeps x agents, the population analysis averages over
agents per action step.  Episodes that break that identity are refused, never
warned about, and the integral always uses ``policy_interval_seconds`` as dt.

L2 is not recoverable here.  Only the per-step mean of |u| is stored, so
mean(u^2) cannot be reconstructed from it, and this module deliberately emits no
``l2_*`` / ``rms_*`` column anywhere; use the population suite for quadratic
energy, per-agent, radius or heatmap analysis.

    python -m eval.analysis.sensitivity_effort compute --run <bundle>/sensitivity
    python -m eval.analysis.sensitivity_effort compare --arm dknn=<csv> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval.artifacts import atomic_open, atomic_write_json, load_episode
from eval.protocol import MAIN_C2_V1, judge_episode_c2
from eval.rmetrics import arm_style

from .control_effort import (EVENT_OFFSETS, POST_C2_WINDOW, PRE_C2_WINDOW,
                             event_aligned)


SCHEMA_VERSION = "main-c2-sensitivity-effort-1.0"
COMPARE_SCHEMA_VERSION = "main-c2-sensitivity-effort-compare-1.0"
SPARSE_SCHEMA = "main-c2-sensitivity-sparse-1.0"
BUNDLE_SCHEMA = "sensitivity-bundle-1.0"
SUITE_ID = "sensitivity_v1"
REFERENCE_POLICY = "pure_acs"
DEFAULT_COMPARE_EXCLUSIONS = ("num_agents_80",)
# The recorded peak is a float32 reduction of a float64 cap, so the cap check
# uses the same relative slack as eval.rmetrics.effort_columns.
U_MAX_RTOL = 1e-6
L2_NOTE = ("only the per-step agent-mean |u| is stored, so mean(u^2) and any "
           "L2/RMS quantity cannot be reconstructed; use the population suite")
NOT_CONTROLLED_NOTE = ("not a controlled comparison: the arms are different "
                       "checkpoints and also differ in reward and observation "
                       "channels")

# Success-gated metrics are NaN whenever the episode never reached main C2;
# t_fire keeps its stored -1 sentinel in the episode CSV and is gated here.
SUCCESS_METRICS = (
    "t_fire",
    "t_fire_seconds",
    "l1_integral_to_c2",
    "env_l1_cost_to_c2",
    "mean_abs_to_c2",
    "peak_abs_to_c2",
    "sat_frac_to_c2",
    "sat_seconds_to_c2",
    "pre100_mean_abs",
    "post100_mean_abs",
    "pre100_sat_frac",
    "post100_sat_frac",
)
HORIZON_METRICS = (
    "l1_integral_to_c2_or_horizon",
    "env_l1_cost_to_c2_or_horizon",
    "full_mean_abs",
    "peak_abs_full",
    "sat_frac_full",
)
METRICS = SUCCESS_METRICS + HORIZON_METRICS
EVENT_METRICS = ("l1_rate", "sat_rate", "peak_rate")
AGGREGATES = ("n", "mean", "median", "q25", "q75")

# Setting parameters carried on every row: the factor axes of the OAT suite.
# A figure needs the baseline's own value on each axis, and the baseline
# setting belongs to all of them, so the value cannot come from factor_value.
FACTOR_PARAMETERS = (
    "num_agents",
    "minimum_turn_radius",
    "interaction_radius",
    "acs_gain_multiplier",
    "initial_position_bound",
)
PROVENANCE_COLUMNS = (
    "arm", "run_id", "checkpoint_id", "checkpoint_hash", "schema_version",
    "suite", "protocol_id", "mode", "factor", "factor_value", "setting_id",
    "memberships", "N", "num_agents", "minimum_turn_radius",
    "interaction_radius", "acs_gain_multiplier", "initial_position_bound",
    "policy", "policy_short", "seed", "horizon", "dt", "speed", "u_max",
    "initial_state_sha256", "path",
)
OUTCOME_COLUMNS = ("t_fire", "t_fire_seconds", "success", "nonconverged", "J")
# t_fire/t_fire_seconds are outcome columns and also aggregate as metrics;
# the episode header keeps one copy of each, in its outcome position.
EPISODE_COLUMNS = (PROVENANCE_COLUMNS + OUTCOME_COLUMNS
                   + tuple(metric for metric in METRICS
                           if metric not in OUTCOME_COLUMNS))
SETTING_IDENTITY = (
    "arm", "run_id", "checkpoint_id", "checkpoint_hash", "mode", "factor",
    "factor_value", "setting_id", "memberships", "N", "num_agents",
    "minimum_turn_radius", "interaction_radius", "acs_gain_multiplier",
    "initial_position_bound", "policy", "policy_short",
)
SETTING_COUNTS = ("episode_count", "success_count", "nonconverged_count",
                  "success_rate")
SETTING_COLUMNS = (SETTING_IDENTITY + SETTING_COUNTS
                   + tuple(f"{metric}_{name}"
                           for metric in METRICS for name in AGGREGATES))
# The population lane pins these ten, in this order; the sensitivity table adds
# arm/setting/policy_short columns around them without reordering them.
EVENT_COLUMNS = ("arm", "setting_id", "N", "policy", "policy_short", "metric",
                 "offset_steps", "offset_seconds", "n", "mean", "q25",
                 "median", "q75")
PAIR_COLUMNS = (("setting_id", "N", "policy", "pure_acs_policy", "seed",
                 "learned_success", "pure_acs_success", "both_success")
                + tuple(f"{prefix}{metric}"
                        for metric in METRICS
                        for prefix in ("learned_", "pure_acs_", "delta_",
                                       "percent_change_")))
CROSS_ARM_COLUMNS = SETTING_COLUMNS
PROVENANCE_ARM_COLUMNS = ("arm", "run_id", "checkpoint_id", "checkpoint_hash",
                          "episode_count", "setting_count",
                          "nonconverged_count")
EXCLUDED_COLUMNS = ("setting_id", "arm_present", "arm_absent", "reason")

FIGURES = (
    ("l1_to_c2_by_factor.png", "l1_integral_to_c2",
     "L1 control effort to main C2 (rad)"),
    ("sat_frac_full_by_factor.png", "sat_frac_full",
     "turn-rate saturation fraction (full episode)"),
    ("l1_to_c2_or_horizon_by_factor.png", "l1_integral_to_c2_or_horizon",
     "L1 control effort to main C2 or horizon (rad)"),
)
# A log y axis is chosen only when every plotted median is positive and the
# largest is at least this many times the smallest; otherwise one arm would be
# flattened onto the axis (or a zero would be undrawable).
LOG_SCALE_RATIO = 4.0

_NUMERIC_COLUMNS = {
    "factor_value": float, "N": int, "num_agents": int,
    "minimum_turn_radius": float, "interaction_radius": float,
    "acs_gain_multiplier": float, "initial_position_bound": float,
    "seed": int, "horizon": int, "dt": float, "speed": float, "u_max": float,
    "t_fire": int, "success": int, "nonconverged": int,
}


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if np.asarray(values).size else float("nan")


def _policy_short(policy: str) -> str:
    return str(policy).removeprefix("learned_")


def compute_sparse_effort_metrics(mean_abs_control: np.ndarray,
                                  max_abs_control: np.ndarray,
                                  saturation_fraction: np.ndarray,
                                  t_fire: int, dt: float, speed: float,
                                  u_max: float) -> Tuple[Dict[str, float],
                                                         Dict[str, np.ndarray]]:
    """Per-episode L1 / peak / saturation effort from the sparse record.

    The three inputs are the per-policy-step series the sensitivity runner
    stores: the agent-mean |u|, the agent-max |u| and the fraction of agents at
    the turn-rate cap.  ``t_fire`` indexes a T+1 state series, so the
    effort-to-C2 slice is ``series[:t_fire]`` exactly as in
    :mod:`eval.analysis.control_effort`, and ``dt`` must be the policy interval.

    Integrals and window means are accumulated in float64; the three
    whole-episode reductions (``full_mean_abs``, ``peak_abs_full``,
    ``sat_frac_full``) are taken in the input's own dtype so that a float32
    bundle series reproduces its ``summaries/episodes.csv`` value bit for bit.

    Deliberately emits no ``l2_energy`` / ``l2_norm`` / ``rms`` column: only
    mean|u| is stored, so mean(u^2) is not recoverable.
    """
    native = {
        "mean_abs_control": np.asarray(mean_abs_control),
        "max_abs_control": np.asarray(max_abs_control),
        "control_saturation_fraction": np.asarray(saturation_fraction),
    }
    for name, series in native.items():
        if series.ndim != 1 or series.size == 0:
            raise ValueError(f"{name} must be a non-empty 1-D series")
        if not np.isfinite(np.asarray(series, dtype=np.float64)).all():
            raise ValueError(f"{name} contains non-finite values")
    shapes = {series.shape for series in native.values()}
    if len(shapes) != 1:
        raise ValueError("control diagnostic series have inconsistent lengths")
    for name, value in (("dt", dt), ("speed", speed), ("u_max", u_max)):
        if not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"episode {name} must be a positive finite value")

    m1 = native["mean_abs_control"].astype(np.float64)
    peak = native["max_abs_control"].astype(np.float64)
    sat = native["control_saturation_fraction"].astype(np.float64)
    if float(m1.min()) < 0.0:
        raise ValueError("mean_abs_control must be non-negative")
    if float(sat.min()) < 0.0 or float(sat.max()) > 1.0:
        raise ValueError("control_saturation_fraction must lie in [0, 1]")
    if float(peak.max()) > float(u_max) * (1.0 + U_MAX_RTOL):
        raise ValueError(
            "control magnitude {:.9g} exceeds the configured cap {:.9g}".format(
                float(peak.max()), float(u_max)))

    horizon = int(m1.size)
    success = int(t_fire) >= 0
    if success and int(t_fire) > horizon:
        raise ValueError("C2 firing index exceeds the control horizon")
    stop = int(t_fire) if success else horizon
    event_index = max(0, int(t_fire) - 1) if success else horizon
    pre = slice(max(0, event_index - PRE_C2_WINDOW), event_index)
    post = slice(event_index, min(horizon, event_index + POST_C2_WINDOW))

    l1_integral = float(dt * np.sum(m1[:stop]))
    success_value = lambda value: value if success else float("nan")
    metrics = {
        "l1_integral_to_c2": success_value(l1_integral),
        "env_l1_cost_to_c2": success_value(float(speed * l1_integral)),
        "mean_abs_to_c2": success_value(_mean_or_nan(m1[:stop])),
        "peak_abs_to_c2": success_value(
            float(np.max(peak[:stop])) if stop else float("nan")),
        "sat_frac_to_c2": success_value(_mean_or_nan(sat[:stop])),
        "sat_seconds_to_c2": success_value(float(dt * np.sum(sat[:stop]))),
        "l1_integral_to_c2_or_horizon": l1_integral,
        "env_l1_cost_to_c2_or_horizon": float(speed * l1_integral),
        "pre100_mean_abs": success_value(_mean_or_nan(m1[pre])),
        "post100_mean_abs": success_value(_mean_or_nan(m1[post])),
        "pre100_sat_frac": success_value(_mean_or_nan(sat[pre])),
        "post100_sat_frac": success_value(_mean_or_nan(sat[post])),
        "full_mean_abs": float(np.mean(native["mean_abs_control"])),
        "peak_abs_full": float(np.max(native["max_abs_control"])),
        "sat_frac_full": float(np.mean(native["control_saturation_fraction"])),
    }
    temporal = {"l1_rate": m1, "sat_rate": sat, "peak_rate": peak}
    return metrics, temporal


def _meta_value(meta: Mapping[str, Any], name: str, path: Any) -> Any:
    if name not in meta or meta[name] is None:
        raise ValueError(f"episode metadata has no {name}: {path}")
    return meta[name]


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        return value
    item = array.item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def _same_number(actual: Any, expected: Any) -> bool:
    left, right = float(actual), float(expected)
    if np.isnan(left) and np.isnan(right):
        return True
    return left == right


def analyze_sparse_episode(path: os.PathLike, *, arm: Optional[str] = None,
                           recompute_judgment: bool = True) -> Dict[str, Any]:
    """Load one sparse sensitivity episode and derive its effort row.

    Returns ``{"row": {...}, "l1_rate": (T,), "sat_rate": (T,),
    "peak_rate": (T,)}``.  Every contract violation raises: a foreign schema, a
    multi-substep policy interval, a policy interval that is not the dynamics
    step, a control magnitude above the configured cap, and -- unless
    ``recompute_judgment`` is false -- a stored outcome that the main C2 judge
    does not reproduce from the stored series.
    """
    view = load_episode(path)
    if view.schema_version != SPARSE_SCHEMA:
        raise ValueError(
            f"sensitivity effort requires {SPARSE_SCHEMA} episodes, got "
            f"{view.schema_version!r}: {view.path}")
    meta = view.meta
    substeps = int(_meta_value(meta, "substeps_per_policy_interval", view.path))
    if substeps != 1:
        raise ValueError(
            "mean_abs_control averages over substeps x agents, so it is only "
            f"the per-step agent-mean |u| at substeps_per_policy_interval == 1; "
            f"got {substeps}: {view.path}")
    dt = float(_meta_value(meta, "policy_interval_seconds", view.path))
    dynamics_dt = float(_meta_value(meta, "dynamics_dt", view.path))
    if dt != dynamics_dt:
        raise ValueError(
            f"policy_interval_seconds {dt!r} != dynamics_dt {dynamics_dt!r}: "
            f"{view.path}")
    horizon = int(_meta_value(meta, "horizon", view.path))
    speed = float(_meta_value(meta, "speed", view.path))
    u_max = float(_meta_value(meta, "max_turn_rate", view.path))

    for name in ("phi", "s_ent", "n_comp_r0", "reward"):
        if name not in view or np.asarray(view[name]).shape != (horizon + 1,):
            raise ValueError(f"{name} must hold horizon+1 values: {view.path}")
    for name in ("mean_abs_control", "max_abs_control",
                 "control_saturation_fraction"):
        if name not in view or np.asarray(view[name]).shape != (horizon,):
            raise ValueError(f"{name} must hold horizon values: {view.path}")

    stored = {
        "t_fire": int(_scalar(view["t_fire"])),
        "success": int(_scalar(view["success"])),
        "J": float(_scalar(view["J"])),
    }
    if recompute_judgment:
        judgment = judge_episode_c2(view["phi"], view["s_ent"],
                                    view["n_comp_r0"], view["reward"])
        if (int(judgment["t_fire"]) != stored["t_fire"]
                or int(judgment["success"]) != stored["success"]
                or not _same_number(judgment["J"], stored["J"])):
            raise ValueError(
                f"recomputed main C2 judgment {judgment} disagrees with the "
                f"stored outcome {stored}: {view.path}")
    else:
        judgment = dict(stored)

    metrics, temporal = compute_sparse_effort_metrics(
        view["mean_abs_control"], view["max_abs_control"],
        view["control_saturation_fraction"], judgment["t_fire"], dt=dt,
        speed=speed, u_max=u_max)

    policy = str(_meta_value(meta, "policy", view.path))
    t_fire = int(judgment["t_fire"])
    success = int(judgment["success"])
    memberships = _meta_value(meta, "memberships", view.path)
    row = {
        "arm": "" if arm is None else str(arm),
        "run_id": str(_meta_value(meta, "run_id", view.path)),
        "checkpoint_id": str(_meta_value(meta, "checkpoint_id", view.path)),
        "checkpoint_hash": str(_meta_value(meta, "checkpoint_hash", view.path)),
        "schema_version": view.schema_version,
        "suite": str(_meta_value(meta, "suite", view.path)),
        "protocol_id": str(_meta_value(meta, "protocol_id", view.path)),
        "mode": str(_meta_value(meta, "mode", view.path)),
        "factor": str(_meta_value(meta, "factor", view.path)),
        "factor_value": float(_meta_value(meta, "factor_value", view.path)),
        "setting_id": str(_meta_value(meta, "setting_id", view.path)),
        "memberships": (json.dumps(list(memberships), separators=(",", ":"))
                        if isinstance(memberships, (list, tuple))
                        else str(memberships)),
        "N": int(_meta_value(meta, "num_agents", view.path)),
        "num_agents": int(_meta_value(meta, "num_agents", view.path)),
        "policy": policy,
        "policy_short": _policy_short(policy),
        "seed": int(_meta_value(meta, "seed", view.path)),
        "horizon": horizon,
        "dt": dt,
        "speed": speed,
        "u_max": u_max,
        "initial_state_sha256": str(
            _meta_value(meta, "initial_state_sha256", view.path)),
        "path": str(view.path),
        "t_fire": t_fire,
        "t_fire_seconds": float(t_fire * dt) if success else float("nan"),
        "success": success,
        "nonconverged": int(not success),
        "J": float(judgment["J"]),
        **metrics,
    }
    for name in FACTOR_PARAMETERS:
        row[name] = float(_meta_value(meta, name, view.path))
    row["num_agents"] = int(row["num_agents"])
    return {"row": row, **temporal}


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    """Success-gate the metrics that only mean something after main C2."""
    if metric in SUCCESS_METRICS and not int(row["success"]):
        return float("nan")
    try:
        return float(row[metric])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)],
                        dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "q25": float("nan"), "q75": float("nan")}
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    return {"n": int(finite.size), "mean": float(np.mean(finite)),
            "median": float(median), "q25": float(q25), "q75": float(q75)}


def summarize_by_setting(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (setting_id, policy), keeping non-convergence visible.

    Every metric carries its own ``_n``, and ``nonconverged_count`` is a column
    rather than a silent drop: the ``*_to_c2`` aggregates of a cell that never
    reached C2 are NaN with ``_n == 0``, while its ``*_or_horizon`` aggregates
    are finite over the whole cell.
    """
    groups: Dict[Tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[tuple(row[name] for name in SETTING_IDENTITY)].append(row)

    output = []
    for identity, rows in sorted(groups.items(), key=lambda item: [
            str(value) for value in item[0]]):
        summary = dict(zip(SETTING_IDENTITY, identity))
        success_count = sum(int(row["success"]) for row in rows)
        nonconverged_count = sum(int(row["nonconverged"]) for row in rows)
        if success_count + nonconverged_count != len(rows):
            raise ValueError(
                f"success/nonconverged counts do not add up for {identity}")
        summary.update({
            "episode_count": len(rows),
            "success_count": success_count,
            "nonconverged_count": nonconverged_count,
            "success_rate": success_count / len(rows),
        })
        for metric in METRICS:
            stats = _quantiles([_metric_value(row, metric) for row in rows])
            for name in AGGREGATES:
                summary[f"{metric}_{name}"] = stats[name]
        output.append(summary)
    return output


class _EventAlignedAccumulator:
    """Keep only the aligned offset window of each successful episode.

    Retaining the 6000-step series of every episode would cost gigabytes, so
    each successful episode contributes one ``offsets``-long window per metric.
    The windows are held at the lane's own storage precision (float32) and cast
    back to float64 per cell before the quantiles are taken.
    """

    def __init__(self, offsets: np.ndarray = EVENT_OFFSETS):
        self.offsets = np.asarray(offsets, dtype=np.int32)
        self._groups: Dict[Tuple, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: {metric: [] for metric in EVENT_METRICS})
        self._dt: Dict[Tuple, float] = {}

    def add(self, row: Mapping[str, Any], series: Mapping[str, np.ndarray]) -> None:
        if not int(row["success"]):
            return
        key = (str(row.get("arm", "")), str(row["setting_id"]), int(row["N"]),
               str(row["policy"]), str(row["policy_short"]))
        dt = float(row["dt"])
        if self._dt.setdefault(key, dt) != dt:
            raise ValueError(f"event alignment requires one dt per cell: {key}")
        group = self._groups[key]
        for metric in EVENT_METRICS:
            group[metric].append(
                event_aligned(series[metric], row["t_fire"],
                              self.offsets).astype(np.float32))

    def rows(self) -> List[Dict[str, Any]]:
        output = []
        for key, group in sorted(self._groups.items()):
            arm, setting_id, n_agents, policy, policy_short = key
            dt = self._dt[key]
            for metric in EVENT_METRICS:
                matrix = np.stack(group[metric]).astype(np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    q25, median, q75 = np.nanpercentile(matrix, [25, 50, 75],
                                                        axis=0)
                    mean = np.nanmean(matrix, axis=0)
                counts = np.sum(np.isfinite(matrix), axis=0)
                for index, offset in enumerate(self.offsets):
                    output.append({
                        "arm": arm, "setting_id": setting_id, "N": n_agents,
                        "policy": policy, "policy_short": policy_short,
                        "metric": metric, "offset_steps": int(offset),
                        "offset_seconds": float(offset * dt),
                        "n": int(counts[index]), "mean": float(mean[index]),
                        "q25": float(q25[index]), "median": float(median[index]),
                        "q75": float(q75[index]),
                    })
        return output


def summarize_event_aligned(items: Sequence[Mapping[str, Any]],
                            offsets: np.ndarray = EVENT_OFFSETS) -> List[Dict[str, Any]]:
    """Median/IQR of the three rate series around main C2, successes only."""
    accumulator = _EventAlignedAccumulator(offsets)
    for item in items:
        accumulator.add(item["row"], item)
    return accumulator.rows()


def _finite_delta(learned: Any, baseline: Any) -> Tuple[float, float]:
    try:
        learned_value, baseline_value = float(learned), float(baseline)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (np.isfinite(learned_value) and np.isfinite(baseline_value)):
        return float("nan"), float("nan")
    delta = learned_value - baseline_value
    percent = (100.0 * delta / baseline_value
               if baseline_value != 0 else float("nan"))
    return delta, percent


def paired_vs_pure_acs(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Match each learned episode to the PureACS episode of the same setup.

    Within one arm the initial state is shared across policies for a given
    (setting_id, seed), which the matching asserts; the comparison is never made
    across arms.
    """
    baselines: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    identities = set()
    for row in records:
        identity = (str(row["setting_id"]), str(row["policy"]), int(row["seed"]))
        if identity in identities:
            raise ValueError(f"duplicate episode identity: {identity}")
        identities.add(identity)
        if str(row["policy_short"]) == REFERENCE_POLICY:
            key = (str(row["setting_id"]), int(row["seed"]))
            if key in baselines:
                raise ValueError(f"multiple PureACS episodes for {key}")
            baselines[key] = row

    output = []
    for learned in records:
        if str(learned["policy_short"]) == REFERENCE_POLICY:
            continue
        key = (str(learned["setting_id"]), int(learned["seed"]))
        if key not in baselines:
            raise ValueError(f"missing paired PureACS episode for {key}")
        baseline = baselines[key]
        if learned["initial_state_sha256"] != baseline["initial_state_sha256"]:
            raise ValueError(
                f"initial states are not paired for setting={key[0]}, "
                f"seed={key[1]}")
        pair = {
            "setting_id": key[0],
            "N": int(learned["N"]),
            "policy": learned["policy"],
            "pure_acs_policy": baseline["policy"],
            "seed": key[1],
            "learned_success": int(learned["success"]),
            "pure_acs_success": int(baseline["success"]),
            "both_success": int(bool(int(learned["success"])
                                     and int(baseline["success"]))),
        }
        for metric in METRICS:
            learned_value = _metric_value(learned, metric)
            baseline_value = _metric_value(baseline, metric)
            if metric in SUCCESS_METRICS and not pair["both_success"]:
                delta, percent = float("nan"), float("nan")
            else:
                delta, percent = _finite_delta(learned_value, baseline_value)
            pair[f"learned_{metric}"] = learned_value
            pair[f"pure_acs_{metric}"] = baseline_value
            pair[f"delta_{metric}"] = delta
            pair[f"percent_change_{metric}"] = percent
        output.append(pair)
    return sorted(output, key=lambda row: (row["setting_id"],
                                           str(row["policy"]), row["seed"]))


# ------------------------------------------------------------------ bundle
def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
               fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def discover_bundle_episodes(run: os.PathLike) -> Tuple[Path, Dict[str, Any],
                                                        List[Dict[str, str]]]:
    """Resolve a sensitivity bundle to its manifest and episodes.csv rows.

    ``eval.analysis.bundle.resolve_analysis_output`` rejects this layout (it is
    a population bundle helper), so discovery reads the sensitivity bundle's own
    manifest and summary table instead.
    """
    supplied = Path(run).expanduser().resolve()
    manifest_path = supplied if supplied.is_file() else supplied / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no sensitivity bundle manifest: {supplied}")
    root = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(
            f"expected a {BUNDLE_SCHEMA} bundle, got "
            f"{manifest.get('schema_version')!r}: {root}")
    if manifest.get("suite") != SUITE_ID:
        raise ValueError(f"expected suite {SUITE_ID}: {root}")
    if manifest.get("status") != "completed":
        raise ValueError(
            f"bundle status is {manifest.get('status')!r}, not 'completed': {root}")
    episodes_csv = root / "summaries" / "episodes.csv"
    if not episodes_csv.is_file():
        raise FileNotFoundError(f"no summaries/episodes.csv under {root}")
    rows = _csv_rows(episodes_csv)
    if not rows:
        raise ValueError(f"summaries/episodes.csv has no rows: {root}")
    return root, manifest, rows


def _analysis_output(root: Path, output: Optional[os.PathLike]) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return root / "analysis" / "control_effort"


BRIDGE_COLUMNS = ("path", "setting_id", "policy", "seed", "t_fire", "success",
                  "mean_abs_control", "max_abs_control",
                  "control_saturation_fraction")


def _bridge_reference(row: Mapping[str, str]) -> Dict[str, str]:
    """Keep only the summary columns the bridge checks, not the whole row."""
    missing = [name for name in BRIDGE_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"summaries/episodes.csv lacks {missing}")
    return {name: row[name] for name in BRIDGE_COLUMNS}


def _cross_check_episodes_csv(row: Mapping[str, Any],
                              reference: Mapping[str, str]) -> None:
    """Bit-exact bridge to the bundle's own per-episode reductions."""
    for derived, stored in (("full_mean_abs", "mean_abs_control"),
                            ("peak_abs_full", "max_abs_control"),
                            ("sat_frac_full", "control_saturation_fraction")):
        if float(row[derived]) != float(reference[stored]):
            raise ValueError(
                f"{derived} {row[derived]!r} != summaries/episodes.csv "
                f"{stored} {reference[stored]!r}: {row['path']}")
    if int(row["t_fire"]) != int(reference["t_fire"]) or (
            int(row["success"]) != int(reference["success"])):
        raise ValueError(
            f"outcome disagrees with summaries/episodes.csv: {row['path']}")


def run_from_bundle(run: os.PathLike, output: Optional[os.PathLike] = None, *,
                    arm: Optional[str] = None, event_aligned: bool = True,
                    settings: Optional[Sequence[str]] = None,
                    policies: Optional[Sequence[str]] = None,
                    seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Derive the sparse effort tables for one sensitivity bundle.

    Episodes are streamed: only the event-aligned offset window of a successful
    episode is retained, never the 6000-step series.  Every episode is also
    checked against ``summaries/episodes.csv``, whose whole-episode reductions
    must match bit for bit.
    """
    root, manifest, csv_rows = discover_bundle_episodes(run)
    output_dir = _analysis_output(root, output)
    wanted_settings = None if settings is None else {str(s) for s in settings}
    wanted_policies = None if policies is None else {str(p) for p in policies}
    wanted_seeds = None if seeds is None else {int(s) for s in seeds}

    selected = [
        _bridge_reference(reference) for reference in csv_rows
        if (wanted_settings is None or reference["setting_id"] in wanted_settings)
        and (wanted_policies is None or reference["policy"] in wanted_policies)
        and (wanted_seeds is None or int(reference["seed"]) in wanted_seeds)
    ]
    del csv_rows
    if not selected:
        raise ValueError(f"no episodes selected in {root}")

    accumulator = _EventAlignedAccumulator()
    records: List[Dict[str, Any]] = []
    for reference in selected:
        path = Path(reference["path"])
        analysis = analyze_sparse_episode(
            path if path.is_absolute() else root / path, arm=arm)
        row = analysis["row"]
        _cross_check_episodes_csv(row, reference)
        if event_aligned:
            accumulator.add(row, analysis)
        records.append(row)
    records.sort(key=lambda row: (str(row["setting_id"]), str(row["policy"]),
                                  int(row["seed"])))

    settings_rows = summarize_by_setting(records)
    pairs = paired_vs_pure_acs(records)
    event_rows = accumulator.rows() if event_aligned else []

    output_dir.mkdir(parents=True, exist_ok=True)
    persisted = []
    for row in records:
        item = dict(row)
        item["path"] = os.path.relpath(row["path"], root)
        persisted.append(item)
    episodes_out = _write_csv(output_dir / "episode_sensitivity_effort.csv",
                              persisted, EPISODE_COLUMNS)
    settings_out = _write_csv(output_dir / "setting_effort.csv", settings_rows,
                              SETTING_COLUMNS)
    pairs_out = _write_csv(output_dir / "paired_vs_pure_acs.csv", pairs,
                           PAIR_COLUMNS)
    outputs = {
        "episodes_csv": str(episodes_out),
        "setting_csv": str(settings_out),
        "paired_csv": str(pairs_out),
    }
    if event_aligned:
        outputs["event_aligned_csv"] = str(
            _write_csv(output_dir / "event_aligned.csv", event_rows,
                       EVENT_COLUMNS))

    nonconverged = sum(int(row["nonconverged"]) for row in records)
    checkpoints = sorted({row["checkpoint_id"] for row in records})
    hashes = sorted({row["checkpoint_hash"] for row in records})
    if len(checkpoints) != 1 or len(hashes) != 1:
        raise ValueError(
            f"one sensitivity bundle is one checkpoint; found {checkpoints} "
            f"/ {hashes}: {root}")
    analysis_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_bundle": os.path.relpath(root, output_dir),
        "source_run_id": manifest.get("run_id"),
        "source_fingerprint": manifest.get("fingerprint"),
        "arm": arm,
        "checkpoint_id": checkpoints[0],
        "checkpoint_hash": hashes[0],
        "protocol": {
            "id": str(records[0]["protocol_id"]),
            "phi_goal": MAIN_C2_V1.phi_goal,
            "alignment_window": MAIN_C2_V1.alignment_window,
            "stability_window": MAIN_C2_V1.stability_window,
            "spatial_band_epsilon": MAIN_C2_V1.spatial_band_epsilon,
        },
        "l2_recoverable": False,
        "l2_note": L2_NOTE,
        "episode_count": len(records),
        "nonconverged_count": nonconverged,
        "setting_count": len({row["setting_id"] for row in records}),
        "substeps_per_policy_interval": 1,
        "dt_seconds": float(records[0]["dt"]),
        "outputs": {name: os.path.relpath(path, output_dir)
                    for name, path in outputs.items()},
    }
    manifest_path = output_dir / "sensitivity_effort_manifest.json"
    atomic_write_json(manifest_path, analysis_manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "episodes": len(records),
        "nonconverged": nonconverged,
        "settings": analysis_manifest["setting_count"],
        "setting_rows": len(settings_rows),
        "pairs": len(pairs),
        "event_aligned_rows": len(event_rows),
        "outputs": outputs,
    }


# ----------------------------------------------------------------- compare
def _typed_episode_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for raw in _csv_rows(path):
        row: Dict[str, Any] = dict(raw)
        for name, caster in _NUMERIC_COLUMNS.items():
            if name in row and row[name] != "":
                row[name] = caster(row[name])
        for name in ("t_fire_seconds", "J") + METRICS:
            if name in row:
                row[name] = float(row[name]) if row[name] != "" else float("nan")
        rows.append(row)
    return rows


def _require_single_checkpoint(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """One arm is one checkpoint; a pooled arm CSV hides which model ran."""
    hashes = sorted({str(row.get("checkpoint_hash", "")) for row in rows})
    if len(hashes) != 1:
        raise ValueError(
            f"arm {name!r} spans {len(hashes)} checkpoint hashes {hashes}; "
            f"slice the episode CSV to one checkpoint per arm")


def run_compare(arms: Dict[str, Path], out: os.PathLike,
                exclude_settings: Sequence[str] = DEFAULT_COMPARE_EXCLUSIONS
                ) -> Dict[str, Any]:
    """Put the arms' per-setting effort side by side, with provenance.

    The arms are different checkpoints, so nothing here is a controlled
    comparison and no cross-arm delta or percent-change column is emitted; each
    row carries the arm, run and checkpoint that produced it.
    """
    out = Path(out).expanduser().resolve()
    loaded = {name: _typed_episode_rows(Path(path)) for name, path in arms.items()}
    for name, rows in loaded.items():
        if not rows:
            raise ValueError(f"arm {name} has no rows")
        _require_single_checkpoint(name, rows)
        for row in rows:
            row["arm"] = name

    summaries = {name: summarize_by_setting(rows) for name, rows in loaded.items()}
    arm_settings = {name: {str(row["setting_id"]) for row in rows}
                    for name, rows in loaded.items()}
    shared = set.intersection(*arm_settings.values()) if arm_settings else set()
    requested = {str(item) for item in exclude_settings}
    kept = sorted(shared - requested)

    excluded_rows = []
    for setting_id in sorted(set().union(*arm_settings.values()) - set(kept)):
        present = sorted(name for name in loaded if setting_id in arm_settings[name])
        absent = sorted(name for name in loaded if setting_id not in arm_settings[name])
        if absent:
            run_ids = sorted({str(rows[0]["run_id"])
                              for name, rows in loaded.items() if name in absent})
            reason = "absent_in_arm:" + ",".join(run_ids)
        else:
            reason = "excluded_by_request"
        excluded_rows.append({
            "setting_id": setting_id,
            "arm_present": ";".join(present),
            "arm_absent": ";".join(absent),
            "reason": reason,
        })

    cross_rows = []
    for name in sorted(summaries):
        for row in summaries[name]:
            if str(row["setting_id"]) in kept:
                cross_rows.append(row)
    cross_rows.sort(key=lambda row: (str(row["setting_id"]),
                                     str(row["policy"]), str(row["arm"])))

    provenance_rows = []
    for name in sorted(loaded):
        rows = loaded[name]
        provenance_rows.append({
            "arm": name,
            "run_id": str(rows[0]["run_id"]),
            "checkpoint_id": str(rows[0]["checkpoint_id"]),
            "checkpoint_hash": str(rows[0]["checkpoint_hash"]),
            "episode_count": len(rows),
            "setting_count": len(arm_settings[name]),
            "nonconverged_count": sum(int(row["nonconverged"]) for row in rows),
        })

    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "arm_provenance_csv": str(_write_csv(
            out / "arm_provenance.csv", provenance_rows, PROVENANCE_ARM_COLUMNS)),
        "cross_arm_setting_csv": str(_write_csv(
            out / "cross_arm_setting_effort.csv", cross_rows, CROSS_ARM_COLUMNS)),
        "excluded_settings_csv": str(_write_csv(
            out / "excluded_settings.csv", excluded_rows, EXCLUDED_COLUMNS)),
    }
    figures, scales = create_compare_figures(cross_rows, out / "figures")
    outputs["figures"] = [str(path) for path in figures]

    compare_manifest = {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "arms": {name: str(Path(path)) for name, path in arms.items()},
        "controlled_comparison": False,
        "reason": (NOT_CONTROLLED_NOTE if len(arms) > 1
                   else "single arm: nothing is compared across arms"),
        "l2_recoverable": False,
        "l2_note": L2_NOTE,
        "shared_setting_count": len(kept),
        "shared_settings": kept,
        "excluded_settings": [row["setting_id"] for row in excluded_rows],
        "figure_yscale": scales,
        "provenance": provenance_rows,
        "outputs": {name: (os.path.relpath(value, out)
                           if isinstance(value, str)
                           else [os.path.relpath(item, out) for item in value])
                    for name, value in outputs.items()},
    }
    manifest_path = out / "compare_manifest.json"
    atomic_write_json(manifest_path, compare_manifest)
    return {
        "out": str(out),
        "arms": sorted(loaded),
        "controlled_comparison": False,
        "reason": compare_manifest["reason"],
        "shared_settings": len(kept),
        "excluded_settings": [row["setting_id"] for row in excluded_rows],
        "cross_arm_rows": len(cross_rows),
        "figure_yscale": scales,
        "manifest": str(manifest_path),
        "outputs": outputs,
    }


def _factor_axes(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({str(row["factor"]) for row in rows
                   if str(row["factor"]) != "baseline"})


def _axis_series(rows: Sequence[Mapping[str, Any]], factor: str, arm: str,
                 policy_short: str, metric: str) -> Dict[str, np.ndarray]:
    """The (x, median, q25, q75, nonconverged) points of one arm on one axis.

    The baseline setting belongs to every factor axis (its ``memberships`` lists
    them all), so it is placed at its own value of that parameter rather than at
    ``factor_value``.
    """
    selected = [row for row in rows
                if str(row["arm"]) == arm
                and str(row["policy_short"]) == policy_short
                and (str(row["factor"]) == factor
                     or factor in json.loads(str(row["memberships"])))]
    selected.sort(key=lambda row: float(row[factor]))
    return {
        "x": np.asarray([float(row[factor]) for row in selected]),
        "median": np.asarray([float(row[f"{metric}_median"]) for row in selected]),
        "q25": np.asarray([float(row[f"{metric}_q25"]) for row in selected]),
        "q75": np.asarray([float(row[f"{metric}_q75"]) for row in selected]),
        "nonconverged": np.asarray([int(row["nonconverged_count"])
                                    for row in selected]),
        "baseline_x": np.asarray([float(row[factor]) for row in selected
                                  if str(row["setting_id"]) == "baseline"]),
    }


def _choose_yscale(values: Iterable[float]) -> str:
    finite = np.asarray([value for value in values if np.isfinite(value)],
                        dtype=np.float64)
    positive = finite[finite > 0]
    if positive.size == 0 or positive.size != finite.size:
        return "linear"
    return ("log" if float(positive.max() / positive.min()) >= LOG_SCALE_RATIO
            else "linear")


def create_compare_figures(cross_rows: Sequence[Mapping[str, Any]],
                           figure_dir: Path) -> Tuple[List[Path], Dict[str, str]]:
    """One panel per factor axis, one series per arm, caveat in the title."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    arms = sorted({str(row["arm"]) for row in cross_rows})
    factors = _factor_axes(cross_rows)
    acs_styles = ("--", ":", "-.")
    written, scales = [], {}
    for filename, metric, label in FIGURES:
        columns = min(3, max(1, len(factors)))
        rows_count = int(np.ceil(len(factors) / columns)) if factors else 1
        figure, axes = plt.subplots(rows_count, columns, squeeze=False,
                                    figsize=(5.4 * columns, 4.2 * rows_count))
        flat = [axis for row in axes for axis in row]
        series_cache = {}
        for factor in factors:
            for arm in arms:
                for policy_short in ("deterministic", REFERENCE_POLICY):
                    series_cache[(factor, arm, policy_short)] = _axis_series(
                        cross_rows, factor, arm, policy_short, metric)
        scale = _choose_yscale(
            value for series in series_cache.values()
            for value in list(series["median"]) + list(series["q25"]))
        scales[filename] = scale
        for index, factor in enumerate(factors):
            axis = flat[index]
            for arm_index, arm in enumerate(arms):
                style = arm_style(arm)
                for policy_short in ("deterministic", REFERENCE_POLICY):
                    series = series_cache[(factor, arm, policy_short)]
                    if series["x"].size == 0:
                        continue
                    is_acs = policy_short == REFERENCE_POLICY
                    color = (arm_style(REFERENCE_POLICY)["color"] if is_acs
                             else style["color"])
                    linestyle = (acs_styles[arm_index % len(acs_styles)]
                                 if is_acs else style["linestyle"])
                    axis.plot(series["x"], series["median"], color=color,
                              linestyle=linestyle, marker="o", markersize=4,
                              linewidth=1.6,
                              label=(f"pure_acs ({arm})" if is_acs
                                     else f"{arm} learned_deterministic"
                                          f"{style['suffix']}"))
                    axis.fill_between(series["x"], series["q25"], series["q75"],
                                      color=color, alpha=0.14, linewidth=0)
                    for x, y, count in zip(series["x"], series["median"],
                                           series["nonconverged"]):
                        if count > 0 and np.isfinite(y):
                            axis.annotate(f"{int(count)} nc", (x, y),
                                          textcoords="offset points",
                                          xytext=(0, 7), fontsize=6,
                                          color=color, ha="center")
                    if series["baseline_x"].size:
                        axis.axvline(float(series["baseline_x"][0]),
                                     color="#9E9E9E", linewidth=0.8,
                                     linestyle="--", zorder=0)
            axis.set_yscale(scale)
            axis.set_title(f"{factor} (dashed grey = baseline)", fontsize=9)
            axis.set_xlabel(factor, fontsize=8)
            axis.set_ylabel(label, fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.25)
        for axis in flat[len(factors):]:
            axis.axis("off")
        handles, labels = flat[0].get_legend_handles_labels()
        if handles:
            flat[0].legend(handles, labels, fontsize=7)
        figure.suptitle(
            f"{label} by sensitivity factor -- "
            f"not a controlled comparison: different checkpoints", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = figure_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written, scales


# --------------------------------------------------------------------- cli
def _comma_strings(value: str) -> List[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a unique comma-separated list")
    return values


def _seed_range(value: str) -> List[int]:
    from eval.population import parse_int_range

    try:
        return parse_int_range(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arm_argument(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("arm name must not be empty")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.analysis.sensitivity_effort",
        description=("post-hoc L1/peak/saturation control effort for the "
                     "sparse sensitivity lane (no L2: see --help notes)"))
    commands = parser.add_subparsers(dest="command", required=True)

    compute = commands.add_parser(
        "compute", help="derive the effort tables of one sensitivity bundle")
    compute.add_argument("--run", required=True, type=Path,
                         help="sensitivity bundle directory or its manifest.json")
    compute.add_argument("--output", type=Path,
                         help="default: <bundle>/analysis/control_effort")
    compute.add_argument("--arm", help="arm label recorded on every row")
    compute.add_argument("--no-event-aligned", action="store_true",
                         help="skip event_aligned.csv")
    compute.add_argument("--settings", type=_comma_strings)
    compute.add_argument("--policies", type=_comma_strings)
    compute.add_argument("--seeds", type=_seed_range, metavar="A-B|A,B,C")

    compare = commands.add_parser(
        "compare", help="put arms side by side; never a controlled comparison")
    compare.add_argument("--arm", action="append", required=True,
                         type=_arm_argument,
                         metavar="NAME=EPISODE_SENSITIVITY_EFFORT_CSV")
    compare.add_argument("--out", required=True, type=Path)
    compare.add_argument("--exclude-setting", action="append", default=None,
                         metavar="SETTING_ID",
                         help=(f"default: {', '.join(DEFAULT_COMPARE_EXCLUSIONS)}"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compute":
        result = run_from_bundle(
            args.run, output=args.output, arm=args.arm,
            event_aligned=not args.no_event_aligned, settings=args.settings,
            policies=args.policies, seeds=args.seeds)
    else:
        exclusions = (DEFAULT_COMPARE_EXCLUSIONS if args.exclude_setting is None
                      else tuple(args.exclude_setting))
        result = run_compare(dict(args.arm), args.out, exclusions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
