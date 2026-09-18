"""Post-hoc selection-axis (R1-R6) and control-effort metrics.

The rollout loops store raw trajectories; every metric here is derived offline
from those arrays, so no lane recomputes anything online and no existing
summary CSV grows a column.  Derived artifacts are written beside the source
bundle, never inside the directories the discovery helpers glob.

    python -m eval.rmetrics compute --kind population --source <bundle>
    python -m eval.rmetrics table --arm dknn=<csv> --arm full_NN=<csv> --out <csv>
    python -m eval.rmetrics figures --arm dknn=<csv> --out <dir>

Definitions (one episode, T action steps, N active agents):

    S_i(t) = { j != i : binary_actions[t, i, j] }        k_i(t) = |S_i(t)|
    D(t)   = pairwise distances of agent_states[t], diagonal +inf
    R1 r_i(t)        = max_{j in S_i} D[i, j]                    (NaN if k = 0)
    R3 nu_i(t)       = k_i / |{ j != i : D[i, j] <= r_i }|       (NaN if k = 0)
    R4 rank_dev_i(t) = |S_i \\ NN_{k_i}(i)| / k_i                 (NaN if k = 0)
    R5 sel/ref counts per distance rank m = 1..N-1
    R6 asym(t)       = |XOR edges| / |OR edges|,  nonrecip_i(t) per agent

``agent_states[t]`` is the state the policy saw when it emitted
``binary_actions[t]``; the two share the index t, which is the same pairing
``eval.analysis.core.compute_cutoff_radii`` uses for pointer radii.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval.analysis.control_effort import compute_effort_metrics
from eval.analysis.core import compute_cutoff_radii
from eval.protocol import judge_episode_c2


RMETRICS_SCHEMA = "rmetrics-1.0"
WINDOW_STEPS = 300
CHUNK_STEPS = 512
SATURATION_RTOL = 1e-6
SATURATION_ATOL = 1e-8
# The env resolves its cutoff at roughly float32 resolution, so a selection
# disagreement is attributed to precision only when some other agent sits within
# a couple of float32 ulps of the ego's own cutoff radius.  The tolerance is
# relative because an ulp grows with the distance: 1.1e-05 m at 92 m, where the
# widest disagreement in the recorded lanes (1.44e-05 m) was found.
BOUNDARY_RTOL = 2.0 * float(np.finfo(np.float32).eps)
BOUNDARY_ATOL = 1e-9
# eval.eval_c2 stores rank_dev as float32, whose quantum near 0.5 is 5.96e-08.
RANKDEV_TOL = 1e-7
DERIVED_DIRNAME = "derived_rmetrics"
SUMMARY_NAME = "rmetrics_summary.csv"
KINDS = ("population", "c2", "knnref")

ARRAY_KEYS = ("r1", "r1_norm", "r3", "r4", "r5_sel_count", "r5_ref_count",
              "r6_asym", "r6_nonrecip")
WINDOW_NAMES = ("early", "ss", "full")
# Report palette (_sessions/S6_handoff.md §5.2).  k19 / knnref / ACS-FC and
# pure ACS are one visual family.
ARM_COLORS = {
    "dknn":     "#2E7D32",   # green
    "full_nn":  "#C62828",   # red
    "pure_acs": "#37474F",   # dark slate
    "knn12":    "#B07A06",   # amber
}
# Documented fallback: an arm whose name matches no family is drawn neutral grey
# and tagged "(unmapped)" in the legend, rather than borrowing a palette colour.
UNMAPPED_ARM_COLOR = "#9E9E9E"
# First match wins, tested as a substring of the lower-cased arm name.  "acs"
# precedes "dknn"/"fnn" so dknn_acs and fnn_acs both land on pure ACS.
_ARM_PATTERNS = (
    ("acs", "pure_acs"), ("k19", "pure_acs"), ("knnref", "pure_acs"),
    ("knn12", "knn12"), ("k12", "knn12"),
    ("dknn", "dknn"),
    ("full_nn", "full_nn"), ("fullnn", "full_nn"), ("fnn", "full_nn"),
)
R_COLUMNS = ("r1", "r1_norm", "k", "k_norm", "r3", "r4", "r6_asym",
             "r6_nonrecip", "k0_rate")
EFFORT_COLUMNS = (
    "l1_integral_to_c2", "env_l1_cost_to_c2", "l2_energy_to_c2",
    "l2_norm_to_c2", "mean_abs_to_c2", "rms_to_c2",
    "l1_integral_to_c2_or_horizon", "env_l1_cost_to_c2_or_horizon",
    "l2_energy_to_c2_or_horizon", "l2_norm_to_c2_or_horizon",
    "pre100_mean_abs", "post100_mean_abs", "pre100_mean_square",
    "post100_mean_square", "pre100_rms", "post100_rms",
    "full_mean_abs", "full_mean_square",
)
CHECK_COLUMNS = ("check_deg_match", "check_r1_ptr_bitwise", "check_r3_one",
                 "check_r4_zero", "check_r5_ref", "check_rankdev_match",
                 "check_effort_matches_control_effort_csv")


# --------------------------------------------------------------- utilities
def boundary_tolerance(radius: np.ndarray) -> np.ndarray:
    """Distance within which the env's own cutoff decision is not resolvable."""
    return BOUNDARY_ATOL + BOUNDARY_RTOL * np.abs(np.asarray(radius))


def arm_style(name: str) -> Dict[str, object]:
    """Report-palette colour plus the decoding linestyle for one arm label."""
    key = str(name).lower()
    family = next((fam for token, fam in _ARM_PATTERNS if token in key), None)
    stochastic = "_sto" in key or "stochastic" in key
    return {"color": ARM_COLORS.get(family, UNMAPPED_ARM_COLOR),
            "linestyle": "--" if stochastic else "-",
            "family": family or "unmapped",
            "suffix": "" if family else " (unmapped)"}


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def window_slices(horizon: int, window: int = WINDOW_STEPS) -> Dict[str, slice]:
    """Action-index windows; ``ss`` is the last ``window`` steps."""
    return {
        "early": slice(0, min(int(window), int(horizon))),
        "ss": slice(max(0, int(horizon) - int(window)), int(horizon)),
        "full": slice(0, int(horizon)),
    }


def default_max_turn_rate() -> float:
    """The env's own default turn-rate cap, for lanes without a manifest."""
    from envs.env import ControlConfig

    return float(ControlConfig().max_turn_rate)


# ------------------------------------------------------------ R1-R6 arrays
def selection_arrays(states: np.ndarray, binary_actions: np.ndarray,
                     r0: float) -> Tuple[Dict[str, np.ndarray], np.ndarray,
                                         np.ndarray]:
    """Return the derived per-agent arrays, realized degrees, and boundary gaps."""
    masks = np.asarray(binary_actions, dtype=bool)
    if masks.ndim != 3 or masks.shape[1] != masks.shape[2]:
        raise ValueError("binary_actions must have shape (T, N, N)")
    horizon, num_agents = int(masks.shape[0]), int(masks.shape[1])
    states = np.asarray(states, dtype=np.float64)
    if states.shape[:2] != (horizon + 1, num_agents):
        raise ValueError("state/mask shapes are inconsistent")
    if num_agents < 2:
        raise ValueError("selection metrics need at least two agents")
    positions = states[:horizon, :, 0:2]
    eye = np.eye(num_agents, dtype=bool)
    upper = np.triu(np.ones((num_agents, num_agents), dtype=bool), 1)
    ranks = num_agents - 1
    rank_index = np.arange(1, num_agents)

    # not a reported metric: how close another agent sits to the ego's cutoff,
    # used only to tell a precision boundary from a real rank violation
    boundary_gap = np.empty((horizon, num_agents), np.float64)
    out = {
        "r1": np.empty((horizon, num_agents), np.float64),
        "r1_norm": np.empty((horizon, num_agents), np.float64),
        "r3": np.empty((horizon, num_agents), np.float64),
        "r4": np.empty((horizon, num_agents), np.float64),
        "r5_sel_count": np.empty((horizon, ranks), np.int16),
        "r5_ref_count": np.empty((horizon, ranks), np.int16),
        "r6_asym": np.empty(horizon, np.float64),
        "r6_nonrecip": np.empty((horizon, num_agents), np.float64),
    }
    degrees = np.empty((horizon, num_agents), np.int64)

    for start in range(0, horizon, CHUNK_STEPS):
        stop = min(start + CHUNK_STEPS, horizon)
        block = positions[start:stop]
        distance = np.linalg.norm(
            block[:, :, None, :] - block[:, None, :, :], axis=-1)
        distance[:, eye] = np.inf
        selected = masks[start:stop] & ~eye
        degree = selected.sum(-1)
        degrees[start:stop] = degree
        chosen = degree >= 1
        safe_degree = np.where(chosen, degree, 1)

        # R1: farthest selected neighbour; NaN where nothing is selected.
        radius = np.where(selected, distance, -np.inf).max(-1)
        radius = np.where(chosen, radius, np.nan)
        out["r1"][start:stop] = radius
        out["r1_norm"][start:stop] = radius / float(r0)

        # Distance of every other agent from the ego's own cutoff radius; the
        # cutoff target itself contributes an exact zero, so it is masked out.
        offset = np.abs(distance - radius[..., None])
        offset[offset == 0.0] = np.inf
        boundary_gap[start:stop] = np.where(chosen, offset.min(-1), np.inf)

        # R3: how much of the closed disk of radius r_i is actually selected.
        within = (distance <= radius[..., None]).sum(-1)
        out["r3"][start:stop] = np.where(
            chosen, degree / np.where(within > 0, within, 1), np.nan)

        # R4: selected edges outside the k_i nearest, as a fraction of k_i.
        order = np.argsort(distance, axis=-1, kind="stable")
        rank_of = np.empty_like(order)
        np.put_along_axis(rank_of, order, np.arange(num_agents), axis=-1)
        outside = (selected & (rank_of >= degree[..., None])).sum(-1)
        out["r4"][start:stop] = np.where(chosen, outside / safe_degree, np.nan)

        # R5: selection counts by distance rank, and the same-k nearest
        # projection they are read against.
        by_rank = np.take_along_axis(selected, order, axis=-1)
        out["r5_sel_count"][start:stop] = by_rank[:, :, :ranks].sum(1)
        out["r5_ref_count"][start:stop] = (
            degree[:, :, None] >= rank_index[None, None, :]).sum(1)

        # R6: undirected edge asymmetry and the per-agent non-reciprocal share.
        reverse = selected.transpose(0, 2, 1)
        either = (selected | reverse) & upper
        exclusive = (selected ^ reverse) & upper
        edges = either.sum((1, 2))
        out["r6_asym"][start:stop] = np.where(
            edges >= 1, exclusive.sum((1, 2)) / np.where(edges > 0, edges, 1),
            np.nan)
        out["r6_nonrecip"][start:stop] = np.where(
            chosen, (selected & ~reverse).sum(-1) / safe_degree, np.nan)

    return out, degrees, boundary_gap


def window_summary(arrays: Dict[str, np.ndarray], degrees: np.ndarray,
                   num_agents: int) -> Dict[str, float]:
    """Pool every agent-step in a window (NaN-skipping, per PLAN 2.0)."""
    horizon = int(degrees.shape[0])
    columns: Dict[str, float] = {}
    for name, window in window_slices(horizon).items():
        degree = degrees[window]
        columns[f"r1_{name}"] = _nanmean(arrays["r1"][window])
        columns[f"r1_norm_{name}"] = _nanmean(arrays["r1_norm"][window])
        columns[f"k_{name}"] = float(degree.mean()) if degree.size else float("nan")
        columns[f"k_norm_{name}"] = (
            columns[f"k_{name}"] / float(num_agents - 1))
        columns[f"r3_{name}"] = _nanmean(arrays["r3"][window])
        columns[f"r4_{name}"] = _nanmean(arrays["r4"][window])
        columns[f"r6_asym_{name}"] = _nanmean(arrays["r6_asym"][window])
        columns[f"r6_nonrecip_{name}"] = _nanmean(arrays["r6_nonrecip"][window])
        columns[f"k0_rate_{name}"] = (
            float((degree == 0).mean()) if degree.size else float("nan"))
    return columns


# ----------------------------------------------------------- effort axis
def effort_columns(control_inputs: np.ndarray, t_fire: int, dt: float,
                   speed: float, u_max: float) -> Dict[str, float]:
    metrics, temporal = compute_effort_metrics(control_inputs, t_fire, dt, speed)
    control = np.asarray(control_inputs, dtype=np.float64)
    horizon = int(control.shape[0])
    success = int(t_fire) >= 0
    stop = int(t_fire) if success else horizon
    l2_rate = temporal["l2_rate"]
    metrics["rms_to_c2_or_horizon"] = (
        float(np.sqrt(np.mean(l2_rate[:stop]))) if stop else float("nan"))
    saturated = np.isclose(np.abs(control), float(u_max),
                           rtol=SATURATION_RTOL, atol=SATURATION_ATOL)
    metrics["sat_frac_to_c2"] = (
        float(saturated[:stop].mean()) if success and stop else float("nan"))
    metrics["sat_frac_full"] = float(saturated.mean())
    if float(np.abs(control).max()) > float(u_max) * (1.0 + 1e-6):
        raise ValueError(
            "control magnitude {:.9g} exceeds the configured cap {:.9g}".format(
                float(np.abs(control).max()), float(u_max)))
    return metrics


# --------------------------------------------------------- self-validation
def _selection_is_rank_faithful(policy: str, has_pointer: bool,
                                fixed_k: Optional[int]) -> bool:
    """True where the arm selects exactly a nearest set by construction.

    Pointer cutoffs, the all-ones ACS-FC mask and fixed-k nearest all imply
    nu == 1, rank_dev == 0 and sel_count == ref_count.  A learned per-edge mask
    implies none of them, so those flags stay blank for it.
    """
    return bool(has_pointer or fixed_k is not None
                or policy in ("pure_acs", "knn12"))


def self_checks(arrays: Dict[str, np.ndarray], degrees: np.ndarray,
                payload: Dict[str, np.ndarray], rank_faithful: bool,
                num_agents: int, boundary_gap: np.ndarray) -> Dict[str, object]:
    """Verify the invariants a rank-faithful arm must satisfy.

    The environment decides ``d_ij <= d_ip`` at roughly float32 resolution, so a
    float64 recomputation from the stored states can disagree about an agent
    that sits within ``BOUNDARY_TOL`` of the ego's cutoff radius.  Such an
    agent-step is excused; a disagreement anywhere else is a real violation and
    is still reported.
    """
    checks: Dict[str, object] = {name: "" for name in CHECK_COLUMNS}
    chosen = degrees >= 1
    at_boundary = boundary_gap <= boundary_tolerance(arrays["r1"])

    if "deg_agents" in payload:
        stored = np.asarray(payload["deg_agents"]).astype(np.int64)
        # a single edge may flip at a precision boundary; anything else counts
        off_by_one = np.abs(stored - degrees) <= 1
        checks["check_deg_match"] = bool(np.all(
            (stored == degrees) | (off_by_one & at_boundary)))

    if "pointer_actions" in payload:
        pointer_radii = compute_cutoff_radii(
            payload["agent_states"], payload["pointer_actions"])
        difference = np.abs(pointer_radii - arrays["r1"])[chosen]
        allowed = boundary_tolerance(pointer_radii)[chosen]
        zeroed = bool(np.all(pointer_radii[~chosen] == 0.0))
        checks["check_r1_ptr_bitwise"] = bool(
            bool(np.all(difference <= allowed)) and zeroed)

    if rank_faithful:
        # Only an agent whose own selection is not the exact nearest set can
        # move a rank count, so all three flags share one excuse.
        breaks_r3 = chosen & (arrays["r3"] != 1.0)
        breaks_r4 = chosen & (arrays["r4"] != 0.0)
        checks["check_r3_one"] = bool(np.all(~breaks_r3 | at_boundary))
        checks["check_r4_zero"] = bool(np.all(~breaks_r4 | at_boundary))
        checks["check_r5_ref"] = bool(
            np.array_equal(arrays["r5_sel_count"], arrays["r5_ref_count"])
            or np.all(~breaks_r4 | at_boundary))

    if "rank_dev" in payload:
        # The wrapper stores one leading NaN, so its index t+1 is action t.
        stored = np.asarray(payload["rank_dev"], dtype=np.float64)[1:]
        derived = np.nanmean(np.nan_to_num(arrays["r4"], nan=0.0), axis=1)
        if stored.shape == derived.shape:
            # A step whose only rank-breaking agents sit at a precision
            # boundary is excused, exactly as for r3/r4/r5; what is left is the
            # float32 rounding of the stored series.
            breaks = chosen & (arrays["r4"] != 0.0)
            excused_step = np.all(~breaks | at_boundary, axis=1)
            difference = np.where(excused_step, 0.0, np.abs(stored - derived))
            checks["check_rankdev_match"] = bool(
                difference.max() <= RANKDEV_TOL)
    return checks


def acs_checks(arrays: Dict[str, np.ndarray], degrees: np.ndarray,
               num_agents: int) -> bool:
    """PLAN 2.8 item 5: a full-coupling bundle has the degenerate values."""
    return bool(
        np.all(degrees == num_agents - 1)
        and np.all(arrays["r4"] == 0.0)
        and np.all(np.nan_to_num(arrays["r6_asym"], nan=0.0) == 0.0)
        and np.all(arrays["r6_nonrecip"] == 0.0)
    )


# ------------------------------------------------------------- discovery
class EpisodeSource:
    """One episode to process, with everything the worker needs."""

    def __init__(self, kind, path, out_path, arm, run_id, policy, seed,
                 u_max, reference=None):
        self.kind = kind
        self.path = Path(path)
        self.out_path = Path(out_path)
        self.arm = arm
        self.run_id = run_id
        self.policy = policy
        self.seed = seed
        self.u_max = float(u_max)
        self.reference = reference or {}

    def as_dict(self):
        return {
            "kind": self.kind, "path": str(self.path),
            "out_path": str(self.out_path), "arm": self.arm,
            "run_id": self.run_id, "policy": self.policy, "seed": self.seed,
            "u_max": self.u_max, "reference": self.reference,
        }


def _read_json(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _episode_meta(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as handle:
        return json.loads(str(handle["meta"]))


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def discover_population(source: Path, out_root: Path, arm: Optional[str],
                        max_turn_rate: Optional[float]) -> List[EpisodeSource]:
    manifest = _read_json(source / "manifest.json")
    configs = manifest["spec"].get("effective_config_by_num_agents", {})
    effort_csv = source / "analysis" / "control_effort" / "episode_control_effort.csv"
    reference_rows: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if effort_csv.is_file():
        for row in _csv_rows(effort_csv):
            key = (row["N"], row["policy"], row["seed"])
            reference_rows[key] = row
    sources = []
    for path in sorted((source / "episodes").rglob("*.npz")):
        meta = _episode_meta(path)
        num_agents = str(meta["num_agents"])
        policy = str(meta["policy"])
        seed = str(meta["seed"])
        cap = max_turn_rate
        if cap is None:
            cap = configs[num_agents]["control"]["max_turn_rate"]
        relative = path.relative_to(source / "episodes")
        # drop the suite directory so derived paths read <ckpt>/N<n>/<policy>/
        parts = relative.parts[1:] if len(relative.parts) > 1 else relative.parts
        out_path = (out_root / "episodes" / Path(*parts)).with_suffix(
            "").with_suffix(".rmetrics.npz")
        sources.append(EpisodeSource(
            "population", path, out_path, arm or policy,
            str(meta.get("run_id", "")), policy, seed, cap,
            {"effort": reference_rows.get((num_agents, policy, seed))}))
    return sources


def discover_c2(source: Path, out_root: Path, arm: Optional[str],
                max_turn_rate: Optional[float]) -> List[EpisodeSource]:
    manifest = _read_json(source / "manifest.json")
    run_id = str(manifest.get("run_id", ""))
    cap = max_turn_rate
    if cap is None:
        cap = manifest["spec"]["effective_configs"][0]["config"]["control"][
            "max_turn_rate"]
    judged: Dict[Tuple[str, str], Dict[str, str]] = {}
    for csv_path in sorted((source / "summaries").glob("*_summary.csv")):
        for row in _csv_rows(csv_path):
            judged[(row.get("label", ""), row["seed"])] = row
    sources = []
    for path in sorted((source / "episodes").rglob("*.npz")):
        meta = _episode_meta(path)
        label = str(meta["policy"])
        seed = str(meta["seed"])
        relative = path.relative_to(source / "episodes")
        parts = relative.parts[1:] if len(relative.parts) > 1 else relative.parts
        out_path = (out_root / "episodes" / Path(*parts)).with_suffix(
            "").with_suffix(".rmetrics.npz")
        sources.append(EpisodeSource(
            "c2", path, out_path, arm or label, run_id, label, seed, cap,
            {"judged": judged.get((label, seed))}))
    return sources


def discover_knnref(source: Path, out_root: Path, arm: Optional[str],
                    max_turn_rate: Optional[float]) -> List[EpisodeSource]:
    condition = source.name
    summary_path = source.parent / f"{condition}_summary.csv"
    judged: Dict[str, Dict[str, str]] = {}
    if summary_path.is_file():
        for row in _csv_rows(summary_path):
            judged[row["seed"]] = row
    cap = max_turn_rate if max_turn_rate is not None else default_max_turn_rate()
    sources = []
    for path in sorted(source.glob("*.npz")):
        meta = _episode_meta(path)
        seed = str(meta["seed"])
        out_path = out_root / f"{path.stem}.rmetrics.npz"
        sources.append(EpisodeSource(
            "knnref", path, out_path, arm or condition, "", condition, seed,
            cap, {"judged": judged.get(seed), "fixed_k": meta.get("k")}))
    return sources


def default_out_root(kind: str, source: Path) -> Path:
    if kind == "knnref":
        return source.parent / f"{source.name}_rmetrics"
    return source / DERIVED_DIRNAME


DISCOVERY = {
    "population": discover_population,
    "c2": discover_c2,
    "knnref": discover_knnref,
}


# --------------------------------------------------------------- one episode
def post_c2_min_pair(min_pair: np.ndarray, t_fire: int, success: int) -> float:
    """Minimum pairwise distance from the C2 declaration onward.

    ``min_pair`` has one entry per state (T+1); ``t_fire`` is the state index
    at which the judge first declared C2.  NaN when the episode never reached
    C2, so this column is success-gated by construction (the full-horizon
    ``min_pair_min`` is not).
    """
    if int(success) != 1 or int(t_fire) < 0:
        return float("nan")
    tail = np.asarray(min_pair, dtype=np.float64)[int(t_fire):]
    return float(np.nanmin(tail)) if tail.size else float("nan")


def process_episode(spec: Dict[str, object]) -> Dict[str, object]:
    path = Path(spec["path"])
    kind = str(spec["kind"])
    reference = dict(spec.get("reference") or {})
    with np.load(path, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}
    meta = json.loads(str(payload["meta"]))

    states = np.asarray(payload["agent_states"], dtype=np.float64)
    masks = np.asarray(payload["binary_actions"], dtype=bool)
    controls = np.asarray(payload["control_inputs"], dtype=np.float64)
    horizon, num_agents = int(masks.shape[0]), int(masks.shape[1])
    r0 = float(meta["r0"])
    dt = float(meta["dt"])
    speed = float(meta["speed"])
    u_max = float(spec["u_max"])

    arrays, degrees, boundary_gap = selection_arrays(states, masks, r0)

    judgment = judge_episode_c2(
        payload["phi"], payload["s_ent"], payload["n_comp_r0"],
        payload["reward"])
    _check_judgment(kind, judgment, payload, reference)

    row: Dict[str, object] = {
        "kind": kind, "arm": spec["arm"], "run_id": spec["run_id"],
        "num_agents": num_agents, "policy": spec["policy"],
        "seed": spec["seed"], "source_path": str(path),
        "t_fire": int(judgment["t_fire"]), "success": int(judgment["success"]),
        "J": judgment["J"],
        "min_pair_min": float(np.nanmin(payload["min_pair"])),
        "min_pair_min_post_c2": post_c2_min_pair(
            payload["min_pair"], int(judgment["t_fire"]),
            int(judgment["success"])),
        "r0": r0, "dt": dt, "speed": speed, "u_max_used": u_max,
    }
    row.update(window_summary(arrays, degrees, num_agents))
    row.update(effort_columns(controls, int(judgment["t_fire"]), dt, speed,
                              u_max))

    fixed_k = reference.get("fixed_k")
    rank_faithful = _selection_is_rank_faithful(
        str(spec["policy"]), "pointer_actions" in payload,
        int(fixed_k) if fixed_k is not None else None)
    checks = self_checks(arrays, degrees, payload, rank_faithful, num_agents,
                         boundary_gap)
    full_coupling = (str(spec["policy"]) == "pure_acs"
                     or (fixed_k is not None and int(fixed_k) == num_agents - 1))
    if full_coupling and not acs_checks(arrays, degrees, num_agents):
        raise ValueError(
            "full-coupling arm violates the degenerate ACS-FC values: " + str(path))
    checks["check_effort_matches_control_effort_csv"] = _check_effort_reference(
        row, reference.get("effort"))
    row.update(checks)

    out_path = Path(spec["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    derived_meta = {
        "schema": RMETRICS_SCHEMA, "source_path": str(path), "kind": kind,
        "arm": spec["arm"], "policy": spec["policy"], "seed": spec["seed"],
        "num_agents": num_agents, "horizon": horizon,
        "r0": r0, "dt": dt, "speed": speed, "u_max_used": u_max,
        "windows": {name: [int(w.start), int(w.stop)]
                    for name, w in window_slices(horizon).items()},
        "k0_convention": "nan",
    }
    np.savez_compressed(
        out_path, meta=json.dumps(derived_meta, sort_keys=True),
        **{key: arrays[key] for key in ARRAY_KEYS})
    return row


def _check_judgment(kind: str, judgment, payload, reference) -> None:
    """The offline re-judgment must reproduce whatever the lane stored."""
    if kind == "population":
        stored = {
            "t_fire": int(np.asarray(payload["t_fire"]).item()),
            "success": int(np.asarray(payload["success"]).item()),
            "J": float(np.asarray(payload["J"]).item()),
        }
    else:
        row = reference.get("judged")
        if row is None:
            return
        stored = {
            "t_fire": int(row["t_fire"]), "success": int(row["success"]),
            "J": float(row["J"]) if row["J"] != "" else float("nan"),
        }
    same_j = (
        (math.isnan(stored["J"]) and math.isnan(float(judgment["J"])))
        or stored["J"] == float(judgment["J"])
    )
    if (stored["t_fire"] != int(judgment["t_fire"])
            or stored["success"] != int(judgment["success"]) or not same_j):
        raise ValueError(
            "re-judgment differs from the stored verdict: {} vs {}".format(
                stored, {k: judgment[k] for k in ("t_fire", "success", "J")}))


def _check_effort_reference(row, reference) -> object:
    if not reference:
        return ""
    for column in ("l1_integral_to_c2", "l2_energy_to_c2", "full_mean_abs",
                   "full_mean_square"):
        if column not in reference:
            return ""
        stored = float(reference[column]) if reference[column] != "" else float("nan")
        derived = float(row[column])
        if math.isnan(stored) and math.isnan(derived):
            continue
        if stored != derived:
            return False
    return True


# ----------------------------------------------------------------- compute
def write_summary(rows: List[Dict[str, object]], path: Path) -> Path:
    columns = ["kind", "arm", "run_id", "num_agents", "policy", "seed",
               "source_path", "t_fire", "success", "J", "min_pair_min",
               "min_pair_min_post_c2", "r0", "dt", "speed", "u_max_used"]
    for window in WINDOW_NAMES:
        columns.extend(f"{name}_{window}" for name in R_COLUMNS)
    columns.extend(EFFORT_COLUMNS)
    columns.extend(["rms_to_c2_or_horizon", "sat_frac_to_c2", "sat_frac_full"])
    columns.extend(CHECK_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (
                str(item["policy"]), int(item["num_agents"]),
                str(item["seed"]))):
            writer.writerow(row)
    return path


def run_compute(kind: str, source: Path, out: Optional[Path],
                arm: Optional[str], workers: int,
                max_turn_rate: Optional[float]) -> Dict[str, object]:
    source = Path(source).expanduser().resolve()
    out_root = (Path(out).expanduser().resolve() if out is not None
                else default_out_root(kind, source))
    sources = DISCOVERY[kind](source, out_root, arm, max_turn_rate)
    if not sources:
        raise ValueError(f"no episodes found under {source}")
    specs = [item.as_dict() for item in sources]
    if int(workers) > 1 and len(specs) > 1:
        import multiprocessing as mp

        context = mp.get_context("spawn")
        with context.Pool(min(int(workers), len(specs))) as pool:
            rows = pool.map(process_episode, specs)
    else:
        rows = [process_episode(spec) for spec in specs]
    summary_path = write_summary(rows, out_root / SUMMARY_NAME)
    failures = {
        name: sum(1 for row in rows if row[name] is False)
        for name in CHECK_COLUMNS
    }
    return {
        "kind": kind, "source": str(source), "out": str(out_root),
        "episodes": len(rows), "summary": str(summary_path),
        "check_failures": {k: v for k, v in failures.items() if v},
    }


# ------------------------------------------------------------------- table
def _median_iqr(values: Sequence[float]) -> Tuple[float, float, float, int]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                        dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    q1, q3 = np.percentile(finite, [25, 75])
    return float(np.median(finite)), float(q1), float(q3), int(finite.size)


# Identifiers and physical constants are not metrics; ``success`` is a rate,
# not a quantity with a meaningful median.
IDENTITY_COLUMNS = ("kind", "arm", "run_id", "policy", "seed", "source_path",
                    "num_agents", "success")
CONSTANT_COLUMNS = ("r0", "dt", "speed", "u_max_used")
# Columns that only mean something when the episode reached C2.  The judge
# stores ``t_fire = -1`` for failures and every collection lane filters on
# ``success == 1`` before summarising it; the table path must do the same.
SUCCESS_GATED_COLUMNS = ("t_fire",)


def _table_metrics(rows: List[Dict[str, str]]) -> List[str]:
    skip = set(CHECK_COLUMNS) | set(IDENTITY_COLUMNS) | set(CONSTANT_COLUMNS)
    return [name for name in rows[0] if name not in skip]


def _cell_value(row: Dict[str, str], column: str) -> float:
    if column in SUCCESS_GATED_COLUMNS and str(row.get("success", "")) != "1":
        return float("nan")
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return float("nan")


def _arm_cells(rows: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    return sorted({(str(row.get("policy", "")), str(row.get("num_agents", "")))
                   for row in rows})


def _require_single_cell(name: str, rows: Sequence[Dict[str, str]]) -> None:
    """One arm must be one (policy, N) cell; pooling cells hides structure.

    A population sweep mixes rank-faithful arms (R4 == 0 by definition) with
    learned ones, so a pooled median collapses toward the smallest positive
    value.  Slice the summary CSV per (policy, num_agents) before passing it.
    """
    cells = _arm_cells(rows)
    if len(cells) > 1:
        raise ValueError(
            f"arm {name!r} spans {len(cells)} (policy, num_agents) cells "
            f"{cells}; slice the summary CSV to one cell per arm")


def run_table(arms: Dict[str, Path], pairs: Sequence[Tuple[str, str]],
              out: Path) -> Dict[str, object]:
    loaded = {name: _csv_rows(path) for name, path in arms.items()}
    for name, rows in loaded.items():
        if not rows:
            raise ValueError(f"arm {name} has no rows")
    for name, rows in loaded.items():
        _require_single_cell(name, rows)
    metrics = _table_metrics(next(iter(loaded.values())))

    def numeric(rows, column):
        return [_cell_value(row, column) for row in rows]

    def pair_key(row):
        return (str(row.get("num_agents", "")), str(row["seed"]))

    records = []
    for arm, rows in loaded.items():
        for metric in metrics:
            median, q1, q3, count = _median_iqr(numeric(rows, metric))
            records.append({
                "comparison": "arm", "arm": arm, "other": "", "metric": metric,
                "median": median, "q1": q1, "q3": q3, "iqr": q3 - q1,
                "n": count,
            })
    matched = {}
    for left, right in pairs:
        if left not in loaded or right not in loaded:
            raise ValueError(f"unknown arm in pair {left}:{right}")
        left_by_key = {pair_key(row): row for row in loaded[left]}
        right_by_key = {pair_key(row): row for row in loaded[right]}
        if len(left_by_key) != len(loaded[left]) or (
                len(right_by_key) != len(loaded[right])):
            raise ValueError(
                f"pair {left}:{right}: duplicate (num_agents, seed) rows")
        shared = sorted(set(left_by_key) & set(right_by_key))
        if not shared:
            raise ValueError(
                f"pair {left}:{right}: no shared (num_agents, seed) keys "
                f"({len(left_by_key)} vs {len(right_by_key)} rows)")
        matched[f"{left}:{right}"] = len(shared)
        for metric in metrics:
            deltas = [_cell_value(left_by_key[key], metric)
                      - _cell_value(right_by_key[key], metric)
                      for key in shared]
            median, q1, q3, count = _median_iqr(deltas)
            records.append({
                "comparison": "paired_difference", "arm": left, "other": right,
                "metric": metric, "median": median, "q1": q1, "q3": q3,
                "iqr": q3 - q1, "n": count,
            })
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "comparison", "arm", "other", "metric", "median", "q1", "q3",
            "iqr", "n"])
        writer.writeheader()
        writer.writerows(records)
    return {"out": str(out), "rows": len(records), "arms": sorted(loaded),
            "cells": {name: _arm_cells(rows)[0] for name, rows in loaded.items()},
            "matched_pairs": matched}


# ----------------------------------------------------------------- figures
def _numeric_column(rows: Sequence[Dict[str, str]], column: str) -> np.ndarray:
    """One summary column as floats; a missing or unparsable cell becomes NaN."""
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (TypeError, ValueError, KeyError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _effort_points(rows: Sequence[Dict[str, str]]) -> Tuple[
        np.ndarray, np.ndarray, int, int]:
    """Plottable (t_fire, effort) pairs for one arm, with drawn/total counts.

    ``t_fire < 0`` marks an episode that never reached C2; its effort-to-C2
    columns are the string ``nan`` by construction, so the point cannot be
    placed at all.  The counts let the caller say so instead of letting
    matplotlib drop the vertex without a visual hint.
    """
    t_fire = _numeric_column(rows, "t_fire")
    energy = _numeric_column(rows, "l2_energy_to_c2")
    shown = np.isfinite(t_fire) & (t_fire >= 0) & np.isfinite(energy)
    return t_fire[shown], energy[shown], int(shown.sum()), len(rows)


def _rank_curves(summary_path: Path, window: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pool the derived episodes listed in one arm CSV into P_hat and P_ref.

    Only the ``.rmetrics.npz`` files whose ``source_path`` appears in the CSV
    are used, so a (policy, N) slice of a summary selects exactly its own
    episodes.  The slice must live inside the derived tree (the search root is
    the CSV's directory); a slice that matches nothing is an error rather than
    a silently missing curve, and mixed N cannot be pooled.
    """
    summary_path = Path(summary_path)
    wanted = {row["source_path"] for row in _csv_rows(summary_path)}
    root = summary_path.parent
    selected, reference, agents, count = None, None, None, 0
    for path in sorted(root.rglob("*.rmetrics.npz")):
        with np.load(path, allow_pickle=True) as handle:
            meta = json.loads(str(handle["meta"]))
            if meta.get("source_path") not in wanted:
                continue
            sel = np.asarray(handle["r5_sel_count"], dtype=np.float64)
            ref = np.asarray(handle["r5_ref_count"], dtype=np.float64)
        num_agents = int(meta["num_agents"])
        if agents is None:
            agents = num_agents
        elif agents != num_agents:
            raise ValueError(
                f"{summary_path}: episodes mix num_agents {agents} and "
                f"{num_agents}; rank curves need one N per arm")
        window_slice = window_slices(int(meta["horizon"]))[window]
        sel = sel[window_slice].sum(0) / (
            (window_slice.stop - window_slice.start) * num_agents)
        ref = ref[window_slice].sum(0) / (
            (window_slice.stop - window_slice.start) * num_agents)
        selected = sel if selected is None else selected + sel
        reference = ref if reference is None else reference + ref
        count += 1
    if selected is None:
        raise ValueError(
            f"{summary_path}: none of its {len(wanted)} episodes has a derived "
            f".rmetrics.npz under {root}; write slices inside the derived tree")
    return selected / count, reference / count


def run_figures(arms: Dict[str, Path], out: Path) -> Dict[str, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    loaded = {name: _csv_rows(path) for name, path in arms.items()}
    written = []

    # (1) rank-selection curves against the same-k nearest projection
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, window in zip(axes, ("full", "ss")):
        for name, path in arms.items():
            selected, reference = _rank_curves(Path(path), window)
            ranks = np.arange(1, selected.size + 1)
            style = arm_style(name)
            axis.plot(ranks, selected, marker="o", markersize=3,
                      color=style["color"], linestyle=style["linestyle"],
                      label=f"{name}{style['suffix']}")
            # Dashed is reserved for stochastic decoding, so the same-k
            # reference is dotted.
            axis.plot(ranks, reference, linestyle=":", linewidth=1,
                      color=style["color"], label=f"{name} (same-k nearest)")
        axis.set_title(f"R5 selection probability by distance rank ({window})")
        axis.set_xlabel("distance rank m")
        axis.set_ylabel("P(select)")
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = out / "r5_rank_selection.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # (2) R1 and R2 distributions, early vs steady state
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (metric, label) in zip(
            axes, (("r1", "R1 effective radius (m)"), ("k", "R2 effective k"))):
        data, labels = [], []
        for name, rows in loaded.items():
            for window in ("early", "ss"):
                values = _numeric_column(rows, f"{metric}_{window}")
                data.append(values[np.isfinite(values)])
                labels.append(f"{name}\n{window}")
        axis.boxplot(data, tick_labels=labels)
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    path = out / "r1_r2_early_vs_ss.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # (3) effort against time to C2
    figure, axis = plt.subplots(figsize=(6, 4.5))
    omitted = []
    for name, rows in loaded.items():
        t_fire, energy, drawn, total = _effort_points(rows)
        axis.scatter(t_fire, energy, s=12, alpha=0.7,
                     color=arm_style(name)["color"],
                     label=f"{name} (n={drawn}/{total})")
        if drawn != total:
            omitted.append(f"{name} {total - drawn}")
    axis.set_xlabel("t_fire (state index)")
    axis.set_ylabel("l2_energy_to_c2 (rad^2/s)")
    axis.set_title("effort to C2 vs time to C2" + (
        "\nnot plotted, never reached C2: " + ", ".join(omitted)
        if omitted else "\nall episodes reached C2"))
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out / "effort_vs_t_fire.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    # (4) how often an agent selects nobody
    figure, axis = plt.subplots(figsize=(6, 4.5))
    names = list(loaded)
    width = 0.8 / max(1, len(WINDOW_NAMES))
    for index, window in enumerate(WINDOW_NAMES):
        values = [_nanmean(_numeric_column(loaded[name], f"k0_rate_{window}"))
                  for name in names]
        bars = axis.bar(np.arange(len(names)) + index * width, values,
                        width=width, label=window)
        # A true zero draws no rectangle and reads as missing data.  Put a
        # visible rule on the baseline where the bar has zero height, and
        # label every bar so "0" is stated, not inferred from an empty slot.
        for rect, value in zip(bars, values):
            if np.isfinite(value) and value == 0.0:
                axis.hlines(0.0, rect.get_x(), rect.get_x() + rect.get_width(),
                            color=rect.get_facecolor(), linewidth=2.5,
                            zorder=3)
        axis.bar_label(
            bars, fontsize=6, padding=2, rotation=90,
            labels=["n/a" if not np.isfinite(v)
                    else ("0" if v == 0.0 else f"{v:.2g}") for v in values])
    axis.margins(y=0.22)              # headroom for the rotated labels
    axis.set_ylim(bottom=0.0)
    axis.set_xticks(np.arange(len(names)) + width)
    axis.set_xticklabels(names, fontsize=8)
    for label, name in zip(axis.get_xticklabels(), names):
        label.set_color(arm_style(name)["color"])
    axis.set_ylabel("k = 0 rate (mean over episodes)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out / "k0_rate.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    return {"out": str(out), "figures": [str(item) for item in written]}


# --------------------------------------------------------------------- cli
def _arm_argument(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("arm name must not be empty")
    return name, Path(path)


def _pairs_argument(value: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("expected A:B[,C:D]")
        left, right = item.split(":", 1)
        pairs.append((left, right))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.rmetrics",
        description="post-hoc R1-R6 and control-effort metrics")
    commands = parser.add_subparsers(dest="command", required=True)

    compute = commands.add_parser(
        "compute", help="derive per-episode metrics from stored raw arrays")
    compute.add_argument("--kind", required=True, choices=KINDS)
    compute.add_argument("--source", required=True, type=Path)
    compute.add_argument("--out", type=Path)
    compute.add_argument("--arm")
    compute.add_argument("--workers", type=int, default=1)
    compute.add_argument("--max-turn-rate", type=float)

    table = commands.add_parser(
        "table", help="median/IQR per arm and paired seed differences")
    table.add_argument("--arm", action="append", required=True,
                       type=_arm_argument, metavar="NAME=SUMMARY_CSV")
    table.add_argument("--pairs", type=_pairs_argument, default=[])
    table.add_argument("--out", required=True, type=Path)

    figures = commands.add_parser("figures", help="the four comparison figures")
    figures.add_argument("--arm", action="append", required=True,
                         type=_arm_argument, metavar="NAME=SUMMARY_CSV")
    figures.add_argument("--out", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compute":
        result = run_compute(args.kind, args.source, args.out, args.arm,
                             args.workers, args.max_turn_rate)
    elif args.command == "table":
        result = run_table(dict(args.arm), args.pairs, args.out)
    else:
        result = run_figures(dict(args.arm), args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
