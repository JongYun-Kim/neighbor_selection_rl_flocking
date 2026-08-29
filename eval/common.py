"""Shared rollout + metric utilities for the criterion-of-record eval harness.

Read-only w.r.t. repo code: imports envs.env and baselines as-is (no modification).
All physical quantities are logged in raw env units (meters, m/s, radians, steps).

Key repo facts this module relies on (see NOTES_env.md):
- k-NN baseline: create_baseline('nearest', k=...); k EXCLUDES self; needs
  observation_type='ego_centric'; stateless.
- Disc baseline: create_baseline('distance', distance_threshold=...) with the
  threshold in units of initial_position_bound/2.
- cfg.env.use_fixed_episode_length=True disables early termination -> fixed horizon.
- gym 0.23 API: env.seed(s) BEFORE reset(); step -> (obs, reward, done, info).
- Ground truth after each step: env.state['agent_states'] rows = active agents via
  env.state['padding_mask']; columns [0:2]=xy, [2:4]=velocity. We derive everything
  from xy/velocity only (no reliance on the heading-column layout).
- Env metrics: spatial_entropy = sqrt(sum var(xy)), velocity_entropy =
  sqrt(sum var(v)) over active agents; exposed per step in info (task_type='acs').
  We recompute both ourselves and cross-check against info when available.

Promoted from studies/acs-conv-knn/src/common.py, which stays in place as the
record of the ACS k-NN convergence study. The module body is unchanged; only
path resolution and imports differ here (repo root via utils.paths instead of
a sys.path insert).
"""
import json
import os

import numpy as np

from envs.env import NeighborSelectionFlockingEnv, load_config, config_to_env_input
from baselines import create_baseline
from utils.paths import repo_path

DEFAULT_YAML = repo_path("envs", "default_env_config.yaml")


SERIES_NAMES = [
    "s_ent",        # spatial entropy, own computation (m)
    "v_ent",        # velocity entropy, own computation (m/s)
    "s_ent_env",    # env-reported (NaN if unavailable)
    "v_ent_env",    # env-reported (NaN if unavailable)
    "phi",          # heading order parameter |mean(v_i/|v_i|)| in [0,1]
    "reward",       # env reward (is_training=False -> -(dt*speed*|u|+rho*dt) mean)
    "nnd_mean",     # mean over agents of nearest-neighbor distance (m)
    "nnd_max",      # max over agents of nearest-neighbor distance (m)
    "min_pair",     # min pairwise distance (m)  (== min NND)
    "diam",         # max pairwise distance (m)
    "radius",       # max distance from centroid (m)
    "n_comp_sel",   # weakly-connected components of SELECTED graph (action, symmetrized)
    "n_comp_r0",    # connected components of proximity graph (dist < r0)
    "churn",        # 1 - Jaccard(selected off-diag edges at t, t-1); 0 = frozen topology
    "deg_mean",     # mean out-degree of selected graph excluding self-loop
]


def build_config(n_agents=20, max_steps=3000, initial_position_bound=None):
    """Load the repo default config and apply study settings (no repo file edits)."""
    cfg = load_config(DEFAULT_YAML)
    cfg.env.task_type = "acs"
    cfg.env.observation_type = "ego_centric"
    cfg.env.num_agents_pool = [int(n_agents)]
    cfg.env.max_time_steps = int(max_steps)
    cfg.env.use_fixed_episode_length = True   # never early-terminate; judge offline
    cfg.env.is_training = False               # reward = plain control cost, not shaped
    cfg.env.continuous_action = False
    cfg.env.get_state_hist = False
    if initial_position_bound is not None:
        cfg.control.initial_position_bound = float(initial_position_bound)
    return cfg


def _n_components(adj_bool):
    """Number of connected components of an undirected boolean graph (small n)."""
    n = adj_bool.shape[0]
    reach = (adj_bool | adj_bool.T | np.eye(n, dtype=bool)).astype(np.float32)
    # Boolean semiring closure via repeated squaring: ceil(log2(n)) matmuls.
    steps = max(1, int(np.ceil(np.log2(max(n, 2)))))
    for _ in range(steps):
        reach = (reach @ reach) > 0
        reach = reach.astype(np.float32)
    return int(np.unique(reach > 0, axis=0).shape[0])


def _pairwise_dist(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return d


def run_episode(k=None, distance_threshold=None, n_agents=20, max_steps=3000,
                initial_position_bound=None, seed=0, pos_stride=10):
    """Run one fixed-horizon episode with a k-NN ('nearest') or disc ('distance')
    baseline and return {series dict, position snapshots, metadata}.

    Exactly one of k / distance_threshold must be given.
    """
    assert (k is None) != (distance_threshold is None), "give k XOR distance_threshold"
    cfg = build_config(n_agents, max_steps, initial_position_bound)
    if k is not None:
        baseline = create_baseline("nearest", k=int(k))
    else:
        baseline = create_baseline("distance", distance_threshold=float(distance_threshold))
    extra = dict(k=None if k is None else int(k),
                 distance_threshold=distance_threshold)
    return rollout(baseline, cfg, seed, pos_stride, extra_meta=extra)


def rollout(policy, cfg, seed, pos_stride=10, extra_meta=None):
    """Run one fixed-horizon episode with ANY policy callable obs -> (N,N) int
    action (baseline or learned), logging the standard metric series."""
    env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
    env.seed(seed)
    obs = env.reset()

    pm0 = env.state["padding_mask"]
    act_idx = np.where(pm0)[0]
    n = len(act_idx)
    r0 = float(cfg.control.r0)
    T = int(cfg.env.max_time_steps)

    rec = {name: np.full(T + 1, np.nan, dtype=np.float32) for name in SERIES_NAMES}
    pos_snaps, snap_ts = [], []

    def state_metrics():
        st = env.state["agent_states"]
        pos = st[act_idx, 0:2].astype(np.float64)
        vel = st[act_idx, 2:4].astype(np.float64)
        return pos, vel

    def log_step(t, action, info):
        pos, vel = state_metrics()
        rec["s_ent"][t] = np.sqrt(pos.var(axis=0).sum())
        rec["v_ent"][t] = np.sqrt(vel.var(axis=0).sum())
        se = info.get("spatial_entropy") if info is not None else None
        ve = info.get("velocity_entropy") if info is not None else None
        rec["s_ent_env"][t] = np.nan if se is None else se
        rec["v_ent_env"][t] = np.nan if ve is None else ve
        speed_i = np.linalg.norm(vel, axis=1)
        unit = vel / np.maximum(speed_i, 1e-12)[:, None]
        rec["phi"][t] = np.linalg.norm(unit.mean(axis=0))
        d = _pairwise_dist(pos)
        np.fill_diagonal(d, np.inf)
        nnd = d.min(axis=1)
        rec["nnd_mean"][t] = nnd.mean()
        rec["nnd_max"][t] = nnd.max()
        rec["min_pair"][t] = nnd.min()
        dd = d.copy()
        dd[np.isinf(dd)] = 0.0
        rec["diam"][t] = dd.max()
        rec["radius"][t] = np.linalg.norm(pos - pos.mean(axis=0), axis=1).max()
        rec["n_comp_r0"][t] = _n_components(d < r0)
        if action is not None:
            a = action[np.ix_(act_idx, act_idx)].astype(bool)
            off = a & ~np.eye(n, dtype=bool)
            rec["deg_mean"][t] = off.sum() / n
            rec["n_comp_sel"][t] = _n_components(off)
        return pos

    prev_edges = None
    done = False
    t = 0
    # t=0 snapshot (pre-first-step); env never reports initial entropies itself.
    pos0 = log_step(0, None, None)
    pos_snaps.append(pos0.astype(np.float32))
    snap_ts.append(0)

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        t += 1
        # Selection-graph series use the REALIZED (N,N) mask. For binary
        # policies that is the action itself; for non-binary encodings
        # (e.g. dynamic_k_nn pointers) the env reports the converted mask in
        # info["binary_action"] when cfg.env.evaluation_diagnostics is on.
        a_mask = action
        if info is not None and "binary_action" in info:
            a_mask = info["binary_action"]
        pos = log_step(t, a_mask, info)
        rec["reward"][t] = reward
        a = a_mask[np.ix_(act_idx, act_idx)].astype(bool)
        edges = a & ~np.eye(n, dtype=bool)
        if prev_edges is not None:
            inter = (edges & prev_edges).sum()
            union = (edges | prev_edges).sum()
            rec["churn"][t] = 1.0 - (inter / union if union else 1.0)
        prev_edges = edges
        if t % pos_stride == 0 or done:
            pos_snaps.append(pos.astype(np.float32))
            snap_ts.append(t)
        if t >= T:
            break

    # Cross-check own metrics vs env-reported (where env provided them).
    ok = ~np.isnan(rec["s_ent_env"])
    max_diff = float(np.nanmax(np.abs(rec["s_ent"][ok] - rec["s_ent_env"][ok]))) if ok.any() else float("nan")
    okv = ~np.isnan(rec["v_ent_env"])
    max_diff_v = float(np.nanmax(np.abs(rec["v_ent"][okv] - rec["v_ent_env"][okv]))) if okv.any() else float("nan")

    meta = dict(
        n_agents=int(n), max_steps=T, seed=int(seed),
        initial_position_bound=float(cfg.control.initial_position_bound),
        r0=r0, speed=float(cfg.control.speed), dt=float(cfg.env.dt),
        k1=float(cfg.control.k1), k2=float(cfg.control.k2),
        lam=float(cfg.control.lam), sig=float(cfg.control.sig),
        rho=float(cfg.control.rho),
        entropy_v_goal=float(cfg.env.entropy_v_goal),
        entropy_p_goal=(0.7 * r0 if cfg.env.entropy_p_goal is None
                        else float(cfg.env.entropy_p_goal)),
        steps_done=int(t),
        env_metric_max_diff_spatial=max_diff,
        env_metric_max_diff_velocity=max_diff_v,
    )
    meta.update(extra_meta or {})
    meta.setdefault("k", None)
    meta.setdefault("distance_threshold", None)
    return rec, np.stack(pos_snaps), np.asarray(snap_ts, dtype=np.int32), meta


def save_run(path, rec, pos_snaps, snap_ts, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pos_snaps=pos_snaps, snap_ts=snap_ts,
                        meta=json.dumps(meta), **rec)
