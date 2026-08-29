"""Nearest-projection ablation runner (study acs-confirm, Phase 5).

Wraps pi_E's per-step action: each agent's degree deg_i(t) is KEPT, but its
selection is replaced by its deg_i nearest active (mask-allowed) neighbors —
rank-deviation is forced to ~0 while the degree schedule is untouched.
Registered reading (PROBLEM.md): fail>=2% -> non-nearest selection is causal
for the insurance; fail<=1% -> distillable to a k(t) schedule.

Usage:
  python eval_project_nearest.py --ckpt <C1 ckpt> --label ablE_L250 \
      --seeds 1500-1999 --bound 250 --workers 20
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_ROOT = os.environ.get("FLOCK_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STUDY = os.path.join(_ROOT, "studies", "acs-confirm")
PRED = os.path.join(_ROOT, "studies", "acs-conv-knn")
sys.path.insert(0, os.path.join(PRED, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from eval_c2_r3 import C2Policy, ForensicsWrapper, judge_npz  # noqa: E402


class NearestProjector:
    """Keep per-agent degree deg_i(t); project selection onto the deg_i
    nearest active, mask-allowed neighbors (self-loop diagonal preserved)."""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, obs):
        a = self.policy(obs)
        pm = obs["padding_mask"].astype(bool)
        act = np.where(pm)[0]
        rel = obs["local_agent_infos"][np.ix_(act, act)][:, :, :2]
        d2 = (rel ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nm = obs["neighbor_masks"].astype(bool)[np.ix_(act, act)]
        d2[~nm] = np.inf  # forbidden edges are never "nearest"
        sel = a[np.ix_(act, act)].astype(bool) & nm
        np.fill_diagonal(sel, False)
        order = np.argsort(d2, axis=1)
        out = np.zeros_like(a)
        out[np.diag_indices_from(out)] = 1  # env contract: diagonal always 1
        for li in range(len(act)):
            deg = min(int(sel[li].sum()), int(np.isfinite(d2[li]).sum()))
            if deg:
                out[act[li], act[order[li, :deg]]] = 1
        return out


def run_one(args):
    ckpt, label, seed, steps, bound, n_agents, outdir = args
    out = os.path.join(outdir, f"{label}_s{seed}.npz")
    if os.path.exists(out):
        return out
    import torch
    torch.set_num_threads(1)
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input
    from common import build_config, rollout, save_run

    cfg = build_config(n_agents=n_agents, max_steps=steps, initial_position_bound=bound)
    cfg.env.expose_aux_target = True
    cfg.env.expose_global_stats = True
    with open(os.path.join(os.path.dirname(ckpt), "params.json")) as f:
        _params = json.load(f)
    cfg.env.obs_position_scale = (_params.get("env_config", {}).get("config", {})
                                  .get("env", {}).get("obs_position_scale", "legacy"))
    assert cfg.env.initial_position_bound_pool is None
    tmp_env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=0))
    policy = ForensicsWrapper(NearestProjector(C2Policy(ckpt, tmp_env)))
    rec, snaps, ts, meta = rollout(policy, cfg, seed, pos_stride=10,
                                   extra_meta=dict(policy=label, ckpt=ckpt,
                                                   ablation="nearest_projection"))
    rec["rank_dev"] = np.concatenate([[np.nan], np.array(policy.rank_dev, dtype=np.float32)])
    deg = np.stack(policy.deg_agents)
    os.makedirs(outdir, exist_ok=True)
    save_run(out, rec, snaps, ts, meta)
    with np.load(out, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    data["deg_agents"] = deg
    np.savez_compressed(out, **data)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", default="1500-1999")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bound", type=float, default=250.0)
    ap.add_argument("--n-agents", type=int, default=20)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    outdir = os.path.join(STUDY, "data", "eval", args.label)
    jobs = [(args.ckpt, args.label, s, args.steps, args.bound, args.n_agents, outdir)
            for s in seeds]
    with Pool(args.workers) as pool:
        paths = pool.map(run_one, jobs)

    import pandas as pd
    df = pd.DataFrame([judge_npz(p) for p in sorted(paths)]).sort_values("seed")
    csv_path = os.path.join(STUDY, "data", "eval", f"{args.label}_summary.csv")
    df.to_csv(csv_path, index=False)
    det = df[df.success == 1]
    print(f"\n=== {args.label}: {len(seeds)} seeds, bound={args.bound} ===")
    print(f"success {int(df.success.sum())}/{len(df)}  t_conv med {det.t_fire.median():.0f}  "
          f"J med {det.J.median():.1f}")
    print(f"rank_dev early med {df.rank_dev_early.median():.3f}  "
          f"ss med {df.rank_dev_ss.median():.3f}  (should be ~0 post-projection)")
    print(f"summary -> {csv_path}")


if __name__ == "__main__":
    main()
