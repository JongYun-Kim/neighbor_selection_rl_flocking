"""C2-protocol checkpoint evaluation — the criterion of record.

Rolls out a trained checkpoint on the settled protocol: paired seeds, N=20,
fixed horizon, deterministic (argmax) actions, CPU. Logs the predecessor-
standard series via eval.common.rollout PLUS adaptivity forensics (per-step
rank-deviation and per-agent degree), judges C2 offline, and reports
success / t_conv / J against the k-NN frontier references.

Usage (always from the repo root, so that `eval` resolves to this package):
  python -m eval.eval_c2 --ckpt <path/to/checkpoint_0000NN> --label lp848 \
      [--seeds 1000-1031] [--steps 6000] [--bound 250] [--workers 8] \
      [--outdir <dir>]
  python -m eval.eval_c2 --rank-runs <run_dir>   # rank checkpoints by eval metrics

Outputs: <outdir>/<label>/<label>_s<seed>.npz + <outdir>/<label>_summary.csv
(default outdir: test_results/eval, gitignored). Per-seed npz are cached: an
existing file is reused, so a partial re-run only fills the gaps.

Promoted from studies/acs-confirm/src/eval_c2_r3.py, which stays in place as
the record of that study. The rollout/judge logic is unchanged; the diff is
path resolution, imports, and the --outdir flag that replaces the study's
hardcoded data directory.

Checkpoint -> policy loading lives in eval.policies; this module keeps the
forensics wrapper, the C2 judge, the rollout driver and the CLI.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only inference
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from eval.policies import (induced_mask_from_obs, is_dknn_params,
                           load_policy)
from utils.paths import repo_path

DEFAULT_OUTDIR = repo_path("test_results", "eval")

PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05


# ---------------------------------------------------------------- forensics
class ForensicsWrapper:
    """Wraps a policy; records per-step rank-deviation + per-agent degree.

    rank_dev[t] = mean over active agents of the fraction of selected off-diag
    edges NOT inside that agent's nearest-deg_i distance set (0 == exact k-NN
    mimicry with per-agent k=deg_i). For pointer policies the selection mask is
    reconstructed from the observation (induced_mask_from_obs); the env applies
    its own conversion to the physics.
    """

    def __init__(self, policy):
        self.policy = policy
        self.rank_dev = []
        self.deg_agents = []

    def __call__(self, obs):
        a = self.policy(obs)
        pm = obs["padding_mask"].astype(bool)
        act = np.where(pm)[0]
        rel = obs["local_agent_infos"][np.ix_(act, act)][:, :, :2]
        d2 = (rel ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        sel_full = a if a.ndim == 2 else induced_mask_from_obs(obs, a)
        sel = sel_full[np.ix_(act, act)].astype(bool)
        np.fill_diagonal(sel, False)
        n = len(act)
        devs, degs = [], []
        order = np.argsort(d2, axis=1)
        for i in range(n):
            deg = int(sel[i].sum())
            degs.append(deg)
            if deg == 0:
                devs.append(0.0)
                continue
            nearest = set(order[i, :deg].tolist())
            outside = sum(1 for j in np.where(sel[i])[0] if j not in nearest)
            devs.append(outside / deg)
        self.rank_dev.append(float(np.mean(devs)))
        self.deg_agents.append(np.array(degs, dtype=np.int16))
        return a


# ---------------------------------------------------------------- C2 judge
def t_fire_c2(phi, s, comp):
    import pandas as pd
    pphi, ps, pcomp = pd.Series(phi), pd.Series(s), pd.Series(comp)
    align = (pphi.rolling(W_A).min() > PHI_GOAL).values
    coh = (pcomp.rolling(W).max() == 1).values
    band = ((ps.rolling(W).max() - ps.rolling(W).min()) / ps.rolling(W).mean()).values
    with np.errstate(invalid="ignore"):
        ok = align & coh & (band < EPS)
    hit = np.flatnonzero(ok)
    return int(hit[0]) if hit.size else -1


def judge_npz(path):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    t = t_fire_c2(z["phi"], z["s_ent"], z["n_comp_r0"])
    r = z["reward"]
    J = float(-np.nansum(r[1:t + 1])) if t >= 0 else np.nan
    rd = z["rank_dev"] if "rank_dev" in z.files else None
    out = dict(seed=m["seed"], t_fire=t, success=int(t >= 0), J=J,
               phi_ss=float(np.nanmedian(z["phi"][-300:])),
               sigma_p_ss=float(np.nanmedian(z["s_ent"][-300:])),
               min_pair=float(np.nanmin(z["min_pair"])),
               deg_ss=float(np.nanmedian(z["deg_mean"][-300:])),
               churn_ss=float(np.nanmedian(z["churn"][-300:])),
               n_comp_end=float(z["n_comp_r0"][-1]))
    if rd is not None:
        out["rank_dev_early"] = float(np.nanmean(rd[:300]))
        out["rank_dev_ss"] = float(np.nanmean(rd[-300:]))
        out["deg_early"] = float(np.nanmean(z["deg_mean"][:300]))
    return out


# ---------------------------------------------------------------- rollout job
def run_one(args):
    ckpt, label, seed, steps, bound, n_agents, outdir = args
    out = os.path.join(outdir, f"{label}_s{seed}.npz")
    if os.path.exists(out):
        return out
    import torch
    torch.set_num_threads(1)
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input
    from eval.common import build_config, rollout, save_run

    cfg = build_config(n_agents=n_agents, max_steps=steps, initial_position_bound=bound)
    # Method dispatch (B6): the checkpoint's params.json names the model and
    # action encoding; the eval env mirrors the training obs/action wiring.
    with open(os.path.join(os.path.dirname(ckpt), "params.json")) as f:
        _params = json.load(f)
    _penv = _params.get("env_config", {}).get("config", {}).get("env", {})
    _is_dknn = is_dknn_params(_params)
    if _is_dknn:
        cfg.env.action_type = "dynamic_k_nn"
        # env-side realized-mask reporting: selection-graph series and any
        # downstream forensics use the env's own pointer->mask conversion
        cfg.env.evaluation_diagnostics = True
        cfg.env.expose_aux_target = False
        cfg.env.expose_global_stats = False
    else:
        cfg.env.expose_aux_target = True
        cfg.env.expose_global_stats = True
    # Scale-robustness study: the eval env must observe in the SAME scale
    # system the checkpoint was trained with. Read it from the run's
    # params.json (old checkpoints lack the field -> legacy). The L pool is
    # never set in eval builds — L comes fixed from --bound.
    cfg.env.obs_position_scale = _penv.get("obs_position_scale", "legacy")
    assert cfg.env.initial_position_bound_pool is None
    tmp_env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=0))
    policy = ForensicsWrapper(load_policy(ckpt, tmp_env))
    rec, snaps, ts, meta = rollout(policy, cfg, seed, pos_stride=10,
                                   extra_meta=dict(policy=label, ckpt=ckpt))
    rec["rank_dev"] = np.concatenate([[np.nan], np.array(policy.rank_dev, dtype=np.float32)])
    deg = np.stack(policy.deg_agents)  # (T, n)
    os.makedirs(outdir, exist_ok=True)
    save_run(out, rec, snaps, ts, meta)
    # append per-agent degrees (separate arrays to keep save_run untouched)
    with np.load(out, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    data["deg_agents"] = deg
    np.savez_compressed(out, **data)
    return out


# ---------------------------------------------------------------- ckpt ranking
def rank_runs(run_dir):
    import csv as csvmod
    for prog in glob.glob(os.path.join(run_dir, "*", "progress.csv")):
        with open(prog, errors="ignore") as fh:
            rows = list(csvmod.reader(fh))
        hdr = rows[0]
        idx = {k: i for i, k in enumerate(hdr)}
        keys = ["training_iteration",
                "evaluation/custom_metrics/c2_success_mean",
                "evaluation/custom_metrics/J_success_mean",
                "evaluation/custom_metrics/t_conv_mean",
                "custom_metrics/c2_success_mean"]
        print(f"--- {prog}")
        print("iter  ev_succ  ev_J   ev_tconv  train_succ  ckpt?")
        ckpts = {int(os.path.basename(d).split("_")[-1])
                 for d in glob.glob(os.path.join(os.path.dirname(prog), "checkpoint_*"))}
        for r in rows[1:]:
            def g(k):
                try:
                    return float(r[idx[k]])
                except Exception:
                    return np.nan
            it = int(g("training_iteration"))
            ev = g(keys[1])
            if not np.isnan(ev) or it in ckpts:
                print(f"{it:4d}  {ev:7.3f}  {g(keys[2]):6.0f} {g(keys[3]):8.0f}  "
                      f"{g(keys[4]):9.3f}  {'*' if it in ckpts else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--label")
    ap.add_argument("--seeds", default="1000-1031")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bound", type=float, default=250.0)
    ap.add_argument("--n-agents", type=int, default=20,
                    help="N for the eval env (N-axis probes; model is N-agnostic)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="base directory for npz + summary CSV "
                         "(default: test_results/eval)")
    ap.add_argument("--rank-runs", dest="rank_runs_dir")
    args = ap.parse_args()

    if args.rank_runs_dir:
        rank_runs(args.rank_runs_dir)
        return
    assert args.ckpt and args.label, "--ckpt and --label required"

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    outdir = os.path.join(args.outdir, args.label)
    jobs = [(args.ckpt, args.label, s, args.steps, args.bound, args.n_agents, outdir)
            for s in seeds]
    with Pool(args.workers) as pool:
        paths = pool.map(run_one, jobs)

    rows = [judge_npz(p) for p in sorted(paths)]
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("seed")
    csv_path = os.path.join(args.outdir, f"{args.label}_summary.csv")
    df.to_csv(csv_path, index=False)
    det = df[df.success == 1]
    print(f"\n=== {args.label}: {len(seeds)} seeds, bound={args.bound}, steps={args.steps} ===")
    print(f"success {int(df.success.sum())}/{len(df)}  "
          f"t_conv med {det.t_fire.median():.0f}  J med {det.J.median():.1f}  "
          f"J mean {det.J.mean():.1f}")
    print(f"phi_ss med {det.phi_ss.median():.4f}  sigma_p_ss med {det.sigma_p_ss.median():.1f}  "
          f"min_pair min {df.min_pair.min():.1f}  deg_ss med {df.deg_ss.median():.2f}  "
          f"churn_ss med {df.churn_ss.median():.4f}")
    if "rank_dev_early" in df:
        print(f"rank_dev early med {df.rank_dev_early.median():.3f}  ss med {df.rank_dev_ss.median():.3f}")
    print(f"summary -> {csv_path}")


if __name__ == "__main__":
    main()
