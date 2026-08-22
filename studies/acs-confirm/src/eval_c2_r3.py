"""Phase 4 evaluation for study acs-c2-train.

Rolls out a trained checkpoint (variant A or B) on the settled protocol:
paired seeds, N=20, fixed horizon, deterministic (argmax) actions, CPU.
Logs the predecessor-standard series via common.rollout PLUS adaptivity
forensics (per-step rank-deviation and per-agent degree), judges C2 offline,
and reports success / t_conv / J against the k-NN frontier references.

Usage:
  python eval_c2.py --ckpt <path/to/checkpoint_0000NN> --label A_iter50 \
      [--seeds 1000-1031] [--steps 6000] [--bound 250] [--workers 8]
  python eval_c2.py --rank-runs <run_dir>   # rank checkpoints by eval metrics

Outputs: data/eval/<label>/<label>_s<seed>.npz + data/eval/<label>_summary.csv
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only inference
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# Repo root: FLOCK_ROOT when set, else resolved from this file's location
# (studies/acs-confirm/src/... -> 4 levels up); /workspace in the canonical layout.
REPO = os.environ.get("FLOCK_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STUDY = os.path.join(REPO, "studies", "acs-confirm")
PRED = os.path.join(REPO, "studies", "acs-conv-knn")
sys.path.insert(0, os.path.join(PRED, "src"))
sys.path.insert(0, REPO)

PHI_GOAL, W_A, W, EPS = 0.98, 50, 300, 0.05


# ---------------------------------------------------------------- policy load
class C2Policy:
    """Deterministic (argmax) policy from an RLlib checkpoint of this study.

    Standalone version of evaluate_checkpoint.RLPolicy that also feeds the
    "global_stats" obs key consumed by use_global_stats models.
    """

    def __init__(self, checkpoint_path, env):
        import pickle
        import torch
        self.torch = torch
        params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
        with open(params_path) as f:
            params = json.load(f)
        mc = params["model"]["custom_model_config"]
        from models.ppo import NeighborSelectionPPORLlib
        N = env.num_agents_max
        self.model = NeighborSelectionPPORLlib(
            obs_space=env.observation_space, action_space=env.action_space,
            num_outputs=2 * N * N, model_config={"custom_model_config": mc},
            name="eval_policy")
        with open(os.path.join(checkpoint_path, "policies", "default_policy",
                               "policy_state.pkl"), "rb") as f:
            state = pickle.load(f)
        torch_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                       for k, v in state["weights"].items()}
        missing, unexpected = self.model.load_state_dict(torch_state, strict=False)
        if missing or unexpected:
            print(f"WARN load_state_dict: missing={missing} unexpected={unexpected}")
        self.model.eval()
        self.N = N

    def __call__(self, obs):
        torch = self.torch
        with torch.no_grad():
            t = {
                "local_agent_infos": torch.from_numpy(obs["local_agent_infos"][None]).float(),
                "neighbor_masks": torch.from_numpy(obs["neighbor_masks"][None]).float(),
                "padding_mask": torch.from_numpy(obs["padding_mask"][None]).float(),
                "is_from_my_env": torch.from_numpy(np.array([True])),
            }
            for k in ("global_agent_infos", "global_stats"):
                if k in obs:
                    t[k] = torch.from_numpy(obs[k][None]).float()
            logits, _ = self.model.forward({"obs": t}, state=[], seq_lens=None)
            lg = logits.numpy()[0].reshape(self.N, self.N, 2)
            return np.argmax(lg, axis=-1).astype(np.int8)


class ForensicsWrapper:
    """Wraps a policy; records per-step rank-deviation + per-agent degree.

    rank_dev[t] = mean over active agents of the fraction of selected off-diag
    edges NOT inside that agent's nearest-deg_i distance set (0 == exact k-NN
    mimicry with per-agent k=deg_i).
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
        sel = a[np.ix_(act, act)].astype(bool)
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
    from common import build_config, rollout, save_run

    cfg = build_config(n_agents=n_agents, max_steps=steps, initial_position_bound=bound)
    cfg.env.expose_aux_target = True
    cfg.env.expose_global_stats = True
    # Scale-robustness study: the eval env must observe in the SAME scale
    # system the checkpoint was trained with. Read it from the run's
    # params.json (old checkpoints lack the field -> legacy). The L pool is
    # never set in eval builds — L comes fixed from --bound.
    with open(os.path.join(os.path.dirname(ckpt), "params.json")) as f:
        _params = json.load(f)
    cfg.env.obs_position_scale = (_params.get("env_config", {}).get("config", {})
                                  .get("env", {}).get("obs_position_scale", "legacy"))
    assert cfg.env.initial_position_bound_pool is None
    tmp_env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=0))
    policy = ForensicsWrapper(C2Policy(ckpt, tmp_env))
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
    ap.add_argument("--rank-runs", dest="rank_runs_dir")
    args = ap.parse_args()

    if args.rank_runs_dir:
        rank_runs(args.rank_runs_dir)
        return
    assert args.ckpt and args.label, "--ckpt and --label required"

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    outdir = os.path.join(STUDY, "data", "eval", args.label)
    jobs = [(args.ckpt, args.label, s, args.steps, args.bound, args.n_agents, outdir)
            for s in seeds]
    with Pool(args.workers) as pool:
        paths = pool.map(run_one, jobs)

    rows = [judge_npz(p) for p in sorted(paths)]
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("seed")
    csv_path = os.path.join(STUDY, "data", "eval", f"{args.label}_summary.csv")
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
