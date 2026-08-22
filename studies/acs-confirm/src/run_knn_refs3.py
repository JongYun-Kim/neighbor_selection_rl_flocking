"""Fresh k-NN reference rollouts + offline C2 judgment (study acs-robust-r2).

Covers conditions absent from acs-c2-train's frontier_L.csv: L=75, N-axis
probes (N=10/40), and big-n seeds 1032+. Uses the predecessor's canonical
episode runner (acs-conv-knn common.run_episode) and the same offline C2
judge as eval_c2.py, so rows are directly comparable/mergeable with
frontier_L.csv (columns k,L,seed,t_fire,J,success + n_agents).

Usage:
  python run_knn_refs.py --k 12 --L 250 --seeds 1032-1499 --workers 15
  python run_knn_refs.py --k 8,10,12,19 --L 75 --seeds 1000-1031 --workers 10
  python run_knn_refs.py --k 6,8,9 --L 177 --n-agents 10 --seeds 1000-1031

Outputs: data/knnref/k{k}_L{L}_N{n}/ (per-seed npz, cached) +
         data/knnref/k{k}_L{L}_N{n}_summary.csv
"""
import argparse
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

from eval_c2_r3 import t_fire_c2  # noqa: E402  (same offline judge as policy evals)


def run_one(args):
    k, L, n_agents, steps, seed, outdir = args
    out = os.path.join(outdir, f"k{k}_L{L:g}_N{n_agents}_s{seed}.npz")
    if os.path.exists(out):
        return out
    from common import run_episode, save_run
    rec, snaps, ts, meta = run_episode(k=k, n_agents=n_agents, max_steps=steps,
                                       initial_position_bound=L, seed=seed,
                                       pos_stride=10)
    save_run(out, rec, snaps, ts, meta)
    return out


def judge_npz(path):
    import json
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    t = t_fire_c2(z["phi"], z["s_ent"], z["n_comp_r0"])
    r = z["reward"]
    J = float(-np.nansum(r[1:t + 1])) if t >= 0 else np.nan
    return dict(k=m["k"], L=m["initial_position_bound"], n_agents=m["n_agents"],
                seed=m["seed"], t_fire=t, success=int(t >= 0), J=J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True, help="comma list, e.g. 8,10,12,19")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--n-agents", type=int, default=20)
    ap.add_argument("--seeds", default="1000-1031")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",")]
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))
    for k in ks:
        assert k < args.n_agents, f"k={k} >= N={args.n_agents}"

    import pandas as pd
    for k in ks:
        cond = f"k{k}_L{args.L:g}_N{args.n_agents}"
        outdir = os.path.join(STUDY, "data", "knnref", cond)
        os.makedirs(outdir, exist_ok=True)
        jobs = [(k, args.L, args.n_agents, args.steps, s, outdir) for s in seeds]
        with Pool(args.workers) as pool:
            pool.map(run_one, jobs)
        # Rebuild the summary from ALL npz in the dir, not just this call's
        # seeds — a partial re-run must never clobber a fuller summary.
        import glob as globmod
        paths = globmod.glob(os.path.join(outdir, "*.npz"))
        df = pd.DataFrame([judge_npz(p) for p in sorted(paths)]).sort_values("seed")
        csv_path = os.path.join(STUDY, "data", "knnref", f"{cond}_summary.csv")
        df.to_csv(csv_path, index=False)
        det = df[df.success == 1]
        print(f"=== {cond}: {len(df)} seeds ===")
        print(f"success {int(df.success.sum())}/{len(df)}  "
              f"t_conv med {det.t_fire.median():.0f}  J med {det.J.median():.1f}  "
              f"-> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
