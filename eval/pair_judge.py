"""Arm-pair comparison matrix over C2 evaluation summaries.

confirm_judge.py hard-binds its verdicts to the archived confirmation lanes;
this reuses its registered stats kit (Wilson CI, exact McNemar, CVaR10,
co-success paired dJ) for an arbitrary arm set, pairing policy arms with
nearest-k references on a shared seed range.

Arm spec (repeatable --arm), resolved against --base:
  NAME=LABEL            <base>/LABEL_summary.csv
  NAME=knnref:K,L,N     <base>/../knnref/kK_LL_NN_summary.csv

--base defaults to the eval harness output directory (test_results/eval), so
freshly produced summaries need no flag; point it at a study data directory
(e.g. studies/acs-confirm/data/eval) to re-read an archived lane.

Usage (always from the repo root):
  python -m eval.pair_judge --seeds 1500-1999 \
      --arm lp848=lp848_confirm --arm k12=knnref:12,250,20 [--csv-out out.csv]

Every arm is summarized on the seed range; every ordered arm pair gets
McNemar + co-success dJ, so the headline policy-vs-policy pairing and the
policy-vs-reference insurances read from one table.

Promoted from studies/acs-confirm/src/pair_judge.py, which stays in place as
the record of that study. The diff is the stats-kit import and the --base
flag that replaces the study's hardcoded data directory.
"""
import argparse
import os

import numpy as np
import pandas as pd

from utils.paths import repo_path
from eval.stats import wilson, mcnemar, cvar10, paired_dj, _load, pair

DEFAULT_BASE = repo_path("test_results", "eval")


def load_arm(spec, seed_range, base):
    """Resolve one --arm spec to a summary CSV and load its seed range.

    Policy arms live directly in <base>; nearest-k references live in the
    sibling knnref/ directory, matching the layout both the eval harness
    (test_results/{eval,knnref}) and the study archives use."""
    name, src = spec.split("=", 1)
    if src.startswith("knnref:"):
        k, L, n = src.split(":", 1)[1].split(",")
        knnref = os.path.join(os.path.dirname(os.path.normpath(base)), "knnref")
        path = f"{knnref}/k{int(k)}_L{float(L):g}_N{int(n)}_summary.csv"
    else:
        path = f"{base}/{src}_summary.csv"
    df = _load(path, seed_range)
    if df is None:
        raise SystemExit(f"missing CSV for arm {name}: {path}")
    return name, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=LABEL or NAME=knnref:K,L,N (repeatable)")
    ap.add_argument("--seeds", default="1500-1999")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="directory holding <LABEL>_summary.csv; knnref arms "
                         "read the sibling knnref/ dir "
                         "(default: test_results/eval)")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-"))
    arms = dict(load_arm(s, (lo, hi), args.base) for s in args.arm)

    print(f"=== arms @ seeds {lo}-{hi} ===")
    per_arm = []
    for name, df in arms.items():
        n = len(df)
        f = int((df.success == 0).sum())
        w_lo, w_hi = wilson(f, n)
        js = df.J[df.success == 1].astype(float)
        row = dict(arm=name, n=n, fail=f, fail_rate=f / n,
                   wilson_lo=w_lo, wilson_hi=w_hi,
                   t_conv_med=float(df.t_fire[df.success == 1].median()) if f < n else np.nan,
                   J_med=float(js.median()) if len(js) else np.nan,
                   cvar10_J=cvar10(js))
        per_arm.append(row)
        print(f"  {name:6s} n={n:4d} fail={f:4d} ({f/n:6.1%}) "
              f"Wilson[{w_lo:.3f},{w_hi:.3f}] t_conv_med={row['t_conv_med']:.0f} "
              f"J_med={row['J_med']:.1f} CVaR10={row['cvar10_J']:.1f}"
              if f < n else
              f"  {name:6s} n={n:4d} fail={f:4d} (100.0%) — no successes")

    print(f"\n=== pairs (row vs col; McNemar b=row-only-fails c=row-only-wins) ===")
    pair_rows = []
    names = list(arms)
    for a in names:
        for b in names:
            if a == b:
                continue
            da, db = pair(arms[a], arms[b])
            if not len(da):
                print(f"  {a} vs {b}: no shared seeds")
                continue
            mb, mc, mp = mcnemar(da, db)
            dj = paired_dj(da, db)
            pair_rows.append(dict(pol=a, ref=b, n_shared=len(da),
                                  mcnemar_b=mb, mcnemar_c=mc, mcnemar_p=mp,
                                  **{f"dj_{k}": v for k, v in dj.items()}))
            print(f"  {a:6s} vs {b:6s} n={len(da):4d} "
                  f"McNemar b={mb:3d} c={mc:3d} p={mp:.3g} | "
                  f"co-succ n={dj['n_pair']:4d} dJ_med={dj['med']:.1f} "
                  f"wilcox_p={dj['wilcox_p']:.3g} sign_p={dj['sign_p']:.3g}")

    if args.csv_out:
        pd.DataFrame(per_arm).to_csv(args.csv_out.replace(".csv", "_arms.csv"), index=False)
        pd.DataFrame(pair_rows).to_csv(args.csv_out.replace(".csv", "_pairs.csv"), index=False)
        print(f"\nwrote {args.csv_out.replace('.csv', '_arms.csv')} / _pairs.csv")


if __name__ == "__main__":
    main()
