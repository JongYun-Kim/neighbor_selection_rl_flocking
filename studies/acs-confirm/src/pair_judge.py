"""Arm-pair comparison matrix for the unified B8 study (plan §B8.4).

confirm_judge.py hard-binds its verdicts to the archived confirmation lanes;
this reuses its registered stats kit (Wilson CI, exact McNemar, CVaR10,
co-success paired dJ) for the B8 arm set, which pairs freshly trained arms
(pi_E', pi_R', dknn-best) with nearest-k references on a shared seed range.

Arm spec (repeatable --arm):
  NAME=LABEL            studies/acs-confirm/data/eval/LABEL_summary.csv
  NAME=knnref:K,L,N     studies/acs-confirm/data/knnref/kK_LL_NN_summary.csv

Usage:
  python pair_judge.py --seeds 1500-1999 \
      --arm piE=c2C1ft_uni_best --arm piR=piR_best --arm dknn=dknn_best \
      --arm k12=knnref:12,250,20 [--csv-out out.csv]

Every arm is summarized on the seed range; every ordered arm pair gets
McNemar + co-success dJ, so the headline policy-vs-policy pairing and the
policy-vs-reference insurances read from one table.
"""
import argparse

import numpy as np
import pandas as pd

from confirm_judge import STUDY, wilson, mcnemar, cvar10, paired_dj, _load, pair


def load_arm(spec, seed_range):
    name, src = spec.split("=", 1)
    if src.startswith("knnref:"):
        k, L, n = src.split(":", 1)[1].split(",")
        path = f"{STUDY}/data/knnref/k{int(k)}_L{float(L):g}_N{int(n)}_summary.csv"
    else:
        path = f"{STUDY}/data/eval/{src}_summary.csv"
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
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-"))
    arms = dict(load_arm(s, (lo, hi)) for s in args.arm)

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
