"""Pre-registered P1-P5 machine judgment (study acs-confirm).

Implements EXACTLY the registered analysis of PROBLEM.md (ratified
2026-08-11 21:52 UTC): per-arm failure counts + Wilson 95% CI + paired exact
McNemar (primary); CVaR10 + co-success paired median dJ + Wilcoxon
signed-rank + sign test (secondary). No mean-based t-tests feed any verdict.

Sections print PENDING when their input lanes are not finished yet, so the
script can run incrementally while rollouts stream in. --sanity-only stops
after the archive-reproduction block (stats machinery check, no fresh data).

Usage: python confirm_judge.py [--sanity-only] [--csv-out data/confirm_stats.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import os as _os
_ROOT = _os.environ.get("FLOCK_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
STUDY = _os.path.join(_ROOT, "studies", "acs-confirm")
R2 = _os.path.join(_ROOT, "studies", "acs-robust-r2", "data")
R3 = _os.path.join(_ROOT, "studies", "acs-robust-r3-stress", "data")

FRESH_N20 = (1500, 1999)   # arms (1): fresh seeds, N=20
NAXIS = (1000, 1499)       # arms (3): N-axis seeds (pair with r3 refs)


# ------------------------------------------------------------------ stats kit
def wilson(fail, n, z=1.96):
    p = fail / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(pol, ref):
    """Exact two-sided McNemar on paired success columns (audit_stats.py
    convention): b = pol fails where ref succeeds, c = pol succeeds where ref
    fails, p = exact binomial two-sided of b among b+c."""
    b = int(((pol.success == 0) & (ref.success == 1)).sum())
    c = int(((pol.success == 1) & (ref.success == 0)).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else np.nan
    return b, c, p


def cvar10(j_success):
    """Mean of the WORST (largest) 10% of success-J values, descriptive."""
    j = np.sort(np.asarray(j_success, dtype=float))
    if len(j) == 0:
        return np.nan
    m = max(1, int(np.ceil(0.1 * len(j))))
    return float(np.mean(j[-m:]))


def paired_dj(pol, ref):
    """Co-success paired dJ (pol - ref): median, Wilcoxon, sign test."""
    both = (pol.success == 1) & (ref.success == 1)
    d = (pol.J[both] - ref.J[both]).values
    n = len(d)
    if n < 6:
        return dict(n_pair=n, med=np.nan, wilcox_p=np.nan, worse=np.nan,
                    sign_p=np.nan)
    return dict(n_pair=n, med=float(np.median(d)),
                wilcox_p=float(stats.wilcoxon(d).pvalue),
                worse=int((d > 0).sum()),
                sign_p=float(stats.binomtest(int((d > 0).sum()), n, 0.5).pvalue))


# ------------------------------------------------------------------ loaders
def _load(path, seed_range):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[(df.seed >= seed_range[0]) & (df.seed <= seed_range[1])]
    return df.set_index("seed").sort_index()


def pol(label, seed_range):
    return _load(f"{STUDY}/data/eval/{label}_summary.csv", seed_range)


def knn(k, L, N, seed_range, base=STUDY + "/data"):
    return _load(f"{base}/knnref/k{k}_L{L:g}_N{N}_summary.csv", seed_range)


def complete(df, n=500):
    return df is not None and len(df) == n


def pair(pol_df, ref_df):
    c = pol_df.index.intersection(ref_df.index)
    return pol_df.loc[c], ref_df.loc[c]


VERDICTS = []


def verdict(pid, name, ok, measured):
    VERDICTS.append(dict(item=pid, name=name,
                         verdict="PASS" if ok else "FAIL", measured=measured))
    print(f"  [{pid}] {name}: {'PASS' if ok else 'FAIL'}  ({measured})")


# ------------------------------------------------------------------ sanity
def sanity():
    """Reproduce published archive numbers with THIS script's machinery."""
    print("=== SANITY: archive reproduction (r3/r2, published numbers) ===")
    exp_fail = [
        ("r3 k11@L125", knn(11, 125, 20, NAXIS, R3), 14),
        ("r3 k12@L125", knn(12, 125, 20, NAXIS, R3), 20),
        ("r3 k13@L250", knn(13, 250, 20, NAXIS, R3), 25),
        ("r3 k10@L500", knn(10, 500, 20, NAXIS, R3), 36),
        ("r3 k6@N10L177", knn(6, 177, 10, NAXIS, R3), 21),
        ("r3 k24@N40L354", knn(24, 354, 40, NAXIS, R3), 18),
        ("r3 k28@N40L354", knn(28, 354, 40, NAXIS, R3), 18),
        ("r2 k12@L250", knn(12, 250, 20, NAXIS, R2), 40),
        ("r2 k12@L500", knn(12, 500, 20, NAXIS, R2), 45),
        ("r3 C1@L125", _load(f"{R3}/eval/C1_i80_L125_s500_summary.csv", NAXIS), 20),
        ("r3 R1@L125", _load(f"{R3}/eval/R1_i110_L125_s500_summary.csv", NAXIS), 0),
    ]
    warn = 0
    for name, df, want in exp_fail:
        if df is None:
            print(f"  {name}: MISSING FILE")
            warn += 1
            continue
        got = int((df.success == 0).sum())
        tag = "ok" if (got == want and len(df) == 500) else "WARN"
        warn += tag == "WARN"
        print(f"  {name}: fail {got}/{len(df)} (expect {want}/500) [{tag}]")
    c1 = _load(f"{R3}/eval/C1_i80_L125_s500_summary.csv", NAXIS)
    r1 = _load(f"{R3}/eval/R1_i110_L125_s500_summary.csv", NAXIS)
    k11 = knn(11, 125, 20, NAXIS, R3)
    for name, p_df, want in (("C1 vs k11@L125", c1, 0.39), ("R1 vs k11@L125", r1, 1.2e-4)):
        a, b = pair(p_df, k11)
        _, _, mp = mcnemar(a, b)
        tag = "ok" if np.isclose(mp, want, rtol=0.15) else "WARN"
        warn += tag == "WARN"
        print(f"  McNemar {name}: p={mp:.3g} (expect ~{want}) [{tag}]")
    print(f"SANITY {'PASS (machinery reproduces archives)' if not warn else f'{warn} WARN'}")
    return warn == 0


# ------------------------------------------------------------------ P1-P5
def p1_p2():
    fresh_k12 = {L: knn(12, L, 20, FRESH_N20) for L in (125, 250, 500)}
    for pid, lab, pred in (("P1", "piR", "0-2/1500 total, <=5/500 each"),
                           ("P2", "piE", "L250/L500 <=5; L125 2-7% & n.s.")):
        print(f"\n=== {pid} ({lab} vs fresh k12; prediction: {pred}) ===")
        for L in (125, 250, 500):
            p = pol(f"{lab}_L{L}", FRESH_N20)
            k = fresh_k12[L]
            if not (complete(p) and complete(k)):
                print(f"  L{L}: PENDING ({0 if p is None else len(p)}/500 pol, "
                      f"{0 if k is None else len(k)}/500 k12)")
                continue
            a, b = pair(p, k)
            pf, kf = int((a.success == 0).sum()), int((b.success == 0).sum())
            lo, hi = wilson(pf, len(a))
            mb, mc, mp = mcnemar(a, b)
            base = (f"fail {pf}/500 Wilson[{lo:.4f},{hi:.4f}] vs k12 {kf}/500; "
                    f"McNemar b={mb} c={mc} p={mp:.3g}")
            if pid == "P1" or L != 125:
                ok = pf <= 5 and (mp < 0.05 if np.isfinite(mp) else False) and pf < kf
                verdict(pid, f"{lab}@L{L} insured (<=1% & McNemar<.05 & fewer)", ok, base)
            else:
                hit = (10 <= pf <= 35) and (not np.isfinite(mp) or mp >= 0.05)
                verdict("P2pred", f"piE@L125 predicted weak (2-7% & n.s. vs k12)",
                        hit, base)


def p3():
    print("\n=== P3 (fixed-k defect >=2% each; FC <=0.4%) ===")
    arms = {125: (11, 12, 13), 250: (12, 13), 500: (10, 12, 13)}
    for L, ks in arms.items():
        for k in ks:
            df = knn(k, L, 20, FRESH_N20)
            if not complete(df):
                print(f"  k{k}@L{L}: PENDING ({0 if df is None else len(df)}/500)")
                continue
            f = int((df.success == 0).sum())
            lo, hi = wilson(f, len(df))
            verdict("P3", f"k{k}@L{L} fail>=2%", f >= 10,
                    f"fail {f}/500 = {f / 5:.1f}% Wilson[{lo:.4f},{hi:.4f}]")
        fc = knn(19, L, 20, FRESH_N20)
        if not complete(fc):
            print(f"  FC k19@L{L}: PENDING ({0 if fc is None else len(fc)}/500)")
            continue
        f = int((fc.success == 0).sum())
        verdict("P3", f"FC(k19)@L{L} fail<=0.4%", f <= 2, f"fail {f}/500")


def p4():
    print("\n=== P4 (secondary: co-success paired median dJ directions) ===")
    checks = []  # (name, pol_label, ref(k), L, judge_fn, describe)
    for L in (125, 250, 500):
        checks.append((f"piE vs FC L{L}", "piE", 19, L,
                       lambda s: s["med"] < 0, "med<0"))
    for L, cond, desc in ((125, lambda s: -15 <= s["med"] <= 10, "[-15,+10]"),
                          (250, lambda s: -15 <= s["med"] <= 10, "[-15,+10]"),
                          (500, lambda s: s["med"] > 20, ">+20")):
        checks.append((f"piE vs k12 L{L}", "piE", 12, L, cond, desc))
    checks.append(("piR vs FC L125", "piR", 19, 125, lambda s: s["med"] < 0, "med<0"))
    checks.append(("piR vs FC L500", "piR", 19, 500,
                   lambda s: abs(s["med"]) < 25 and s["wilcox_p"] >= 0.05,
                   "|med|<25 & n.s."))
    for name, lab, k, L, cond, desc in checks:
        p = pol(f"{lab}_L{L}", FRESH_N20)
        r = knn(k, L, 20, FRESH_N20)
        if not (complete(p) and complete(r)):
            print(f"  {name}: PENDING")
            continue
        a, b = pair(p, r)
        s = paired_dj(a, b)
        ok = np.isfinite(s["med"]) and cond(s)
        verdict("P4", f"{name} {desc}", ok,
                f"med {s['med']:+.1f} n={s['n_pair']} w_p {s['wilcox_p']:.3g} "
                f"worse {s['worse']}/{s['n_pair']} s_p {s['sign_p']:.3g}")
    # CVaR10 descriptive block (registered secondary, no verdict)
    print("  -- CVaR10 (worst-10% mean success-J; descriptive) --")
    for L in (125, 250, 500):
        row = []
        for name, df in (("piE", pol(f"piE_L{L}", FRESH_N20)),
                         ("piR", pol(f"piR_L{L}", FRESH_N20)),
                         ("k12", knn(12, L, 20, FRESH_N20)),
                         ("k13", knn(13, L, 20, FRESH_N20)),
                         ("FC", knn(19, L, 20, FRESH_N20))):
            row.append(f"{name} {cvar10(df.J[df.success == 1]) if complete(df) else np.nan:.1f}"
                       if complete(df) else f"{name} PEND")
        print(f"  L{L}: " + "  ".join(row))


def p5():
    print("\n=== P5 (N-axis n=500, seeds 1000-1499) ===")
    refs = {10: [(6, 177, R3)], 40: [(24, 354, R3), (28, 354, R3)]}
    for N, L in ((10, 177), (40, 354)):
        for lab in ("piE", "piR"):
            p = pol(f"{lab}_N{N}L{L}", NAXIS)
            if not complete(p):
                print(f"  {lab}@N{N}L{L}: PENDING ({0 if p is None else len(p)}/500)")
                continue
            f = int((p.success == 0).sum())
            lo, hi = wilson(f, len(p))
            verdict("P5", f"{lab}@N{N}L{L} fail<=2%", f <= 10,
                    f"fail {f}/500 Wilson[{lo:.4f},{hi:.4f}] (pred <=1%)")
            for k, Lr, base in refs[N]:
                r = knn(k, Lr, N, NAXIS, base)
                if not complete(r):
                    print(f"    vs k{k}: ref PENDING/MISSING")
                    continue
                a, b = pair(p, r)
                mb, mc, mp = mcnemar(a, b)
                rf = int((b.success == 0).sum())
                ok = (mp < 0.05 if np.isfinite(mp) else False) and f < rf
                verdict("P5", f"{lab}@N{N} McNemar< vs k{k} (ref fail {rf})", ok,
                        f"b={mb} c={mc} p={mp:.3g}")
    for N, L, k in ((10, 177, 9), (40, 354, 39)):
        df = knn(k, L, N, NAXIS)
        if not complete(df):
            print(f"  FC k{k}@N{N}: PENDING ({0 if df is None else len(df)}/500)")
            continue
        f = int((df.success == 0).sum())
        verdict("P5", f"FC(k{k})@N{N}L{L} fail<=1/500", f <= 1, f"fail {f}/500")


def ablation():
    print("\n=== Optional ablation (nearest-projection; registered wording) ===")
    for L in (250, 125):
        df = pol(f"ablE_L{L}", FRESH_N20)
        if not complete(df):
            print(f"  ablE@L{L}: PENDING ({0 if df is None else len(df)}/500)")
            continue
        f = int((df.success == 0).sum())
        lo, hi = wilson(f, len(df))
        read = ("fail>=2% -> non-nearest selection IS causal for insurance"
                if f >= 10 else
                ("fail<=1% -> distillable to a k(t) schedule" if f <= 5 else
                 "1-2% gray zone (registered thresholds straddled)"))
        print(f"  ablE@L{L}: fail {f}/500 Wilson[{lo:.4f},{hi:.4f}] "
              f"rank_dev_ss med {df.rank_dev_ss.median():.3f} -> {read}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--csv-out", default=f"{STUDY}/data/confirm_stats.csv")
    args = ap.parse_args()

    ok = sanity()
    if args.sanity_only:
        return
    if not ok:
        print("!! sanity WARN — verdicts below are suspect until resolved")
    p1_p2()
    p3()
    p4()
    p5()
    ablation()

    df = pd.DataFrame(VERDICTS)
    if len(df):
        df.to_csv(args.csv_out, index=False)
        n_pass = int((df.verdict == "PASS").sum())
        print(f"\n=== TOTAL: {n_pass}/{len(df)} PASS -> {args.csv_out} ===")
        for pid in ("P1", "P2", "P2pred", "P3", "P4", "P5"):
            sub = df[df.item == pid]
            if len(sub):
                print(f"  {pid}: {int((sub.verdict == 'PASS').sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
