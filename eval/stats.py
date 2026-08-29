"""Registered statistics kit for arm comparison (Wilson / McNemar / CVaR10 / dJ).

The six functions below are lifted verbatim from
studies/acs-confirm/src/confirm_judge.py, the pre-registered P1-P5 judge of
study acs-confirm (registered analysis of that study's PROBLEM.md, ratified
2026-08-11 21:52 UTC): per-arm failure counts + Wilson 95% CI + paired exact
McNemar (primary); CVaR10 + co-success paired median dJ + Wilcoxon
signed-rank + sign test (secondary). No mean-based t-tests feed any verdict.

Only the kit is promoted. The P1-P5 verdict sections of confirm_judge.py are
hard-bound to that study's arms and data paths, so they stay there as the
record; eval.pair_judge is the reusable consumer of this kit.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats


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


def pair(pol_df, ref_df):
    c = pol_df.index.intersection(ref_df.index)
    return pol_df.loc[c], ref_df.loc[c]
