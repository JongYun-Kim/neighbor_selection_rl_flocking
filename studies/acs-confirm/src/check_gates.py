"""Gate checks G1/G2 for acs-confirm: copy-fidelity vs archived npz.

G1: fresh-runner k12@L250 seeds 1000-1007 must be bit-exact vs the r2
    archive (np.array_equal with equal_nan=True — r3's NaN trap).
G2: fresh-runner C1(pi_E)@L250 seeds 1000-1003 must match the r3 archive:
    bit-exact arrays where possible, and judged t_fire equal / dJ == 0.
"""
import json
import os
import sys

import numpy as np

_ROOT = os.environ.get("FLOCK_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, "studies", "acs-confirm", "src"))
from eval_c2_r3 import t_fire_c2  # noqa: E402

CONFIRM = os.path.join(_ROOT, "studies", "acs-confirm", "data")
R2 = os.path.join(_ROOT, "studies", "acs-robust-r2", "data")
R3 = os.path.join(_ROOT, "studies", "acs-robust-r3-stress", "data")


def arrays_equal(a, b):
    try:
        return np.array_equal(a, b, equal_nan=True)
    except TypeError:
        return np.array_equal(a, b)


def cmp_npz(mine, theirs, skip_meta_keys=()):
    za = np.load(mine, allow_pickle=True)
    zb = np.load(theirs, allow_pickle=True)
    msgs = []
    ka, kb = set(za.files), set(zb.files)
    for k in sorted(ka & kb):
        if k == "meta":
            ma, mb = json.loads(str(za["meta"])), json.loads(str(zb["meta"]))
            diff = sorted(kk for kk in set(ma) | set(mb)
                          if kk not in skip_meta_keys and ma.get(kk) != mb.get(kk))
            if diff:
                msgs.append(f"meta diff: {diff}")
        elif not arrays_equal(za[k], zb[k]):
            msgs.append(f"array '{k}' differs")
    if ka != kb:
        msgs.append(f"key sets differ: only_mine={sorted(ka - kb)} only_theirs={sorted(kb - ka)}")
    return msgs


def judge(path):
    z = np.load(path, allow_pickle=True)
    t = t_fire_c2(z["phi"], z["s_ent"], z["n_comp_r0"])
    J = float(-np.nansum(z["reward"][1:t + 1])) if t >= 0 else np.nan
    return t, J


def main():
    print("=== G1: k12@L250 seeds 1000-1007 vs r2 archive (bit-exact) ===")
    g1_ok = True
    for s in range(1000, 1008):
        mine = f"{CONFIRM}/knnref/k12_L250_N20/k12_L250_N20_s{s}.npz"
        theirs = f"{R2}/knnref/k12_L250_N20/k12_L250_N20_s{s}.npz"
        msgs = cmp_npz(mine, theirs)
        print(f"  s{s}: {'OK' if not msgs else 'MISMATCH ' + '; '.join(msgs)}")
        g1_ok &= not msgs
    print(f"G1 {'PASS' if g1_ok else 'FAIL'}")

    print("\n=== G2: pi_E@L250 seeds 1000-1003 vs r3 C1_i80_L250_s500 ===")
    g2_ok = True
    for s in range(1000, 1004):
        mine = f"{CONFIRM}/eval/gateG2_C1_L250/gateG2_C1_L250_s{s}.npz"
        theirs = f"{R3}/eval/C1_i80_L250_s500/C1_i80_L250_s500_s{s}.npz"
        msgs = cmp_npz(mine, theirs, skip_meta_keys=("policy",))
        tm, jm = judge(mine)
        tt, jt = judge(theirs)
        dj = jm - jt if tm >= 0 else float("nan")
        bit = "bit-exact" if not msgs else "; ".join(msgs)
        print(f"  s{s}: t_fire {tm} vs {tt} (dt={tm - tt})  dJ={dj:.6f}  [{bit}]")
        g2_ok &= (tm == tt) and (dj == 0.0) and not msgs
    print(f"G2 {'PASS' if g2_ok else 'FAIL'}")

    print("\n=== G3: fresh-seed smoke k12@L250 seeds 1500-1503 (re-judge) ===")
    for s in range(1500, 1504):
        t, J = judge(f"{CONFIRM}/knnref/k12_L250_N20/k12_L250_N20_s{s}.npz")
        print(f"  s{s}: t_fire={t}  J={J:.1f}")
    print("G3 PASS (pipeline produces judged fresh-seed episodes)")


if __name__ == "__main__":
    main()
