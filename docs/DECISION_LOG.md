# Decision log

Decisions that changed a premise an earlier plan was written on, so that reading
the earlier plan does not mislead. Study documents under `studies/` are *not*
retro-edited: each records what was true when it was written.

---

## 2026-08-29 — Dynamic-k NN becomes the main line; `dknn` is the canonical profile

### What changed

The integration plan (`INTEGRATION_PLAN.md`, external to this repo) was written
on two premises that no longer hold:

1. **"`dknn_legacy` is defined but not run"** (Q3/Q4 rider). The original
   Dynamic-k regime existed in `train_unified.py` only as a preserved
   definition, gated behind `--allow-legacy`, because the integration study
   trained Dynamic-k under the C2 regime instead.
2. **"equal budget per method (~2M steps)"** (Q4 framing). Both methods were to
   be compared at a matched budget under a shared regime.

Re-evaluating the pre-rename checkpoint **ck848** under the unified C2 harness
reversed both. On the confirmation lane (seeds 1500–1999, N=20, L=250, argmax,
6000-step cap) ck848 is 0 failures / 500, t_conv median 532, J median 152.6,
CVaR10 201.0 — the best of every arm measured at that condition, ahead of π_R
(J 180.9), π_E (J 260.4) and every fixed-k reference, and Pareto-dominant over
the whole nearest-k sweep. ck848 came from an **8M-step run under the original
Dynamic-k regime**, not from the C2 regime at 2M.

The arm the equal-budget frame actually produced — Dynamic-k trained under the
C2 regime with the entropy-penalty probe config, best checkpoint at 0.5M —
failed 364/500 (72.8%) on the same lane. The bottleneck was diagnosed as
deterministic-collapse dynamics under a constant entropy penalty, not step
starvation, so extending that recipe's budget was not the fix.

### What was decided

- **The `dknn` profile is the canonical recipe and the default**, defined as the
  ck848 original: fixed 1000-step episodes, legacy shaped reward, minibatch 256,
  10 SGD iters, 8M-step budget, lr 2e-5 → 1e-7 anchored at 8M, RLlib `seed` set.
  `tools/check_ck848_parity.py` checks the resolved config against the archived
  `params.json` field by field and is the regression gate on future edits.
- **The accelerated variant was not adopted.** `legacy/train_dynamic_knn.py`
  defaults to minibatch 512 / 7 SGD iters / 6M steps / 18h cap. No checkpoint
  from that variant has been measured under the unified C2 harness, so it is not
  a reproduction basis; the whole ck848 series in
  `checkpoints/legacy_distance_pointer_260818/` came from the original.
- **`--allow-legacy` was removed.** The guard existed to enforce premise (1).
- **`dknn_c2` is kept, marked experimental.** It remains the starting point for
  a later "8M budget × C2 regime" ablation.
- **README order was inverted**: Dynamic-k NN is presented as the main line,
  binary edge selection second, the nearest-k/ACS-FC family as the reference.

### Profile rename map

Old profile names are gone rather than aliased — the only in-repo caller was
docker's `FLOCK_PROFILE`, and an unknown name now fails immediately on argparse
choices instead of silently training a different recipe.

| old | new | note |
|---|---|---|
| `dknn_legacy` | **`dknn`** | now the default; hyperparameters replaced with the ck848 original |
| `policy_robust` | **`pi_r`** | hyperparameters unchanged |
| `dknn_c2` | `dknn_c2` | unchanged |
| regime `"legacy_dknn"` | regime `"dknn_original"` | contents unchanged |

### Evidence

- In-repo, tracked: `checkpoints/PROVENANCE.md`
  § *Legacy Dynamic-k NN checkpoints (pre-rename, added 2026-08-27)* — md5-verified
  provenance of the ck848 series, its training configuration, and the measured
  C2 numbers.
- In-repo, untracked lane data: `studies/acs-confirm/data/eval/lp848_*_arms.csv`
  and `lp848_confirm_summary.csv` (study `data/` directories are gitignored).
- External to this repo: `/workspace/LEGACY848_REPORT.md` (ck848 re-evaluation),
  `/workspace/B8_B9_REPORT.md` (the C2-regime Dynamic-k arm), and
  `/workspace/INTEGRATION_PLAN.md` (the plan whose premises these reverse).

---

## 2026-08-29 — `eval/` is the criterion of record; study copies are records

The C2 evaluation harness was implemented inside `studies/acs-confirm/src/`
(building on `studies/acs-conv-knn/src/common.py`), reachable only through
`sys.path` inserts and writing into that study's `data/` directory. It has been
promoted to the repo-root `eval/` package: `common.py`, `eval_c2.py`,
`run_knn_refs.py`, `stats.py`, `pair_judge.py`.

The rollout, judge and statistics logic is unchanged. Equivalence was verified
against the archived lane before the promotion commit: ck848 seeds 1500–1504
reproduce `lp848_confirm_summary.csv` exactly (integer columns identical, float
columns bit-identical), k=12/L=250 seeds 1500–1502 reproduce the archived knnref
rows exactly, and `pair_judge` over the full 1500–1999 lane reproduces the
published verdict (McNemar b=0 c=32, p=4.66e-10; co-success dJ median −10.4).

The study copies stay in place, unmodified. `checkpoints/PROVENANCE.md`'s
earlier instruction to "always use the acs-confirm copies" is superseded: the
version to run is `eval/`.

`confirm_judge.py`, `check_gates.py`, `make_tables.py` and
`eval_project_nearest.py` were **not** promoted — they are bound to
acs-confirm's registered arms and lanes. Only `confirm_judge.py`'s statistics
kit (Wilson CI, exact McNemar, CVaR10, co-success paired dJ) was lifted, into
`eval/stats.py`.

---

## 2026-08-29 — Retired trainers moved to their studies; old paths in study docs left alone

Six single-method trainers moved out of the repo root, contents unmodified:

| was | now |
|---|---|
| `train_robust.py` | `studies/acs-robust-train/src/` |
| `train_robust2.py` | `studies/acs-robust-r2/src/` |
| `train_c2_a.py`, `train_c2_b.py` | `studies/acs-c2-train/src/` |
| `train.py`, `train_hardtopk.py` | `legacy/` |
| `train_dynamic_knn.py` | `legacy/` |

They import only repo-root modules, so they still run from the repo root with
`PYTHONPATH=.` — `checkpoints/PROVENANCE.md` spells that out per command.

**`train.py` was deliberately not renamed away and `train_unified.py` was
deliberately not renamed to `train.py`.** The identity of the old root
`train.py` (the continuous/beta_dist trainer) is described in four study
records; renaming would silently repoint those references at a different file.

Study documents that mention the old root paths are left as they are. They
record what was true when they were written; this entry is the pointer.

---

## 2026-08-29 — Other consolidation decisions

- **`dynamic_k_nn` package → single module.** The package was two files whose
  `__init__.py` only re-exported the five constants in `identifiers.py`. Import
  sites now read `from dynamic_k_nn import ...`. No study code imported it —
  `eval_c2_r3.py` hardcodes the identifier strings.
- **`baselines.py` split.** Nine of its twelve heuristics (~1,800 of 2,100
  lines) are referenced only by `test_baselines.py` and the dormant
  `legacy/verify_*.py`. They moved to `legacy/baselines_extra.py` and are
  re-exported from the bottom of `baselines.py`, so every import site and
  `create_baseline` type still resolves unchanged.
- **`evaluate_checkpoint.py`, `models/ppo_centralized.py`,
  `models/beta_dist.py` stay put.** `studies/acs-conv-knn/src/run_nn_rollouts.py`
  imports `RLPolicy` from the first at runtime, `models/__init__.py` imports the
  second unconditionally, and the third is used by the trainers now in
  `legacy/`. Moving any of them would break a record script for no benefit;
  they are dormant code with zero carrying cost.
- **In-training evaluation is monitoring, not selection.** Every profile
  evaluates under the C2 protocol so the signal is comparable to the criterion
  of record, but checkpoint selection is the offline `eval/` harness's job:
  `--rank-runs` screens a run, then dev seeds 1000–1031, then confirmation seeds
  1500–1999.
- **Checkpoints: freq 8, keep everything.** Tune's `keep_checkpoints_num` scores
  on `episode_reward_mean`, which is not the criterion of record, so it would
  prune on the wrong axis.
