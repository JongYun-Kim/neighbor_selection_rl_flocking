# Canonical checkpoints — provenance

Canonical checkpoint packages live here with the same load contract as Tune
trials: `params.json` is the sibling of the RLlib checkpoint directory. Unlike
the regenerable `test_results/` tree, files under `checkpoints/` are intentionally
version-controlled so a fresh clone can load the recorded policy directly.

## Dynamic-k NN training-curve checkpoint (2026-09-01 run)

`dynamic_k_nn_best/checkpoint_000592` is the best **saved** checkpoint by
`episode_reward_mean` in the completed 6M-step `main-dknn-wandb-ref-s42-mb512-e7-6m-20260901`
run. Its score is -64.85467598696692 at iteration 592 / 4,849,664 timesteps.
Iterations 594–595 reached a slightly higher curve value (-64.69183846712471),
but no checkpoint was written at those iterations, so checkpoint 592 is the
loadable training-curve winner. This designation is not a C2 confirmation result.

The package contains the complete RLlib checkpoint, its sibling `params.json`,
the original `params.pkl`, selection metadata, and SHA-256 checksums. Load it via:

```bash
python -m eval population \
  --run-id <run-id> --output-root test_results/evaluation
```

The historical entries below document earlier policy handoffs. Their binaries
must be copied from the source machine when they are not present in a checkout.

Unpack a handed-over archive at exactly the paths in the tables below: the layout
*is* the contract, because every tool finds `params.json` by walking up from the
checkpoint directory. One file is needed even when you are not re-measuring
anything — the ck848 run's `params.json` (~2.6 KB), without which
`tools/check_ck848_parity.py` exits with `FileNotFoundError`. Its `--ref` flag
takes an out-of-tree path if you would rather not move the archive.

| policy | path (under `checkpoints/`) | role |
|---|---|---|
| **π_E** | `c2C1_ft40_lmix_260808/manual/checkpoint_000080` | Headline efficiency policy (C1): weights-only fine-tune from A it40 over L-mix. Closes the J gap (−19/−1/+31 vs specialist). Confirmed in acs-confirm; sole registered miss: N10 failure 2.2% (n.s.), N-claim demoted per pre-registered mapping. |
| **π_R** | `c2R1_lmix_legacy_260807/GradLoggingPPO_…_cc4e6_…/checkpoint_000110` | Reliability/insurance policy (R1): scratch L-mix training, legacy obs. Failures 1/1000 vs k12 85/1000 (S1); insurance confirmed on every axis in acs-confirm (L 2/1500, N 1/1000). |
| **A it40** | `c2A_bernoulli_260806/GradLoggingPPO_…_d5510_…/checkpoint_000040` | L=250 specialist (variant A, bernoulli head) at iter 40 — the fine-tune init that produced π_E. Kept for lineage/reproduction. |

## Legacy Dynamic-k NN checkpoints (pre-rename, added 2026-08-27)

`legacy_distance_pointer_260818/PPO_neighbor_selection_flocking_env_8d07b_00000_0_2026-08-18_06-10-46/`
holds `checkpoint_{000848,000856,000952,000960,000968,000977}` plus the run's original
`params.json`/`params.pkl`, copied verbatim (md5-verified) from the
`neighbor_selection_rl_flocking_legacy_checkpoint` clone
(`test_results/distance_pointer_neighbor_selection/`). That clone and the
`dynamic_k_nn` branch clone are siblings off commit `b23c3aa`, sharing 89 of 92
commits; `models/ppo_dynamic_k_nn.py` and `models/modules/*` are byte-identical
between them, so these weights load into the current model with
`strict=True` (100/100 keys) and forward outputs match bitwise.

The run predates the Dynamic-k NN rename, so its `params.json` records
`custom_model: distance_pointer_neighbor_selector_rl` and
`action_type: distance_pointer`. It is kept unedited — `eval_c2_r3.py` maps both
spellings (commit `500b8b4`), as does the promoted `eval/eval_c2.py`.
Training: 977 iterations / 8.0M steps, lr 2e-5,
batch 8192, minibatch 256, sgd_iter 10, N=20, L=250 fixed, legacy reward, **no
entropy penalty**; pointer entropy decays 59.66 -> 15.41 over the run.

Measured under the acs-confirm criterion of record (seeds 1500-1999, n=500,
argmax, 6000 steps): ck848 = 100% success, t_conv 532, J 152.6, CVaR10 201.0 —
the best of every arm measured at L250/N20. Full comparison and the robustness
matrix: `studies/legacy-ck848/REPORT_KO.md`.

## Reproduction (repo root, pinned stack, GPU)

These single-method trainers now live in the study that ran them, unmodified.
They import repo-root modules, so run them from the repo root with `PYTHONPATH=.`:

- A / it40: `PYTHONPATH=. python studies/acs-c2-train/src/train_c2_a.py` — NOTE: its
  `RUN_NAME` constant currently points at the ablation run (`c2A2_…`); set it back to
  `c2A_bernoulli_260806`-style to reproduce the canonical run.
  Study: `studies/acs-c2-train/`.
- π_R: `PYTHONPATH=. python studies/acs-robust-train/src/train_robust.py --variant legacy`
  (120 it; ckpt of record = it110). Study: `studies/acs-robust-train/`.
  The same recipe is also the `pi_r` profile of `train_unified.py`.
- π_E: `PYTHONPATH=. python studies/acs-robust-r2/src/train_robust2.py --run-name <name>
  --init-ckpt <A it40 path>` (weights-only init + init-fidelity gate, flat lr 1e-4,
  80 it, L pool {125,250,500}). Study: `studies/acs-robust-r2/`.
- ck848 (Dynamic-k NN): `python train_unified.py --profile dknn` — the default profile
  is that run's original recipe, and `python train_unified.py --profile dknn --dry-run |
  python tools/check_ck848_parity.py` checks the resolved config field-by-field against
  the `params.json` archived here.

Original artifacts remain in `test_results/<run>/…` (full checkpoint series, tune
logs, `result.json`). Training was seeded but full bit-reproducibility across
hardware is not guaranteed — for exact numbers use these checkpoint copies.

## Evaluation

Criterion-of-record offline eval is the repo-root `eval/` package, run from the
repo root:

```
python -m eval.eval_c2 --ckpt <checkpoint dir> --label <name> --seeds 1500-1999
python -m eval.run_knn_refs --k 12 --L 250 --seeds 1500-1999
python -m eval.pair_judge --arm pol=<name> --arm k12=knnref:12,250,20
```

It was promoted from `studies/acs-confirm/src/` and reproduces that study's
archived numbers exactly. The copies under `studies/*/src/` are the records of
the studies that produced them, not the version to run; earlier studies'
`eval_c2.py` copies are additionally outdated (acs-c2-train's lacks the
`obs_position_scale` handling). Results of the confirmation study:
`studies/acs-confirm/REPORT_KO.md`.

`legacy/evaluate_checkpoint.py` (retired; it was at the repo root until
2026-08-29) is the pre-C2 Monte Carlo harness for the centralized-obs model
variant and is not the judge for these ego-centric policies. Its policy-loading
half now lives on as `eval/policies.py`.
