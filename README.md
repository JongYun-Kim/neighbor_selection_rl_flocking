# Neighbor-Selection RL for Flocking — Dynamic-k NN × Binary Edge Selection

RL for **which neighbors to listen to** in a flocking swarm. The policy never
outputs motion commands: it selects a neighbor subgraph, and the low-level
ACS/Vicsek controller inside the simulator
(`NeighborSelectionFlockingEnv`) turns that subgraph into velocity updates. Two
policy families share the simulator and one Transformer/pointer core:

- **`dynamic_knn` — cutoff-pointer Dynamic-k NN (main line)**: for every ego
  agent the policy points to one active *cutoff* agent; that agent and every
  active agent no farther away become directed neighbors, so the number of
  external neighbors `k` changes with the observation and the policy output.
  Pointing at the ego itself means `k=0`; equal-distance ties are included
  (`action_type="dynamic_k_nn"`, model `DynamicKNNPPORLlib` — encoder/pointer
  core with an ego-wise N-way categorical head).
- **`policy` — binary edge selection**: PPO outputs a binary adjacency matrix
  per step (`action_type="binary_vector"`, model `NeighborSelectionPPORLlib`).

The reference point for both is the heuristic family in `baselines.py`: nearest-k
selection at each fixed `k`, with **ACS FC** (fully connected, `k = N-1`) as the
upper end of that sweep.

## Current results (2026-08)

Everything below is measured under one **criterion of record (C2)**: 500 fresh
seeds (1500–1999), N=20, L=250, deterministic (argmax) actions, 6000-step cap,
convergence judged offline. That judge is the `eval/` package in this repo.

At the training condition (L=250, N=20):

| arm | failures / 500 | t_conv med | J med | CVaR10 J |
|---|---|---|---|---|
| **ck848 — Dynamic-k NN** | **0** | **532** | **152.6** | **201.0** |
| π_R — binary edge, reliability | 0 | 554 | 180.9 | 269.6 |
| π_E — binary edge, efficiency | 0 | 645.5 | 260.4 | 409.8 |
| nearest k=12 (best fixed-k on J) | 32 | 521 | 160.6 | 350.6 |
| nearest k=19 = ACS FC | 0 | 591.5 | 201.0 | 542.6 |

**ck848 Pareto-dominates the entire fixed-k sweep** (k = 12–19): no arm has both
fewer failures and a lower median J, and none matches its CVaR10. Against k=12
the paired verdict is McNemar b=0 c=32, p=4.7e-10, co-success dJ median −10.4.

Off its training scale it is a **specialist**, and the binary-edge policies
remain the insurance: at L=500 ck848 fails 3.2% where π_R fails 0.2%; at L=125
nearest k=12 converges more cheaply (J 141.5 vs 198.8). Across the **N** axis at
matched density it holds up — 0/500 at both N=10 (L=177) and N=40 (L=354), where
the best fixed-k reference fails 21 and 18 times respectively.

Provenance, exact checkpoint paths and the reproduction commands:
**`checkpoints/PROVENANCE.md`** (§ *Legacy Dynamic-k NN checkpoints* for ck848).
Checkpoint binaries are **not in git**. The research record is `studies/`, a
chronological chain ending at **`studies/acs-confirm/`** (pre-registered
fresh-seed confirmation, 35/37 PASS) — start from its `REPORT_KO.md` (Korean).
Per-study `data/` directories are untracked; the summary CSVs behind the table
above live in `studies/acs-confirm/data/eval/`. Design decisions that reversed an
earlier plan are logged in `docs/DECISION_LOG.md`.

## Setup

Pinned stack — do not upgrade any of these without a coordinated bump:

| package | pin | note |
|---|---|---|
| Python | 3.9 | |
| `torch` | 1.12.1+cu113 | |
| `ray` | 2.1.0 | RLlib; the old `tune.run` API |
| `gym` | 0.23.1 | **not** Gymnasium; `env.seed()` before `reset()` |
| `pydantic` | 1.10.13 | v1 API (`@validator`, `.dict()`) |
| `numpy` | 1.23.4 | |
| `pandas` | 2.2.2 | |
| `scipy` | 1.13.1 | |
| `wandb` | 0.22.3 | logging is opt-in; see below |

- pip: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
- or Docker: see `docker/` (same pins baked in; mounts the repo at `/workspace`)

Repo-root resolution goes through `utils.paths` (`FLOCK_ROOT` overrides), so the
trainer, `eval/` and `tools/` run from any clone path. Scripts under `figures/`
and `legacy/` still hardcode `/workspace/...` — keep the repo at `/workspace`
(the docker setup does) or edit their `STUDY`/`REPO` constants.

### What a clone does *not* carry

Everything needed to **train and evaluate from scratch** is tracked; two kinds
of binary artifact are not, and are handed over out of band:

| artifact | why it is missing | what breaks without it |
|---|---|---|
| `checkpoints/**` (~69 MB) | `.gitignore` rule `checkpoints/*`; only `PROVENANCE.md` is tracked | nothing in training; you cannot re-measure the published policies, and `tools/check_ck848_parity.py` cannot run |
| `test_results/**` | gitignored run artifacts | nothing — this is where your own runs land |

**The ck848 parity reference.** `tools/check_ck848_parity.py` diffs a resolved
`dknn` config against the archived `params.json` of the ck848 run. That file is
~2.6 KB but lives inside the untracked `checkpoints/` tree, so on a fresh clone
the gate exits with `FileNotFoundError`. Restore it by unpacking the handed-over
ck848 directory at the path `PROVENANCE.md` records:

```
checkpoints/legacy_distance_pointer_260818/
  PPO_neighbor_selection_flocking_env_8d07b_00000_0_2026-08-18_06-10-46/
    params.json          <- the only file the parity gate needs
    params.pkl
    checkpoint_0008{48,56}/ checkpoint_0009{52,60,68,77}/   <- for re-measuring
```

`params.json` alone is enough for the gate; the checkpoint directories are only
needed to re-run `eval/` against the policy of record. If you keep the archive
elsewhere, point at it instead of moving it:

```bash
python train_unified.py --profile dknn --dry-run > /tmp/cfg.json
python tools/check_ck848_parity.py --config /tmp/cfg.json --ref <path>/params.json
```

### Shared multi-GPU hosts

- **A pre-set `CUDA_VISIBLE_DEVICES` wins over `--gpu`.** The trainer uses
  `setdefault`, deliberately, so a scheduler's or container's allocation is
  never overridden — but it means `export CUDA_VISIBLE_DEVICES=0` in your shell
  makes `--gpu 2` a silent no-op. Either export nothing and use `--gpu`, or
  export the allocation and drop `--gpu`. `FLOCK_GPU` is the env-var spelling of
  `--gpu` and follows the same precedence.
- **`--gpu` selects visibility, `--num-gpus` requests the learner's share.**
  `--gpu 2` with the default `--num-gpus 1` gives the trial the one GPU it can
  see. On a CPU-only host pass `--num-gpus 0`, or Tune leaves the trial PENDING
  forever instead of failing.
- **One run per GPU is the simple split**: `--gpu 0`, `--gpu 1`, … as separate
  processes. `--seeds a,b` inside one process makes one Tune *trial* per seed
  sharing whatever that process can see, which is not the same thing.
- **Docker:** `GPU_REQUEST` is passed straight to `docker run --gpus` (default
  `all`); use `GPU_REQUEST='"device=2"'` for a single card. The container
  runtime sets `CUDA_VISIBLE_DEVICES` inside, so `FLOCK_GPU` is ignored there by
  the precedence rule above — select the device with `GPU_REQUEST`. Give each
  concurrent container its own `CONTAINER_NAME` and `--run-id`.

## Layout

| path | what |
|---|---|
| `envs/env.py` | the entire simulator (`NeighborSelectionFlockingEnv`); serves both action encodings via `action_type` |
| `models/` | the two live policy models — `ppo.py` (ego-centric core, binary edge-selection head) and `ppo_dynamic_k_nn.py` (Dynamic-k cutoff-pointer subclass); dormant variants are under `legacy/` |
| `dynamic_k_nn.py` | identifiers for the Dynamic-k method (action type/encoding, model id, experiment names) |
| `train_unified.py` | **the trainer** — all three profiles (see below) |
| `eval/` | **criterion-of-record and analysis harness** — the unified `python -m eval` CLI covers checkpoint selection, staged C2, population recording, validation, and radius/heatmap/control-effort analysis; the older per-module CLIs remain available for reproduction |
| `tools/check_ck848_parity.py` | config-equivalence gate: the `dknn` profile vs the archived ck848 `params.json` |
| `baselines.py` | heuristic baselines + `create_baseline` factory (nine dormant ones re-exported from `legacy/baselines_extra.py`) |
| `callbacks.py`, `grad_logging_ppo.py` | shared RLlib callbacks / PPO subclass — pinned at the repo root: RLlib 2.1 pickles the custom policy class by module path into every checkpoint, so moving them would break existing loads |
| `test_baselines.py` | heuristic-baseline smoke/regression gate — keep it green |
| `test_dynamic_k_nn.py` | Dynamic-k test suite (action conversion, padding, cross-size strict-load, short PPO rollout, legacy binary regression) — keep it green |
| `test_c2_suite.py`, `test_checkpoint_selection.py`, `test_control_effort.py`, `test_eval_*.py` | canonical evaluation workflow regression suites — keep them green |
| `studies/` | research record; per study: `PROBLEM` / `PLAN` / `RUNLOG` / `REPORT_KO`, plus the `src/` that ran it. `c2-regime-dual-policy/` compared the two methods under one regime and `legacy-ck848/` then corrected its headline — read that pair for why Dynamic-k is the main line |
| `figures/` | paper-figure pipeline (`figures/README.md`) |
| `checkpoints/` | canonical policy copies (binaries untracked; `PROVENANCE.md`) |
| `docs/` | evaluation workflow (`EVALUATION.md`), baseline catalog, heuristic-author guide, `DECISION_LOG.md` |
| `legacy/` | retired trainers, the retired pre-C2 MC evaluator (`evaluate_checkpoint.py`), dormant model variants (`ppo_centralized.py`, `beta_dist.py`) and dormant experiment scripts + frozen era log (`legacy/HANDOFF.md`). Kept runnable: run from the repo root with `PYTHONPATH=.` |

## Training — `train_unified.py`

One trainer, three profiles:

| `--profile` | what | budget |
|---|---|---|
| **`dknn`** (default) | Dynamic-k NN under the **original ck848 recipe**: fixed 1000-step episodes, legacy shaped reward, minibatch 256, 10 SGD iters, lr 2e-5 → 1e-7 anchored at 8M | 8M steps |
| `pi_r` | binary edge selection, the confirmed π_R recipe verbatim (aux 0.3/0.05, bernoulli head, batch 16000) under the C2 training regime | 120 iters ≈ 1.92M |
| `dknn_c2` | **experimental, not the canonical line** — Dynamic-k under the C2 training regime (c2_shaping reward, C2 early termination, cap 2000, L-mix {125,250,500}); D2 probe axes exposed as flags | 2M steps |

```bash
python train_unified.py                                  # dknn, 8M steps
python train_unified.py --profile pi_r --gpu 1
python train_unified.py --profile dknn --seeds 42,1042 --gpu 1,3
python train_unified.py --profile dknn --smoke           # 2-iteration CPU smoke
python train_unified.py --profile dknn --dry-run         # resolved config, no run
python train_unified.py --profile dknn --dry-run | python tools/check_ck848_parity.py
```

- **ck848 parity.** `--dry-run` prints the resolved RLlib config as JSON;
  `tools/check_ck848_parity.py` diffs it against the archived `params.json` and
  fails on anything not on its whitelist. Run it after touching a profile. It
  needs the ck848 `params.json`, which a clone does not carry — see *What a
  clone does not carry* above.
- **Seeds.** `--seeds a,b,c` makes one Tune trial per seed; the RLlib `seed` of
  each trial follows its `env_config.seed_id`, and worker envs derive
  `seed + 10007*worker_index + 101*vector_index`.
- **In-training evaluation.** Every profile evaluates under the **same C2
  protocol** it will be judged by offline: 8 argmax episodes every 16 iterations
  (`--eval-interval`, `0` = off), c2 termination, L=250, cap 6000
  (`--eval-cap`), on dedicated workers running parallel to training. It is a
  monitoring signal — checkpoint selection is decided offline by `eval/`.
  `--cap` applies to the training env only. With evaluation disabled,
  `progress.csv` has no C2 columns for `python -m eval checkpoints` to rank;
  direct `--checkpoint` evaluation remains available.
- **Checkpoints.** Every 8 iterations, all kept, plus one at the end (848 = 8 ×
  106, so a reproduction run lands on the grid ck848 came from). ~11 MB each,
  under gitignored `test_results/`.
- **GPU.** `--gpu` sets `CUDA_VISIBLE_DEVICES`; `--num-gpus` is the RLlib
  learner request (default 1). **On a CPU-only host pass `--num-gpus 0`** —
  otherwise the trial requests a GPU and Tune leaves it PENDING instead of
  failing. `--smoke` forces both off.
- **Resume.** `--resume`, or `FLOCK_RESUME=1`, restores the same Tune trial
  (`AUTO+ERRORED`) after a process failure, container restart or host reboot.

## Evaluating

The criterion of record is the `eval/` package. Run it **from the repo root**,
as modules:

```bash
# a checkpoint on the confirmation lane
python -m eval.eval_c2 --ckpt <checkpoint_000848 dir> --label lp848 \
    --seeds 1500-1999 --workers 24
# screen a whole run by its in-training eval metrics first
python -m eval.eval_c2 --rank-runs test_results/<run>/
# fixed-k references
python -m eval.run_knn_refs --k 12,19 --L 250 --seeds 1500-1999 --workers 15
# arm-pair matrix: Wilson CI, exact McNemar, CVaR10, co-success paired dJ
python -m eval.pair_judge --seeds 1500-1999 \
    --arm pol=lp848 --arm k12=knnref:12,250,20
```

The legacy per-module commands above write to the gitignored
`test_results/{eval,knnref}/`; `--outdir` / `--base` point them at another
directory (e.g. a study's archived lane). The unified CLI described below
instead defaults to versioned bundles under `test_results/evaluation/<run-id>/`.
Method dispatch is automatic: the checkpoint's `params.json` decides between the
pointer and binary policies, and both the pre- and post-rename Dynamic-k
identifiers are accepted.

The copies under `studies/*/src/` are the **records** of the studies that
produced them — kept unmodified, not the version to run. `eval/` reproduces the
acs-confirm lane exactly (verified seed-by-seed at promotion time).

For the canonical `main_c2_v1` contract, top-5+final checkpoint funnel,
dev/confirmation lanes, full N=10/20/40 population recording, artifact
validation, and ranked-radius/heatmap/control-effort analysis, see
[`docs/EVALUATION.md`](docs/EVALUATION.md). New workflows can use the unified
CLI (`python -m eval checkpoints|c2|population|validate|radii|heatmaps|control-effort`)
directly or the foreground one-shot Docker wrapper:

```bash
CHECKPOINT_ROOT=/absolute/checkpoint/root \
ARTIFACT_ROOT=/absolute/artifact/root \
./docker/run_eval.sh --dry-run population --help
```

Extending:

- Adding a heuristic baseline: `docs/FOR_HEURISTIC_DEVELOPERS.md`, then
  `python test_baselines.py`.
- Dynamic-k test suite: `python -m unittest -v test_dynamic_k_nn`.
- All maintained regression suites, including the seven evaluation modules:
  `python -m unittest -v`.

## W&B logging (optional)

`train_unified.py` logs to W&B **only when `WANDB_ENABLED` is truthy** (default:
off). When it is on, a non-empty API key file is required or the run fails
immediately. Store the key in a private file the runner mounts read-only (never
in env vars or argv):

```bash
mkdir -p ~/.config/wandb
chmod 700 ~/.config/wandb
${EDITOR:-vi} ~/.config/wandb/api_key
chmod 600 ~/.config/wandb/api_key
```

`WANDB_API_KEY_FILE` selects the path (default `/run/secrets/wandb_api_key`),
`WANDB_PROJECT` the project (default `nb-selection-dynamic-k-nn`) and
`WANDB_RUN_NAME` the run name (default: the Tune experiment name).

## Durable background training (Docker)

Build the Python 3.9 / Ray 2.1 / CUDA 11.3 image with `./docker/build.sh`
(host needs Docker, the NVIDIA Container Toolkit, and a compatible driver).
Start a named run in a detached container:

```bash
./docker/run_train.sh start --run-id dknn-n20-seed42
./docker/run_train.sh status | logs | stop
```

The service runs `TRAIN_ENTRY` (default `train_unified.py`) with
`FLOCK_PROFILE` (default `dknn`) and exports `FLOCK_RESUME=1`, so the container's
`restart=unless-stopped` policy and Tune's `AUTO+ERRORED` together resume the
same trial from its latest checkpoint after any unexpected
process/container/host restart. After success the service writes
`.training_complete` and stays idle; a deliberate `stop` is not auto-restarted
(resume with `docker start <container>`). Results land under
`test_results/<run-id>/` on the host. Use a different `CONTAINER_NAME` for
concurrent containers. Example overrides:

```bash
FLOCK_PROFILE=pi_r FLOCK_SEEDS=7 \
CONTAINER_NAME=pir-seed7 \
./docker/run_train.sh start --run-id pir-n20-seed7
```

The profile, seeds and device are selected by `FLOCK_PROFILE` / `FLOCK_SEEDS` /
`FLOCK_GPU`, which `run_train.sh` forwards into the container. Exact-recipe
overrides are forwarded too: `FLOCK_STEPS`, `FLOCK_MINIBATCH`,
`FLOCK_SGD_ITER`, `FLOCK_LR_END`, and `FLOCK_EVAL_INTERVAL` map respectively to
`--steps`, `--minibatch`, `--sgd-iter`, `--lr-end`, and `--eval-interval`;
leaving them unset preserves the selected profile's defaults. `start` echoes
the resolved `profile=… seeds=… gpus=… wandb=…` line, so check it against what
you asked for.
**W&B is off by default here too** (same opt-in policy as `train_unified.py`);
`WANDB_ENABLED=1` turns it on and then a mode-600 key file at
`WANDB_API_KEY_FILE_HOST` is required or `start` refuses.

Setting `TRAIN_ENTRY=legacy/train_dynamic_knn.py` runs the retired env-var-driven
Dynamic-k trainer instead; note its defaults are the *accelerated* variant
(minibatch 512 / 7 SGD iters / 6M steps), not the ck848 recipe. See
`docker/train_service.sh`.

Tests inside the image:

```bash
docker run --rm --init --shm-size 4g \
  --workdir /workspace/source \
  --mount type=bind,src="$(pwd)",dst=/workspace/source,readonly \
  uom-neighbor-selection \
  python -m unittest -v
```
