# Neighbor-Selection RL for Flocking — Binary Edge Selection × Dynamic-k NN

RL for **which neighbors to listen to** in a flocking swarm. Two policy families
share one simulator (`NeighborSelectionFlockingEnv`) and one Transformer/pointer
core; the low-level ACS/Vicsek controller inside the env turns the
selected-neighbor subgraph into velocity updates (the policy never outputs motion
commands directly):

- **`policy` — binary edge selection (main line)**: PPO outputs a binary adjacency
  matrix per step (`action_type="binary_vector"`, model `NeighborSelectionPPORLlib`).
- **`dynamic_knn` — cutoff-pointer dynamic k-NN**: for every ego agent the policy
  points to one active cutoff agent; that agent and all active agents no farther
  away become directed neighbors, so the number of external neighbors `k` changes
  with the observation and policy output. Selecting the ego itself means `k=0`;
  equal-distance ties are included (`action_type="dynamic_k_nn"`, model
  `DynamicKNNPPORLlib` — same encoder/pointer core, ego-wise N-way categorical head).

## Current results (2026-08, binary-edge line)

The confirmed research line lives in `studies/` — a chronological chain ending at
**`studies/acs-confirm/`** (pre-registered fresh-seed confirmation, 35/37 PASS).
Start with `studies/acs-confirm/REPORT_KO.md` (Korean). Policies of record:

- **π_E** — efficiency policy (`c2C1` fine-tune, it80)
- **π_R** — reliability/"insurance" policy (`c2R1` scratch L-mix, it110)

Checkpoint binaries are **not in git**: see `checkpoints/PROVENANCE.md` for exact
paths, provenance, and reproduction commands (copies live on the lab machine).

The dynamic-k line has no committed checkpoints yet; it is retrained under the
unified C2 regime on the integration branch (paired comparison in progress).

## Setup

Pinned stack — do not upgrade any of these without a coordinated bump (details in
`CLAUDE.md`): Python 3.9, `torch==1.12.1+cu113`, `ray==2.1.0` (RLlib),
`gym==0.23.1` (not Gymnasium), `pydantic==1.10.13` (v1 API), `numpy==1.23.4`,
`wandb==0.22.3` (runtime requirement of the dynamic-k trainer; logging itself is
opt-in).

- pip: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113`
- or Docker: see `docker/` (same pins baked in; mounts the repo at `/workspace`)

Many studies/figures scripts hardcode `/workspace/...` paths — keep the repo at
`/workspace` (the docker setup does) or edit their `STUDY`/`REPO` constants.

## Layout

| path | what |
|---|---|
| `envs/env.py` | the entire simulator (`NeighborSelectionFlockingEnv`); serves both action encodings via `action_type` |
| `models/` | PPO models — `ppo.py` (ego-centric core, current line), `ppo_dynamic_k_nn.py` (dynamic-k pointer subclass), `ppo_centralized.py` (legacy variant) |
| `dynamic_k_nn/` | identifiers for the dynamic-k method (action type/encoding, model id, experiment names) |
| `baselines.py` | heuristic baselines + `create_baseline` factory |
| `callbacks.py`, `grad_logging_ppo.py` | shared RLlib callbacks / PPO subclass used by every trainer |
| `train*.py` | current-line trainers (see below) |
| `evaluate_checkpoint.py` | MC eval harness (centralized-variant checkpoints) |
| `test_baselines.py` | repo-wide smoke/regression gate — keep it green |
| `test_dynamic_k_nn.py` | dynamic-k test suite (action conversion, padding, cross-size strict-load, short PPO rollout, legacy binary regression) — keep it green |
| `studies/` | research record; per study: `PROBLEM` / `PLAN` / `RUNLOG` / `REPORT_KO` |
| `figures/` | paper-figure pipeline (`figures/README.md`) |
| `checkpoints/` | canonical policy copies (untracked; `PROVENANCE.md`) |
| `docs/` | baseline catalog + guide for adding a heuristic |
| `legacy/` | dormant Jan–May 2026 experiment scripts + frozen era log (`legacy/HANDOFF.md`) |

Root trainers: `train.py` (documented baseline entry point for the binary-edge
line; not the winning recipe), `train_c2_a.py` / `train_c2_b.py` (C2 arms A/B),
`train_robust.py` (→ π_R), `train_robust2.py` (weights-only fine-tune, → π_E),
`train_hardtopk.py` (Phase-14 ancestor of the C2 line),
`train_dynamic_knn.py` (dynamic-k original trainer, env-var configured — the
6M-step fixed-length recipe validated on an i9-9900KF + RTX 3090: 8 workers × 2
envs, batch 8192, lr `2e-5`→`1e-7`; every value overridable via the environment
variables listed by `./docker/run_train.sh --help`).

## Evaluating / extending

- Criterion-of-record evaluation: `studies/acs-confirm/src/` (`eval_c2_r3.py`,
  `run_knn_refs3.py`, `confirm_judge.py`). Earlier studies carry older copies of
  `eval_c2.py` — **always use the acs-confirm copies**.
- Adding a heuristic baseline: `docs/FOR_HEURISTIC_DEVELOPERS.md`, then run
  `python test_baselines.py`.
- Dynamic-k test suite: `python -m unittest -v test_dynamic_k_nn` (also runnable
  inside the Docker image, see below).

## W&B logging (optional)

`wandb==0.22.3` is pinned in `requirements.txt`. The dynamic-k trainer logs to
W&B **when `WANDB_ENABLED` is truthy** (its own default is on; set
`WANDB_ENABLED=false` to train offline). Store the API key in a private file the
runner mounts read-only (never in env vars or argv):

```bash
mkdir -p ~/.config/wandb
chmod 700 ~/.config/wandb
${EDITOR:-vi} ~/.config/wandb/api_key
chmod 600 ~/.config/wandb/api_key
```

The default project is `nb-selection-dynamic-k-nn`; override with `WANDB_PROJECT`.

## Durable background training (Docker)

Build the Python 3.9 / Ray 2.1 / CUDA 11.3 image with `./docker/build.sh`
(host needs Docker, the NVIDIA Container Toolkit, and a compatible driver).
Start a named run in a detached container:

```bash
./docker/run_train.sh start --run-id dynamic-k-n20-seed42
./docker/run_train.sh status | logs | stop
```

The service entry point is selectable via `TRAIN_ENTRY` (see
`docker/train_service.sh`). Results land under `test_results/<run-id>/` on the
host. The container uses `restart=unless-stopped` and Ray Tune uses
`AUTO+ERRORED`, so an unexpected process/container/host restart resumes from the
latest checkpoint; after success the service writes `.training_complete` and
stays idle. A deliberate `stop` is not auto-restarted (resume with
`docker start <container>`). Use a different `CONTAINER_NAME` for concurrent
containers. Example overrides:

```bash
WANDB_PROJECT=my-project BASE_ENV_SEED=7 \
TOTAL_TRAINING_TIMESTEPS=1000000 \
CONTAINER_NAME=dynamic-k-seed7 \
./docker/run_train.sh start --run-id dynamic-k-n20-seed7
```

Tests inside the image:

```bash
docker run --rm --init --shm-size 4g \
  --workdir /workspace/source \
  --env START_SSHD=0 \
  --mount type=bind,src="$(pwd)",dst=/workspace/source,readonly \
  uom-neighbor-selection \
  python -m unittest -v test_dynamic_k_nn
```
