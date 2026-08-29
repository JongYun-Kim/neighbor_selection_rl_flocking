"""Round-2 scale-robust training: weights-only fine-tune from a specialist ckpt
(study acs-robust-r2).

Same recipe as train_robust.py round 1 (L-mix pool {125,250,500}, legacy obs,
bernoulli head, cap 2000) EXCEPT: (a) initial policy weights are loaded from an
existing checkpoint (--init-ckpt, weights only — fresh optimizer/iteration/lr
state, NO tune restore), (b) lr is FLAT (--lr-flat, pre-registered 1e-4 for
both round-2 runs), (c) manual train loop (tune.run cannot take a pre-built
algo). UnifiedLogger keeps result.json / params.json / progress.csv in the run
dir so the round-1 monitor/eval tooling works unchanged.

Usage:
  python train_robust2.py --run-name c2F1_ft60_lmix_<date> \
      --init-ckpt <...>/checkpoint_000060 --gpu 1 [--lr-flat 1e-4] [--iters 80]
  python train_robust2.py --run-name X --init-ckpt <...> --smoke       # 2-iter CPU
  python train_robust2.py --run-name X --init-ckpt <...> --init-check  # save ckpt0 only
  python train_robust2.py --run-name X --init-ckpt <...> --gpu 1 --resume
"""
import argparse
import glob
import json
import os
import re

# GPU must be pinned BEFORE ray/torch import; parse argv manually here.
_ap = argparse.ArgumentParser()
_ap.add_argument("--run-name", required=True)
_ap.add_argument("--init-ckpt", required=True,
                 help="checkpoint_0000NN dir holding policies/default_policy/policy_state.pkl")
_ap.add_argument("--gpu", default=os.environ.get("FLOCK_GPU"),
                 help="CUDA device index; required for full runs (or set FLOCK_GPU); "
                      "ignored for --smoke/--init-check (CPU)")
_ap.add_argument("--variant", choices=["legacy", "r0log"], default="legacy",
                 help="obs_position_scale (round 2 uses legacy for both runs)")
_ap.add_argument("--lr-flat", type=float, default=1e-4,
                 help="flat learning rate, no schedule (pre-registered 1e-4)")
_ap.add_argument("--iters", type=int, default=80)
_ap.add_argument("--smoke", action="store_true", help="2-iter CPU smoke test")
_ap.add_argument("--init-check", action="store_true",
                 help="build, load init weights, save checkpoint_000000, exit (CPU)")
_ap.add_argument("--resume", action="store_true",
                 help="resume from this run's last checkpoint (full restore incl. iter)")
ARGS = _ap.parse_args()

CPU_ONLY = ARGS.smoke or ARGS.init_check
if not CPU_ONLY:
    assert ARGS.gpu is not None, "--gpu <index> (or FLOCK_GPU) required for a full run"
if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ARGS.gpu)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from utils.paths import repo_path  # noqa: E402

import numpy as np  # noqa: E402
import ray  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.tune.logger import UnifiedLogger  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from envs.env import NeighborSelectionFlockingEnv, load_config  # noqa: E402
from models.ppo import NeighborSelectionPPORLlib  # noqa: E402
from callbacks import C2Callbacks  # noqa: E402
from grad_logging_ppo import GradLoggingPPO  # noqa: E402

ENV_NAME = "neighbor_selection_flocking_env"
MODEL_NAME = "neighbor_selector_rl"
BASE_SEED = 42          # identical to round-1 R1: init ckpt stays the ONLY variable
EVAL_SEED = 900000
L_POOL = [125.0, 250.0, 500.0]
LOGDIR = os.path.join(repo_path("test_results"), ARGS.run_name, "manual")


def build_env_config(is_training: bool) -> dict:
    # Absolute path: background launches must not depend on the shell cwd.
    my_config = load_config(repo_path("envs", "default_env_config.yaml"))

    # environment configs (A-line C2 block, identical to round-1 train_robust.py)
    my_config.env.action_type = "binary_vector"
    my_config.env.comm_range = None
    my_config.env.dt = 0.1
    my_config.env.env_mode = "single_env"
    my_config.env.is_training = is_training
    my_config.env.num_agents_pool = [20]
    my_config.env.obs_dim = 4
    my_config.env.observation_type = "ego_centric"
    my_config.env.periodic_boundary = False
    my_config.env.task_type = "acs"
    my_config.env.use_fixed_episode_length = False
    my_config.env.use_rotated_ego_obs = True
    my_config.env.continuous_action = False
    my_config.env.expose_aux_target = True
    my_config.env.termination_mode = "c2"
    my_config.env.reward_mode = "c2_shaping"          # inert when is_training=False
    my_config.env.c2_phi_goal = 0.98
    my_config.env.c2_align_window = 50
    my_config.env.c2_window = 300
    my_config.env.c2_eps = 0.05
    my_config.env.c2_w_pos = 4.0
    my_config.env.c2_w_vel = 0.2
    my_config.env.c2_w_ctrl = 0.1
    my_config.env.c2_success_bonus = 10.0
    my_config.env.expose_global_stats = True
    my_config.env.max_time_steps = 2000
    my_config.env.initial_position_bound_pool = L_POOL if is_training else None
    my_config.env.obs_position_scale = "r0_log" if ARGS.variant == "r0log" else "legacy"

    # control config (repo defaults, pinned explicitly)
    my_config.control.beta = 1 / 3
    my_config.control.initial_position_bound = 250.0
    my_config.control.k1 = 1.0
    my_config.control.k2 = 3.0
    my_config.control.lam = 5.0
    my_config.control.max_turn_rate = 8 / 15
    my_config.control.r0 = 60.0
    my_config.control.rho = 1.0
    my_config.control.sig = 1.0
    my_config.control.speed = 15.0

    return my_config.dict()


def make_env(env_context):
    """Per-env RNG de-duplication (worker/vector offsets), as in train_c2_*."""
    ctx = dict(env_context)
    seed = ctx.get("seed_id")
    if seed is not None:
        wi = getattr(env_context, "worker_index", 0)
        vi = getattr(env_context, "vector_index", 0)
        ctx["seed_id"] = seed + 10007 * wi + 101 * vi
    return NeighborSelectionFlockingEnv(ctx)


# Bernoulli head, NO distance prior (proven A recipe; identical to round 1).
CUSTOM_MODEL_CONFIG = {
    "d_embed_context": 128,
    "d_embed_input": 128,
    "d_ff": 256,
    "d_ff_decoder": 256,
    "d_model": 128,
    "d_model_decoder": 128,
    "d_subobs": 4,
    "dr_rate": 0,
    "is_bias": False,
    "n_layers_decoder": 1,
    "n_layers_encoder": 3,
    "norm_eps": 1e-05,
    "num_heads": 4,
    "scale_factor": 0.10,
    "share_layers": False,
    "use_FNN_in_decoder": True,
    "use_residual_in_decoder": True,
    "aux_enabled": True,
    "aux_type": "pair_embedding",
    "aux_loss_coef": 0.3,
    "aux_target_dim": 4,
    "aux_loss_coef_critic": 0.05,
    "continuous_action": False,
    "per_agent_credit": False,
    "selection_head": "bernoulli",
    "top_k": None,
    "hard_top_k": False,
    "dist_aux_coef": 0.0,
    "use_global_stats": True,
}


def build_algo_config(smoke: bool) -> dict:
    config = {
        "env": ENV_NAME,
        "env_config": {"seed_id": BASE_SEED, "config": build_env_config(is_training=True)},
        "framework": "torch",
        "callbacks": C2Callbacks,
        "model": {
            "custom_model": MODEL_NAME,
            "custom_model_config": CUSTOM_MODEL_CONFIG,
        },
        # --- Resources ---
        "num_gpus": 0 if CPU_ONLY else 1,
        "num_workers": 1 if smoke else 4,
        "num_envs_per_worker": 2 if smoke else 4,
        # --- Rollout / batch ---
        "rollout_fragment_length": 500 if smoke else 1000,
        "train_batch_size": 1000 if smoke else 16000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 2 if smoke else 10,
        # --- Learning rate: FLAT (round-2 delta; protects the specialist init) ---
        "lr": ARGS.lr_flat,
        "lr_schedule": None,
        # --- PPO (identical to round 1) ---
        "vf_loss_coeff": 0.5,
        "use_critic": True,
        "use_gae": True,
        "gamma": 0.99,
        "lambda": 0.95,
        "kl_coeff": 0,
        "clip_param": 0.15,
        "vf_clip_param": 256,
        "grad_clip": 1.0,
        "kl_target": 0.01,
        "entropy_coeff": 1e-3,
        "normalize_actions": False,
        # --- Evaluation: deterministic, fixed L=250, raw-cost reward ---
        "evaluation_interval": None if smoke else 10,
        "evaluation_duration": 16,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 0 if smoke else 2,
        "evaluation_config": {
            "explore": False,
            "env_config": {"seed_id": EVAL_SEED, "config": build_env_config(is_training=False)},
        },
    }
    return config


SMOKE_KEYS = [
    ("episode_reward_mean", "sampler_results/episode_reward_mean"),
    ("episode_len_mean", "sampler_results/episode_len_mean"),
    ("gnorm_actor", "info/learner/default_policy/learner_stats/gnorm_actor_preclip"),
    ("entropy", "info/learner/default_policy/learner_stats/entropy"),
    ("sat_p_dev", "info/learner/default_policy/learner_stats/sat_p_dev"),
    ("aux_mse", "info/learner/default_policy/learner_stats/aux_mse"),
]


def dig(d, path):
    cur = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_init_weights(algo):
    """Weights-only init from ARGS.init_ckpt (fresh optimizer/lr/iteration)."""
    import pickle
    p = os.path.join(ARGS.init_ckpt, "policies", "default_policy", "policy_state.pkl")
    with open(p, "rb") as f:
        state = pickle.load(f)
    weights = state["weights"]
    algo.get_policy().set_weights(weights)
    # Verify the assignment actually landed (compare one large tensor).
    import torch
    got = algo.get_policy().get_weights()
    key = max(weights, key=lambda k: np.asarray(weights[k]).size)
    assert np.allclose(np.asarray(got[key]), np.asarray(weights[key]), atol=1e-6), \
        f"set_weights verification failed on {key}"
    algo.workers.sync_weights()
    if getattr(algo, "evaluation_workers", None) is not None:
        algo.evaluation_workers.sync_weights(from_worker=algo.workers.local_worker())
    n_par = sum(np.asarray(v).size for v in weights.values())
    print(f"[init] loaded {len(weights)} tensors ({n_par:,} params) from {ARGS.init_ckpt}",
          flush=True)


def compact_line(result):
    cm = result.get("custom_metrics") or {}
    ev = (result.get("evaluation") or {}).get("custom_metrics") or {}
    ls = dig(result, "info/learner/default_policy/learner_stats") or {}

    def r(x, n=1):
        return "-" if x is None else round(float(x), n)
    s = (f"it {result['training_iteration']:>3} "
         f"len {r(result.get('episode_len_mean'), 0)} "
         f"succ {r(cm.get('c2_success_mean'), 2)} J {r(cm.get('J_success_mean'), 0)} "
         f"gnorm {r(ls.get('gnorm_actor_preclip'), 2)} ent {r(ls.get('entropy'), 1)} "
         f"lr {ls.get('cur_lr')} {r(result.get('time_this_iter_s'), 0)}s")
    if ev.get("c2_success_mean") is not None:
        s += f" | EVAL succ {r(ev.get('c2_success_mean'), 2)} J {r(ev.get('J_success_mean'), 0)}"
    return s


def main():
    ray.init(num_cpus=8 if CPU_ONLY else 16)
    register_env(ENV_NAME, make_env)
    ModelCatalog.register_custom_model(MODEL_NAME, NeighborSelectionPPORLlib)

    # init-check shares the light smoke resource block (weights are all that
    # matter for the saved ckpt; the full run rewrites params.json anyway).
    config = build_algo_config(ARGS.smoke or ARGS.init_check)
    os.makedirs(LOGDIR, exist_ok=True)

    def logger_creator(cfg):
        return UnifiedLogger(cfg, LOGDIR)

    algo = GradLoggingPPO(config=config, logger_creator=logger_creator)

    if ARGS.smoke:
        load_init_weights(algo)
        for i in range(2):
            result = algo.train()
            print(f"--- smoke iter {i + 1} [{ARGS.run_name}] ---")
            for label, path in SMOKE_KEYS:
                v = dig(result, path)
                print(f"  {label:20s} = {v if v is None else round(float(v), 5)}")
        algo.stop()
        return

    if ARGS.init_check:
        load_init_weights(algo)
        path = algo.save(LOGDIR)
        print(f"[init-check] checkpoint saved -> {path}", flush=True)
        algo.stop()
        return

    # ---- full run ----
    if ARGS.resume:
        ckpts = sorted(glob.glob(os.path.join(LOGDIR, "checkpoint_*")),
                       key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
        assert ckpts, f"--resume but no checkpoints under {LOGDIR}"
        algo.restore(ckpts[-1])
        print(f"[resume] restored {ckpts[-1]} (iteration {algo.iteration})", flush=True)
    else:
        load_init_weights(algo)
        path = algo.save(LOGDIR)  # checkpoint_000000 = the verified init state
        print(f"[init] checkpoint_000000 saved -> {path}", flush=True)

    while algo.iteration < ARGS.iters:
        result = algo.train()
        print(compact_line(result), flush=True)
        if algo.iteration % 10 == 0 or algo.iteration >= ARGS.iters:
            path = algo.save(LOGDIR)
            print(f"[ckpt] -> {path}", flush=True)

    algo.stop()
    print(f"[done] {ARGS.run_name}: {ARGS.iters} iters complete", flush=True)


if __name__ == "__main__":
    main()
