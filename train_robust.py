"""Scale-robust neighbor-selection training (study acs-robust-train).

Single policy trained across initial scales L ~ U{125, 250, 500} under the C2
convergence criterion, bernoulli head (the proven A-line recipe). Two variants
(single-variable ablation):
  --variant legacy : L-mixed episodes, legacy per-episode L/2 obs normalization
                     -> run c2R1_lmix_legacy_260807 on cuda:1
  --variant r0log  : L-mixed episodes + scale-free obs d->unit(d)*log1p(|d|/r0)
                     -> run c2R2_lmix_r0log_260807 on cuda:3

Usage:
    python train_robust.py --variant legacy            # full run (120 iters)
    python train_robust.py --variant r0log --smoke     # 2-iter CPU smoke
    python train_robust.py --variant legacy --resume   # resume after a crash
"""
import argparse
import os
import sys

# GPU must be pinned BEFORE ray/torch import; parse argv manually here.
_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", choices=["legacy", "r0log"], required=True)
_ap.add_argument("--smoke", action="store_true", help="2-iter CPU smoke test")
_ap.add_argument("--resume", action="store_true",
                 help="resume this variant's run from its last checkpoint")
ARGS = _ap.parse_args()

VARIANT = ARGS.variant
# Historical per-variant GPU pin, overridable: pre-set CUDA_VISIBLE_DEVICES
# wins, then FLOCK_GPU, then the legacy default below.
GPU = {"legacy": "1", "r0log": "3"}[VARIANT]
RUN_NAME = {"legacy": "c2R1_lmix_legacy_260807",
            "r0log": "c2R2_lmix_r0log_260807"}[VARIANT]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("FLOCK_GPU", GPU))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from utils.paths import repo_path  # noqa: E402

import ray  # noqa: E402
from ray import tune  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from envs.env import NeighborSelectionFlockingEnv, load_config  # noqa: E402
from models.ppo import NeighborSelectionPPORLlib  # noqa: E402
from callbacks import C2Callbacks  # noqa: E402
from grad_logging_ppo import GradLoggingPPO  # noqa: E402

ENV_NAME = "neighbor_selection_flocking_env"
MODEL_NAME = "neighbor_selector_rl"
BASE_SEED = 42
EVAL_SEED = 900000
L_POOL = [125.0, 250.0, 500.0]


def build_env_config(is_training: bool) -> dict:
    # Absolute path: background launches must not depend on the shell cwd.
    my_config = load_config(repo_path("envs", "default_env_config.yaml"))

    # environment configs (A-line C2 block; see acs-c2-train PLAN Phase 3)
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
    # --- scale-robustness deltas (this study) ---
    # Training: episode L ~ U{125,250,500}; cap 2000 (k-NN(12) fires ~530-590
    # at L=500, gate-verified). Eval: FIXED L=250 for continuity with the
    # A-line online eval traces (offline grid eval is the arbiter).
    my_config.env.max_time_steps = 2000
    my_config.env.initial_position_bound_pool = L_POOL if is_training else None
    my_config.env.obs_position_scale = "r0_log" if VARIANT == "r0log" else "legacy"

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


# Bernoulli head, NO distance prior (proven A recipe).
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


def build_tune_config(smoke: bool) -> dict:
    config = {
        "env": ENV_NAME,
        "env_config": {"seed_id": BASE_SEED, "config": build_env_config(is_training=True)},
        "framework": "torch",
        "callbacks": C2Callbacks,
        "model": {
            "custom_model": MODEL_NAME,
            "custom_model_config": CUSTOM_MODEL_CONFIG,
        },
        # --- Resources (per trial) ---
        "num_gpus": 0 if smoke else 1,
        "num_workers": 1 if smoke else 4,
        "num_envs_per_worker": 2 if smoke else 4,
        # --- Rollout / batch ---
        "rollout_fragment_length": 500 if smoke else 1000,
        "train_batch_size": 1000 if smoke else 16000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 2 if smoke else 10,
        # --- Learning rate ---
        "lr": 5e-4,
        "lr_schedule": [[0, 5e-4], [800000, 1e-4]],
        # --- PPO ---
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
        # Flat 1e-3 (the A recipe). The A2 anneal was proven a no-op for the
        # bernoulli head (entropy gradient vanishes at p~0.5) — not repeated.
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


def main():
    ray.init(num_cpus=8 if ARGS.smoke else 16)
    register_env(ENV_NAME, make_env)
    ModelCatalog.register_custom_model(MODEL_NAME, NeighborSelectionPPORLlib)

    config = build_tune_config(ARGS.smoke)

    if ARGS.smoke:
        algo = GradLoggingPPO(config=config)
        for i in range(2):
            result = algo.train()
            print(f"--- smoke iter {i + 1} [{VARIANT}] ---")
            for label, path in SMOKE_KEYS:
                v = dig(result, path)
                print(f"  {label:20s} = {v if v is None else round(float(v), 5)}")
        algo.stop()
        return

    tune.run(
        GradLoggingPPO,
        name=RUN_NAME,
        local_dir=repo_path("test_results"),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"training_iteration": 120},
        config=config,
        resume="AUTO" if ARGS.resume else False,
    )


if __name__ == "__main__":
    main()
