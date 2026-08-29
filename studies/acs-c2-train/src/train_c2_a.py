"""Variant A ("bernoulli") training under the C2 convergence criterion.

Study: studies/acs-c2-train (see PROBLEM.md/PLAN.md there).
Plain per-edge binary selection (the legacy no-top-K logits path), NO distance
prior (dist_aux off). Env: C2 success-only early termination + c2_shaping
reward + global_stats obs. GPU: cuda:1 (user-assigned).

Usage:
    CUDA_VISIBLE_DEVICES is forced to "1" below (before ray/torch import).
    python train_c2_a.py            # full run (100 iters, GPU)
    python train_c2_a.py --smoke    # 2-iter CPU smoke test (Phase 2 gate)
"""
import argparse
import os

# Pin to GPU 1 for this experiment (variant B uses GPU 3). Must be set BEFORE
# importing ray/torch. Keep worker CPU threads at 1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("FLOCK_GPU", "1"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from utils.paths import repo_path

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, load_config
from models.ppo import NeighborSelectionPPORLlib
from callbacks import C2Callbacks
from grad_logging_ppo import GradLoggingPPO

RUN_NAME = "c2A2_bernoulli_entsched_260806"
ENV_NAME = "neighbor_selection_flocking_env"
MODEL_NAME = "neighbor_selector_rl"
BASE_SEED = 42
EVAL_SEED = 900000


def build_env_config(is_training: bool) -> dict:
    my_config = load_config(repo_path("envs", "default_env_config.yaml"))

    # environment configs (shared A/B; see PLAN Phase 3)
    my_config.env.action_type = "binary_vector"
    my_config.env.comm_range = None
    my_config.env.dt = 0.1
    my_config.env.env_mode = "single_env"
    my_config.env.is_training = is_training
    my_config.env.max_time_steps = 1500
    my_config.env.num_agents_pool = [20]
    my_config.env.obs_dim = 4
    my_config.env.observation_type = "ego_centric"
    my_config.env.periodic_boundary = False
    my_config.env.task_type = "acs"
    my_config.env.use_fixed_episode_length = False
    my_config.env.use_rotated_ego_obs = True
    my_config.env.continuous_action = False
    my_config.env.expose_aux_target = True
    # C2 criterion + shaping (env fields added by study acs-c2-train)
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
    """De-duplicate per-env RNG streams: the legacy scripts seeded every env 42,
    giving all parallel envs identical episode sequences. Offset by worker and
    vector index instead (eval workers use a different base seed)."""
    ctx = dict(env_context)
    seed = ctx.get("seed_id")
    if seed is not None:
        wi = getattr(env_context, "worker_index", 0)
        vi = getattr(env_context, "vector_index", 0)
        ctx["seed_id"] = seed + 10007 * wi + 101 * vi
    return NeighborSelectionFlockingEnv(ctx)


# Variant A model: plain bernoulli edges, NO distance prior.
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
    # --- variant A head ---
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
        "entropy_coeff": 1e-3,
        # A2: anneal the entropy bonus (same knob as B2/B3). Run
        # c2A_bernoulli_260806 (flat 1e-3) reached eval J 145-157 mid-training
        # but drifted to J 171-190 as success solidified at 1.00 — sampling
        # stayed near-uniform (entropy ~262) so training-time success pressure
        # never acted on the argmax structure directly. Annealing aligns
        # late-training sampling with the deterministic readout.
        "entropy_coeff_schedule": [[0, 1e-3], [500000, 1e-4]],
        "normalize_actions": False,
        # --- Evaluation: deterministic, raw-cost reward (is_training=False) ---
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
    ("dist_aux", "info/learner/default_policy/learner_stats/dist_aux"),
    ("dist_aux_coef", "info/learner/default_policy/learner_stats/dist_aux_coef_current"),
]


def dig(d, path):
    cur = d
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-iter CPU smoke test")
    ap.add_argument("--resume", action="store_true",
                    help="resume the run of RUN_NAME from its last checkpoint "
                         "(tune resume='AUTO'; use if the process died mid-run)")
    args = ap.parse_args()

    ray.init(num_cpus=8 if args.smoke else 16)
    register_env(ENV_NAME, make_env)
    ModelCatalog.register_custom_model(MODEL_NAME, NeighborSelectionPPORLlib)

    config = build_tune_config(args.smoke)

    if args.smoke:
        algo = GradLoggingPPO(config=config)
        for i in range(2):
            result = algo.train()
            print(f"--- smoke iter {i + 1} ---")
            for label, path in SMOKE_KEYS:
                v = dig(result, path)
                print(f"  {label:16s} = {v if v is None else round(float(v), 5)}")
        algo.stop()
        return

    tune.run(
        GradLoggingPPO,
        name=RUN_NAME,
        local_dir=repo_path("test_results"),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"training_iteration": 100},
        config=config,
        resume="AUTO" if args.resume else False,
    )


if __name__ == "__main__":
    main()
