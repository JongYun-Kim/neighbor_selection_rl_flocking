"""Unified dual-method trainer (integration/dual-policy line).

One entry point trains either neighbor-selection method under the shared C2
regime (INTEGRATION_PLAN Q2/Q4; profiles below), with the D6 seed scheme,
deterministic in-training evaluation, C2Callbacks metrics, and W&B off by
default (Q10: set WANDB_ENABLED=1 to opt in).

Profiles
  policy_robust  binary_vector + NeighborSelectionPPORLlib — the confirmed
                 pi_R recipe of train_robust.py --variant legacy, verbatim
                 (aux 0.3/0.05, bernoulli head, lr 5e-4->1e-4@800k,
                 batch 16000, 120 iters ~= 1.92M steps).
  dknn_c2        dynamic_k_nn + DynamicKNNPPORLlib under the SAME C2 regime
                 (c2_shaping reward, C2 early termination, cap 2000, N=20,
                 L-mix {125,250,500}). HP start from the dknn original recipe;
                 the D2 probe axes are exposed as CLI flags.
  dknn_legacy    the dknn original regime (fixed 1000-step episodes, legacy
                 shaped reward, 6M-step schedule) — DEFINED BUT NOT RUN in the
                 integration study (plan Q3/Q4 rider); requires --allow-legacy.

Usage
  python train_unified.py --profile policy_robust --gpu 1
  python train_unified.py --profile dknn_c2 --gpu 1 --steps 500000
  python train_unified.py --profile dknn_c2 --seeds 42,1042 --gpu 1,3
  python train_unified.py --profile dknn_c2 --smoke        # short CPU smoke

Seeds: --seeds a,b,c makes one Tune trial per seed (env seed_id grid; worker
envs derive seed + 10007*worker_index + 101*vector_index, main-style D6).
Existing single-method trainers (train_robust*.py, train_dynamic_knn.py, ...)
are preserved unchanged.
"""
import argparse
import json
import os

_ap = argparse.ArgumentParser()
_ap.add_argument("--profile", default=os.environ.get("FLOCK_PROFILE", "dknn_c2"),
                 choices=["policy_robust", "dknn_c2", "dknn_legacy"])
_ap.add_argument("--seeds", default=os.environ.get("FLOCK_SEEDS", "42"),
                 help="comma-separated training seeds; one Tune trial per seed")
_ap.add_argument("--gpu", default=os.environ.get("FLOCK_GPU"),
                 help="CUDA_VISIBLE_DEVICES value (e.g. '1' or '1,3'). Unset: "
                      "leave device visibility untouched (container-friendly).")
_ap.add_argument("--steps", type=int, default=None,
                 help="stop at this many env steps (default: profile budget)")
_ap.add_argument("--iters", type=int, default=None,
                 help="ALSO stop at this training iteration (policy_robust "
                      "default: 120, the recipe's grid)")
_ap.add_argument("--run-name", default=None)
_ap.add_argument("--smoke", action="store_true", help="short CPU smoke (2 iters)")
_ap.add_argument("--resume", action="store_true")
_ap.add_argument("--allow-legacy", action="store_true",
                 help="required to actually run the dknn_legacy profile")
# --- D2 probe axes (dknn_c2 tuning) + general knobs; None = profile default ---
_ap.add_argument("--lr", type=float, default=None)
_ap.add_argument("--lr-end", type=float, default=None,
                 help="linear lr endpoint at --steps (omit: flat/profile schedule)")
_ap.add_argument("--grad-clip", type=float, default=None)
_ap.add_argument("--entropy-coeff", type=float, default=None)
_ap.add_argument("--clip-param", type=float, default=None)
_ap.add_argument("--entropy-penalty", type=float, default=None,
                 help="dknn only: pointer-entropy penalty coef applied in "
                      "custom_loss (RLlib forbids entropy_coeff < 0)")
_ap.add_argument("--minibatch", type=int, default=None)
_ap.add_argument("--sgd-iter", type=int, default=None)
_ap.add_argument("--workers", type=int, default=None)
_ap.add_argument("--envs-per-worker", type=int, default=None)
_ap.add_argument("--fragment", type=int, default=None)
_ap.add_argument("--cap", type=int, default=None, help="max_time_steps override")
_ap.add_argument("--eval-interval", type=int, default=10)
_ap.add_argument("--num-cpus", type=int, default=None,
                 help="ray.init num_cpus (default: workers + eval workers + 2)")
_ap.add_argument("--object-store-gb", type=float, default=8.0)
ARGS = _ap.parse_args()

# Device pinning must precede ray/torch imports. Pre-set CUDA_VISIBLE_DEVICES
# wins; --gpu/FLOCK_GPU next; otherwise visibility is left untouched.
if ARGS.smoke:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif ARGS.gpu is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ARGS.gpu)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ray  # noqa: E402
from ray import tune  # noqa: E402
from ray.rllib.models import ModelCatalog  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from envs.env import NeighborSelectionFlockingEnv, load_config  # noqa: E402
from models.ppo import NeighborSelectionPPORLlib  # noqa: E402
from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib  # noqa: E402
from dynamic_k_nn.identifiers import ACTION_TYPE, MODEL_ID  # noqa: E402
from callbacks import C2Callbacks  # noqa: E402
from grad_logging_ppo import GradLoggingPPO  # noqa: E402
from utils.paths import repo_path  # noqa: E402

ENV_NAME = "neighbor_selection_flocking_env"
EVAL_SEED = 900000                    # D6
L_POOL = [125.0, 250.0, 500.0]        # D4


def build_env_config(is_training, action_type, regime):
    """Shared env builder. regime: 'c2' (unified study) or 'legacy_dknn'."""
    cfg = load_config(repo_path("envs", "default_env_config.yaml"))
    e = cfg.env
    e.action_type = action_type
    e.comm_range = None
    e.dt = 0.1
    e.env_mode = "single_env"
    e.is_training = is_training
    e.num_agents_pool = [20]
    e.obs_dim = 4
    e.observation_type = "ego_centric"
    e.periodic_boundary = False
    e.task_type = "acs"
    e.use_rotated_ego_obs = True
    e.continuous_action = False

    if regime == "c2":
        # A-line C2 block, identical to train_robust.py (pi_R recipe).
        e.use_fixed_episode_length = False
        e.termination_mode = "c2"
        e.reward_mode = "c2_shaping"          # inert when is_training=False
        e.c2_phi_goal = 0.98
        e.c2_align_window = 50
        e.c2_window = 300
        e.c2_eps = 0.05
        e.c2_w_pos = 4.0
        e.c2_w_vel = 0.2
        e.c2_w_ctrl = 0.1
        e.c2_success_bonus = 10.0
        e.max_time_steps = 2000               # D3
        e.initial_position_bound_pool = L_POOL if is_training else None  # D4
        e.obs_position_scale = "legacy"
        # Method-side exposure flags are set by the profile (aux/global_stats
        # are policy-method components, Q6).
    elif regime == "legacy_dknn":
        e.use_fixed_episode_length = True
        e.max_time_steps = 1000
        e.termination_mode = "legacy"
        e.reward_mode = "legacy"
        e.acs_train_w_ctrl = 0.02
        e.acs_train_w_pos = 1.0
        e.acs_train_w_vel = 0.2
    else:
        raise ValueError(f"unknown regime: {regime}")

    c = cfg.control
    c.beta = 1 / 3
    c.initial_position_bound = 250.0
    c.k1 = 1.0
    c.k2 = 3.0
    c.lam = 5.0
    c.max_turn_rate = 8 / 15
    c.r0 = 60.0
    c.rho = 1.0
    c.sig = 1.0
    c.speed = 15.0
    return cfg


def make_env(env_context):
    """D6 worker-seed derivation (main-style), as in train_robust/train_c2_*."""
    ctx = dict(env_context)
    seed = ctx.get("seed_id")
    if seed is not None:
        wi = getattr(env_context, "worker_index", 0)
        vi = getattr(env_context, "vector_index", 0)
        ctx["seed_id"] = seed + 10007 * wi + 101 * vi
    return NeighborSelectionFlockingEnv(ctx)


_SHARED_MODEL = {
    "d_embed_context": 128, "d_embed_input": 128, "d_ff": 256, "d_ff_decoder": 256,
    "d_model": 128, "d_model_decoder": 128, "d_subobs": 4, "dr_rate": 0,
    "is_bias": False, "n_layers_decoder": 1, "n_layers_encoder": 3,
    "norm_eps": 1e-05, "num_heads": 4, "share_layers": False,
    "use_FNN_in_decoder": True, "use_residual_in_decoder": True,
}

POLICY_MODEL_CONFIG = {
    **_SHARED_MODEL,
    # proven A/robust recipe (train_robust.py verbatim)
    "scale_factor": 0.10,
    "aux_enabled": True, "aux_type": "pair_embedding", "aux_loss_coef": 0.3,
    "aux_target_dim": 4, "aux_loss_coef_critic": 0.05,
    "continuous_action": False, "per_agent_credit": False,
    "selection_head": "bernoulli", "top_k": None, "hard_top_k": False,
    "dist_aux_coef": 0.0, "use_global_stats": True,
}

DKNN_MODEL_CONFIG = {
    **_SHARED_MODEL,
    # dknn original (train_dynamic_knn.py verbatim); aux family is additionally
    # force-disabled inside DynamicKNNPPORLlib.
    "scale_factor": 1.0,
}

PROFILES = {
    "policy_robust": dict(
        method="policy", action_type="binary_vector",
        model_name="neighbor_selector_rl", model_cls=NeighborSelectionPPORLlib,
        model_config=POLICY_MODEL_CONFIG, regime="c2",
        env_extra=dict(expose_aux_target=True, expose_global_stats=True),
        steps=1_920_000, iters=120,
        tune=dict(
            num_workers=4, num_envs_per_worker=4, rollout_fragment_length=1000,
            train_batch_size=16000, sgd_minibatch_size=256, num_sgd_iter=10,
            lr=5e-4, lr_schedule=[[0, 5e-4], [800000, 1e-4]],
            clip_param=0.15, grad_clip=1.0, entropy_coeff=1e-3,
        ),
    ),
    "dknn_c2": dict(
        method="dynamic_knn", action_type=ACTION_TYPE,
        model_name=MODEL_ID, model_cls=DynamicKNNPPORLlib,
        model_config=DKNN_MODEL_CONFIG, regime="c2",
        env_extra=dict(expose_aux_target=False, expose_global_stats=False),
        steps=2_000_000, iters=None,
        tune=dict(
            num_workers=8, num_envs_per_worker=2, rollout_fragment_length=512,
            train_batch_size=8192, sgd_minibatch_size=512, num_sgd_iter=7,
            # dknn original lr anchors; --lr/--lr-end rescale over --steps
            lr=2e-5, lr_schedule=None,
            clip_param=0.2, grad_clip=0.5, entropy_coeff=0.0,
        ),
    ),
    "dknn_legacy": dict(
        method="dynamic_knn", action_type=ACTION_TYPE,
        model_name=MODEL_ID, model_cls=DynamicKNNPPORLlib,
        model_config=DKNN_MODEL_CONFIG, regime="legacy_dknn",
        env_extra=dict(expose_aux_target=False, expose_global_stats=False),
        steps=6_000_000, iters=None,
        tune=dict(
            num_workers=8, num_envs_per_worker=2, rollout_fragment_length=512,
            train_batch_size=8192, sgd_minibatch_size=512, num_sgd_iter=7,
            lr=2e-5, lr_schedule=[[0, 2e-5], [6_000_000, 1e-7]],
            clip_param=0.2, grad_clip=0.5, entropy_coeff=0.0,
        ),
    ),
}


def main():
    prof = PROFILES[ARGS.profile]
    if ARGS.profile == "dknn_legacy" and not ARGS.allow_legacy:
        raise SystemExit(
            "dknn_legacy is defined for preservation only (plan Q3/Q4 rider); "
            "pass --allow-legacy to actually run it.")

    if ARGS.entropy_penalty is not None:
        if prof["method"] != "dynamic_knn":
            raise SystemExit("--entropy-penalty is a dknn-only probe knob")
        prof = dict(prof)
        prof["model_config"] = {**prof["model_config"],
                                "entropy_penalty_coef": ARGS.entropy_penalty}

    seeds = [int(s) for s in ARGS.seeds.split(",") if s.strip()]
    total_steps = ARGS.steps if ARGS.steps is not None else prof["steps"]
    stop_iters = ARGS.iters if ARGS.iters is not None else prof["iters"]

    tune_hp = dict(prof["tune"])
    for cli, key in [("lr", "lr"), ("grad_clip", "grad_clip"),
                     ("entropy_coeff", "entropy_coeff"), ("clip_param", "clip_param"),
                     ("minibatch", "sgd_minibatch_size"), ("sgd_iter", "num_sgd_iter"),
                     ("workers", "num_workers"), ("envs_per_worker", "num_envs_per_worker"),
                     ("fragment", "rollout_fragment_length")]:
        v = getattr(ARGS, cli)
        if v is not None:
            tune_hp[key] = v
    if ARGS.workers is not None or ARGS.envs_per_worker is not None or ARGS.fragment is not None:
        tune_hp["train_batch_size"] = (tune_hp["num_workers"]
                                       * tune_hp["num_envs_per_worker"]
                                       * tune_hp["rollout_fragment_length"])
    if ARGS.lr_end is not None:
        tune_hp["lr_schedule"] = [[0, tune_hp["lr"]], [total_steps, ARGS.lr_end]]
    elif ARGS.lr is not None:
        tune_hp["lr_schedule"] = None  # explicit flat lr overrides profile schedule

    train_cfg = build_env_config(True, prof["action_type"], prof["regime"])
    eval_cfg = build_env_config(False, prof["action_type"], prof["regime"])
    for cfg in (train_cfg, eval_cfg):
        for k, v in prof["env_extra"].items():
            setattr(cfg.env, k, v)
        if ARGS.cap is not None:
            cfg.env.max_time_steps = ARGS.cap

    if ARGS.smoke:
        tune_hp.update(num_workers=1, num_envs_per_worker=2,
                       rollout_fragment_length=250, train_batch_size=500,
                       sgd_minibatch_size=125, num_sgd_iter=2)

    register_env(ENV_NAME, make_env)
    ModelCatalog.register_custom_model("neighbor_selector_rl", NeighborSelectionPPORLlib)
    ModelCatalog.register_custom_model(MODEL_ID, DynamicKNNPPORLlib)

    seed_axis = seeds[0] if len(seeds) == 1 else tune.grid_search(seeds)
    config = {
        "env": ENV_NAME,
        "env_config": {"seed_id": seed_axis, "config": train_cfg.dict()},
        "framework": "torch",
        "callbacks": C2Callbacks,
        "model": {"custom_model": prof["model_name"],
                  "custom_model_config": prof["model_config"]},
        "num_gpus": 0 if ARGS.smoke else 1,
        "num_workers": tune_hp["num_workers"],
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": tune_hp["num_envs_per_worker"],
        "rollout_fragment_length": tune_hp["rollout_fragment_length"],
        "train_batch_size": tune_hp["train_batch_size"],
        "sgd_minibatch_size": tune_hp["sgd_minibatch_size"],
        "num_sgd_iter": tune_hp["num_sgd_iter"],
        "lr": tune_hp["lr"],
        "vf_loss_coeff": 0.5,
        "use_critic": True,
        "use_gae": True,
        "gamma": 0.99,
        "lambda": 0.95,
        "kl_coeff": 0,
        "clip_param": tune_hp["clip_param"],
        "vf_clip_param": 256,
        "grad_clip": tune_hp["grad_clip"],
        "kl_target": 0.01,
        "entropy_coeff": tune_hp["entropy_coeff"],
        "normalize_actions": False,
        "evaluation_interval": None if ARGS.smoke else ARGS.eval_interval,
        "evaluation_duration": 16,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 0 if ARGS.smoke else 2,
        "evaluation_config": {
            "explore": False,
            "env_config": {"seed_id": EVAL_SEED, "config": eval_cfg.dict()},
        },
    }
    if tune_hp.get("lr_schedule"):
        config["lr_schedule"] = tune_hp["lr_schedule"]

    run_name = ARGS.run_name or "uni_{}_s{}".format(
        ARGS.profile, "-".join(str(s) for s in seeds))
    stop = {"timesteps_total": total_steps}
    if stop_iters is not None:
        stop["training_iteration"] = stop_iters

    resolved = {"profile": ARGS.profile, "method": prof["method"],
                "action_type": prof["action_type"], "seeds": seeds,
                "steps": total_steps, "iters": stop_iters, "run_name": run_name,
                "hp": {k: v for k, v in tune_hp.items()},
                "entropy_penalty": ARGS.entropy_penalty,
                "cap": train_cfg.env.max_time_steps,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    print("[unified-config] " + json.dumps(resolved, sort_keys=True), flush=True)

    n_eval = 0 if ARGS.smoke else 2
    num_cpus = ARGS.num_cpus or (tune_hp["num_workers"] + n_eval + 2)
    ray.init(num_cpus=num_cpus,
             object_store_memory=int(ARGS.object_store_gb * 1024**3),
             include_dashboard=False)

    if ARGS.smoke:
        config["evaluation_interval"] = None
        algo = GradLoggingPPO(config={**config,
                                      "env_config": {"seed_id": seeds[0],
                                                     "config": train_cfg.dict()}})
        for i in range(2):
            result = algo.train()
            sr = result.get("sampler_results", {})
            ls = (result.get("info", {}).get("learner", {})
                  .get("default_policy", {}).get("learner_stats", {}))
            print("--- smoke iter {} [{}] reward_mean={} len_mean={} entropy={} "
                  "gnorm_actor={}".format(
                      i + 1, ARGS.profile, sr.get("episode_reward_mean"),
                      sr.get("episode_len_mean"), ls.get("entropy"),
                      ls.get("gnorm_actor_preclip")), flush=True)
        algo.stop()
        return

    tune.run(
        GradLoggingPPO,
        name=run_name,
        # TRAINING_RESULTS_DIR: docker/train_service.sh convention (the repo
        # mount is read-only there); default = <repo>/test_results (gitignored).
        local_dir=os.environ.get("TRAINING_RESULTS_DIR") or repo_path("test_results"),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=3,
        resume="AUTO" if ARGS.resume else False,
    )


if __name__ == "__main__":
    main()
