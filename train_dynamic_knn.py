import os
import json
import multiprocessing as mp
from pathlib import Path

import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from dynamic_k_nn.identifiers import (
    ACTION_ENCODING,
    ACTION_TYPE,
    EXPERIMENT_NAME,
    MODEL_ID,
    WANDB_PROJECT as DEFAULT_WANDB_PROJECT,
)
from envs.env import NeighborSelectionFlockingEnv, Config, load_config
from models.ppo import NeighborSelectionPPORLlib
from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib


def _env_int(name, default, minimum=1):
    raw_value = os.environ.get(name)
    value = default if raw_value is None else int(raw_value.replace("_", ""))
    if value < minimum:
        raise ValueError("{} must be >= {}, found {}".format(name, minimum, value))
    return value


def _env_float(name, default, minimum=0.0):
    raw_value = os.environ.get(name)
    value = default if raw_value is None else float(raw_value)
    if value < minimum:
        raise ValueError("{} must be >= {}, found {}".format(name, minimum, value))
    return value


def _env_bool(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError("{} must be a boolean, found {!r}".format(name, raw_value))


BASE_ENV_SEED = _env_int("BASE_ENV_SEED", 42, minimum=0)
TRAINING_SWARM_SIZE = _env_int("TRAINING_SWARM_SIZE", 20)
NUM_ROLLOUT_WORKERS = _env_int("NUM_ROLLOUT_WORKERS", 8)
NUM_ENVS_PER_WORKER = _env_int("NUM_ENVS_PER_WORKER", 2)
ROLLOUT_FRAGMENT_LENGTH = _env_int("ROLLOUT_FRAGMENT_LENGTH", 512)
TOTAL_TRAINING_TIMESTEPS = _env_int("TOTAL_TRAINING_TIMESTEPS", 6_000_000)
MAX_TRAINING_TIME_S = _env_int("MAX_TRAINING_TIME_S", 18 * 60 * 60)
DEFAULT_TRAIN_BATCH_SIZE = (
    NUM_ROLLOUT_WORKERS * NUM_ENVS_PER_WORKER * ROLLOUT_FRAGMENT_LENGTH
)
TRAIN_BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", DEFAULT_TRAIN_BATCH_SIZE)
SGD_MINIBATCH_SIZE = _env_int("SGD_MINIBATCH_SIZE", 512)
NUM_SGD_ITER = _env_int("NUM_SGD_ITER", 7)
INITIAL_LR = _env_float("INITIAL_LR", 2e-5)
FINAL_LR = _env_float("FINAL_LR", 1e-7)
CHECKPOINT_FREQ = _env_int("CHECKPOINT_FREQ", 8)
KEEP_CHECKPOINTS_NUM = _env_int("KEEP_CHECKPOINTS_NUM", 5)
WANDB_ENABLED = _env_bool("WANDB_ENABLED", True)
WANDB_PROJECT = os.environ.get(
    "WANDB_PROJECT", DEFAULT_WANDB_PROJECT
)
WANDB_RUN_NAME = os.environ.get(
    "WANDB_RUN_NAME",
    "ppo-dynamic-k-nn-n{}-seed{}-{}m-mb{}-e{}".format(
        TRAINING_SWARM_SIZE,
        BASE_ENV_SEED,
        TOTAL_TRAINING_TIMESTEPS // 1_000_000,
        SGD_MINIBATCH_SIZE,
        NUM_SGD_ITER,
    ),
)
WANDB_API_KEY_FILE = Path(
    os.environ.get("WANDB_API_KEY_FILE", "/run/secrets/wandb_api_key")
)
TRAINING_RESULTS_DIR = os.environ.get(
    "TRAINING_RESULTS_DIR", "/workspace/test_results"
)
TUNE_EXPERIMENT_NAME = os.environ.get("TUNE_EXPERIMENT_NAME", EXPERIMENT_NAME)

if TRAIN_BATCH_SIZE % SGD_MINIBATCH_SIZE != 0:
    raise ValueError(
        "TRAIN_BATCH_SIZE={} must be divisible by SGD_MINIBATCH_SIZE={}".format(
            TRAIN_BATCH_SIZE, SGD_MINIBATCH_SIZE
        )
    )
if FINAL_LR > INITIAL_LR:
    raise ValueError(
        "FINAL_LR={} must not exceed INITIAL_LR={}".format(FINAL_LR, INITIAL_LR)
    )


def create_env(env_context):
    """Create independently seeded vector environments on every Ray worker."""
    env_input = dict(env_context)
    base_seed = env_input.get("seed_id")
    if base_seed is not None:
        worker_index = int(getattr(env_context, "worker_index", 0))
        vector_index = int(getattr(env_context, "vector_index", 0))
        env_input["seed_id"] = base_seed + (1000 * worker_index) + vector_index
    return NeighborSelectionFlockingEnv(env_input)


if __name__ == "__main__":

    # Ray 2.1's W&B callback subclasses multiprocessing.Process. With modern
    # W&B's background service, forking the already multi-threaded Ray driver
    # can segfault the logger process; spawn is portable and keeps the logger
    # isolated while preserving the official callback's queue protocol.
    mp.set_start_method("spawn", force=True)

    workflow_run_id = os.environ.get("WORKFLOW_RUN_ID", "manual-dynamic-k-nn")
    if WANDB_ENABLED and (
        not WANDB_API_KEY_FILE.is_file() or WANDB_API_KEY_FILE.stat().st_size == 0
    ):
        raise FileNotFoundError(
            "A non-empty W&B API key file is required at {}".format(
                WANDB_API_KEY_FILE
            )
        )

    enable_debugging = False
    # enable_debugging = True

    if enable_debugging:
        ray.init(local_mode=True)

    # Set up environment configuration
    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)

    # environment configs
    my_config.env.acs_train_w_ctrl = 0.02
    my_config.env.acs_train_w_pos  = 1.0
    my_config.env.acs_train_w_vel  = 0.2
    my_config.env.action_type = ACTION_TYPE
    my_config.env.agent_name_prefix = "agent_"
    my_config.env.alignment_goal = 0.97
    my_config.env.alignment_rate_goal = 0.03
    my_config.env.alignment_window_length = 32
    my_config.env.comm_range = None
    my_config.env.dt = 0.1
    my_config.env.entropy_p_goal = None
    my_config.env.entropy_p_rate_goal = 0.1
    my_config.env.entropy_rate_window_length = 50
    my_config.env.entropy_v_goal = 0.1
    my_config.env.entropy_v_rate_goal = 0.2
    my_config.env.env_mode = "single_env"
    my_config.env.get_action_hist = False
    my_config.env.get_state_hist = False
    my_config.env.ignore_comm_lost_agents = False
    my_config.env.is_training = True
    my_config.env.max_time_steps = 1000
    my_config.env.num_agents_pool = [TRAINING_SWARM_SIZE]
    my_config.env.obs_dim = 4
    my_config.env.observation_type = "ego_centric"
    my_config.env.periodic_boundary = False
    my_config.env.seed = None
    my_config.env.task_type = "acs"
    my_config.env.use_fixed_episode_length = True

    # control config:
    my_config.control.beta = 1/3
    my_config.control.initial_position_bound = 250.0
    my_config.control.k1 = 1.0
    my_config.control.k2 = 3.0
    my_config.control.lam = 5.0
    my_config.control.max_turn_rate = 8/15
    my_config.control.r0 = 60.0
    my_config.control.rho = 1.0
    my_config.control.sig = 1.0
    my_config.control.speed = 15.0

    # register your custom environment
    env_config = {
        "seed_id": BASE_ENV_SEED,
        "config": my_config.dict(),  # pass dict to save the config
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, create_env)

    # Set up custom model configuration
    custom_model_config = {
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
        "scale_factor": 1.0,
        "share_layers": False,
        "use_FNN_in_decoder": True,
        "use_residual_in_decoder": True,
    }

    # The binary model remains available for legacy binary-vector experiments.
    ModelCatalog.register_custom_model("neighbor_selector_rl", NeighborSelectionPPORLlib)
    model_name = MODEL_ID
    ModelCatalog.register_custom_model(model_name, DynamicKNNPPORLlib)

    resolved_training_config = {
        "base_env_seed": BASE_ENV_SEED,
        "training_swarm_size": TRAINING_SWARM_SIZE,
        "action_type": ACTION_TYPE,
        "action_encoding": ACTION_ENCODING,
        "tune_experiment_name": TUNE_EXPERIMENT_NAME,
        "num_rollout_workers": NUM_ROLLOUT_WORKERS,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
        "num_sgd_iter": NUM_SGD_ITER,
        "initial_lr": INITIAL_LR,
        "final_lr": FINAL_LR,
        "total_training_timesteps": TOTAL_TRAINING_TIMESTEPS,
        "max_training_time_s": MAX_TRAINING_TIME_S,
        "checkpoint_freq": CHECKPOINT_FREQ,
        "keep_checkpoints_num": KEEP_CHECKPOINTS_NUM,
        "wandb_enabled": WANDB_ENABLED,
        "wandb_project": WANDB_PROJECT,
        "wandb_run_name": WANDB_RUN_NAME,
    }
    print(
        "[training-config] {}".format(
            json.dumps(resolved_training_config, sort_keys=True)
        ),
        flush=True,
    )

    callbacks = []
    if WANDB_ENABLED:
        callbacks.append(
            WandbLoggerCallback(
                project=WANDB_PROJECT,
                group=workflow_run_id,
                api_key_file=str(WANDB_API_KEY_FILE),
                excludes=[
                    "hist_stats",
                    "sampler_results/hist_stats",
                    "evaluation/hist_stats",
                    "media",
                ],
                log_config=False,
                save_checkpoints=False,
                name=WANDB_RUN_NAME,
                tags=[
                    "dynamic-k-nn",
                    "ppo",
                    "n{}".format(TRAINING_SWARM_SIZE),
                    "seed{}".format(BASE_ENV_SEED),
                    "{}m".format(TOTAL_TRAINING_TIMESTEPS // 1_000_000),
                    "mb{}".format(SGD_MINIBATCH_SIZE),
                    "epochs{}".format(NUM_SGD_ITER),
                ],
                job_type="training",
                resume="allow",
            )
        )

    # train
    tune.run(
        "PPO",
        name=TUNE_EXPERIMENT_NAME,
        local_dir=TRAINING_RESULTS_DIR,
        # AUTO+ERRORED restores the same Tune trial/checkpoint after either a
        # process/container failure or a host reboot. time_total_s is carried
        # in the restored trial, so downtime does not consume the safety cap.
        resume="AUTO+ERRORED",
        # The validated default reaches 6M fixed-20-agent steps in about 12.93
        # hours on the target i9-9900KF/RTX 3090. The wall-clock condition is a
        # safety cap for slower runs; Tune stops when either condition is met.
        stop={
            "timesteps_total": TOTAL_TRAINING_TIMESTEPS,
            "time_total_s": MAX_TRAINING_TIME_S,
        },
        # 8 * 8192 = 65,536 steps, roughly 8.5 minutes on the target GPU.
        checkpoint_freq=CHECKPOINT_FREQ,
        # Ray's persistent checkpoint manager keeps the top five scored
        # checkpoints and the newest checkpoint; checkpoint_at_end makes that
        # newest checkpoint the logical final model.
        keep_checkpoints_num=KEEP_CHECKPOINTS_NUM,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        max_failures=3,
        callbacks=callbacks,
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "seed": BASE_ENV_SEED,
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
            },
            # i9-9900KF: reserve one CPU for the driver and avoid saturating all
            # 16 logical CPUs with rollout actors. Two vector envs modestly
            # improve batch inference while keeping only eight worker processes.
            "num_gpus": 1,
            "num_workers": NUM_ROLLOUT_WORKERS,
            "num_cpus_per_worker": 1,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
            "num_sgd_iter": NUM_SGD_ITER,
            "lr": INITIAL_LR,
            "lr_schedule": [[0, INITIAL_LR],
                            [TOTAL_TRAINING_TIMESTEPS, FINAL_LR],],
            "vf_loss_coeff": 0.5,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.95,
            "kl_coeff": 0,
            "clip_param": 0.2,
            "vf_clip_param": 256,
            "grad_clip": 0.5,
            "kl_target": 0.01,
        },
    )
