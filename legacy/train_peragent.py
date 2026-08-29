import copy
import sys

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, Config, load_config
from models.ppo import NeighborSelectionPPORLlib
from legacy.beta_dist import TorchContinuousWeightDist
from callbacks import FlockingCallbacks
from grad_logging_ppo import GradLoggingPPO

if __name__ == "__main__":

    w_ctrl = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02

    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)

    w_align = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--w_align="):
            w_align = float(a.split("=")[1])

    my_config.env.acs_train_w_ctrl = w_ctrl
    my_config.env.acs_train_w_pos  = 1.0
    my_config.env.acs_train_w_vel  = 0.2
    my_config.env.acs_train_w_conn = 0.0
    my_config.env.acs_train_w_align = w_align
    my_config.env.action_type = "binary_vector"
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
    my_config.env.num_agents_pool = [20]
    my_config.env.obs_dim = 4
    my_config.env.periodic_boundary = False
    my_config.env.seed = None
    my_config.env.task_type = "acs"
    my_config.env.use_fixed_episode_length = True
    my_config.env.expose_aux_target = True
    my_config.env.continuous_action = True

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

    env_config_dict = my_config.dict()
    env_config = {
        "seed_id": 42,
        "config": env_config_dict,
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))

    alpha = 1.0
    for a in sys.argv[2:]:
        try:
            if not a.startswith("--"):
                alpha = float(a)
        except ValueError:
            pass

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
        "scale_factor": 0.10,
        "share_layers": False,
        "use_FNN_in_decoder": True,
        "use_residual_in_decoder": True,
        "aux_enabled": True,
        "aux_type": "pair_embedding",
        "aux_loss_coef": 0.3,
        "aux_target_dim": 4,
        "aux_loss_coef_critic": 0.05,
        "continuous_action": True,
        "per_agent_credit": "--no_pa" not in sys.argv,
        "pa_credit_alpha": alpha,
        "pa_replace_ppo": "--replace" in sys.argv,
    }

    model_name = "neighbor_selector_rl"
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)
    ModelCatalog.register_custom_action_dist("beta_dist", TorchContinuousWeightDist)

    mode = "replace" if "--replace" in sys.argv else "additive"
    pa_on = "pa" if "--no_pa" not in sys.argv else "nopa"
    run_name = f"align{w_align:.1f}_wctrl{w_ctrl:.2f}_{pa_on}_260527"

    tune.run(
        GradLoggingPPO,
        name=run_name,
        local_dir="/workspace/test_results",
        checkpoint_freq=10,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        stop={"training_iteration": 100},
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "callbacks": FlockingCallbacks,
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
                "custom_action_dist": "beta_dist",
            },
            "num_gpus": 0.5,
            "num_workers": 4,
            "num_envs_per_worker": 4,
            "rollout_fragment_length": 1000,
            "train_batch_size": 16000,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 1 if "--replace" in sys.argv else 5,
            "lr": 5e-4,
            "lr_schedule": [[0, 5e-4], [800000, 1e-4]],
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
            "entropy_coeff": 0.01 if "--replace" in sys.argv else 0.001,
            "normalize_actions": False,
            "evaluation_interval": None,
        },
    )
