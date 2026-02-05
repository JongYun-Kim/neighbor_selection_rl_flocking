import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from envs.env import NeighborSelectionFlockingEnv, Config, load_config
from models.ppo import NeighborSelectionPPORLlib

if __name__ == "__main__":

    # enable_debugging = False
    enable_debugging = True

    if enable_debugging:
        ray.init(local_mode=True)

    # Set up environment configuration
    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)

    # environment configs
    my_config.env.acs_train_w_ctrl = 0.02
    my_config.env.acs_train_w_pos  = 1.0
    my_config.env.acs_train_w_vel  = 0.2
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
        "seed_id": 42,
        "config": my_config.dict(),  # pass dict to save the config
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))

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
        "scale_factor": 0.002,
        "share_layers": False,
        "use_FNN_in_decoder": True,
        "use_residual_in_decoder": True,
    }

    # register your custom model
    model_name = "neighbor_selector_rl"
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)

    # train
    tune.run(
        "PPO",
        name="neighbor_selection_test_260205",
        # resume=True,
        checkpoint_freq=8,
        keep_checkpoints_num=32,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
            },
            "num_gpus": 0.5,
            "num_workers": 16,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 1024,
            "train_batch_size": 16384,
            "sgd_minibatch_size": 256,
            "num_sgd_iter": 10,
            "lr": 2e-5,
            "lr_schedule": [[0, 2e-5],
                            [1e7, 1e-7],],
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
