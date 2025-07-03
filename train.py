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
    # env-env config:
    my_config.env.num_agents_pool = [8, 16]
    my_config.env.max_time_steps = 200
    my_config.env.use_fixed_episode_length = True
    # my_config.env.comm_range = None
    my_config.env.is_training = True
    # env-control config:
    # my_config.control.speed = 15.0
    # my_config.control.initial_position_bound = 250.0
    # my_config.control.max_turn_rate = 15/8

    # register your custom environment
    env_config = {
        "seed_id": None,
        "config": my_config.dict(),  # pass dict to save the config
    }
    env_name = "neighbor_selection_flocking_env"
    register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))

    # Set up custom model configuration
    custom_model_config = {
        "share_layers": False,
        "d_subobs": 4,
        "d_embed_input": 256,
        "d_embed_context": 256,
        "d_model": 256,
        "d_model_decoder": 256,
        "n_layers_encoder": 3,
        "n_layers_decoder": 1,
        "num_heads": 8,
        "d_ff": 512,
        "d_ff_decoder": 512,
        "dr_rate": 0,
        "norm_eps": 1e-5,
        "is_bias": False,  # Default is False, but True used in LazyControl
        "use_residual_in_decoder": True,
        "use_FNN_in_decoder": True,
        "scale_factor": 0.05,
    }

    # register your custom model
    model_name = "neighbor_selector_rl"
    ModelCatalog.register_custom_model(model_name, NeighborSelectionPPORLlib)

    # train
    tune.run(
        "PPO",
        name="neighbor_selection_test_250702",
        # resume=True,
        # stop={"episode_reward_mean": -101},
        # stop={"training_iteration": 300},
        checkpoint_freq=0,
        keep_checkpoints_num=16,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            #
            # "callbacks": MyCallbacks,
            #
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
                # "custom_action_dist": "det_cont_action_dist" if custom_model_config["use_deterministic_action_dist"] else None,
            },
            "num_gpus": 1,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 400,
            "train_batch_size": 400,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 4,
            # "batch_mode": "complete_episodes",
            # "batch_mode": "truncate_episodes",
            "lr": 4e-5,
            # "lr_schedule": [[0, 2e-5],
            #                 [1e7, 1e-7],
            #                 ],
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.25,
            # In the...
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.991,
            "lambda": 0.96,
            "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
            # "entropy_coeff": tune.grid_search([0, 0.001, 0.0025, 0.01]),
            # "entropy_coeff_schedule": None,
            # "entropy_coeff_schedule": [[0, 0.003],
            #                            [5e4, 0.002],
            #                            [1e5, 0.001],
            #                            [2e5, 0.0005],
            #                            [5e5, 0.0002],
            #                            [1e6, 0.0001],
            #                            [2e6, 0],
            #                            ],
            "clip_param": 0.22,  # 0.3
            "vf_clip_param": 256,
            # "grad_clip": None,
            "grad_clip": 10.0,
            "kl_target": 0.01,
        },
    )