"""Regression and integration tests for Dynamic-k NN selection."""

import os
import unittest
from collections import Counter

import numpy as np
import torch

from ray.rllib.models import ModelCatalog

from baselines import RandomNeighborSelection
from dynamic_k_nn.identifiers import ACTION_TYPE
from envs.env import (
    Config,
    NeighborSelectionFlockingEnv,
    config_to_env_input,
    load_dict,
)
from models.modules.pointer_net import RawAttentionScoreGenerator
from models.ppo import NeighborSelectionPPORLlib
from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib


TRAINING_SWARM_SIZE = 20


def dynamic_k_nn_config(**overrides):
    config = Config(**load_dict("envs/default_env_config.yaml"))
    config.env.action_type = ACTION_TYPE
    config.env.num_agents_pool = [10, 15, 20]
    config.env.comm_range = None
    config.env.observation_type = "ego_centric"
    config.env.max_time_steps = 4
    config.env.use_fixed_episode_length = True
    for key, value in overrides.items():
        setattr(config.env, key, value)
    return config


def model_config():
    return {
        "d_embed_context": 16,
        "d_embed_input": 16,
        "d_ff": 32,
        "d_ff_decoder": 32,
        "d_model": 16,
        "d_model_decoder": 16,
        "d_subobs": 4,
        "dr_rate": 0,
        "is_bias": False,
        "n_layers_decoder": 1,
        "n_layers_encoder": 1,
        "norm_eps": 1e-5,
        "num_heads": 4,
        "scale_factor": 1.0,
        "share_layers": False,
        "use_FNN_in_decoder": True,
        "use_residual_in_decoder": True,
    }


def make_dynamic_k_nn_env(seed=0, **overrides):
    return NeighborSelectionFlockingEnv(
        config_to_env_input(dynamic_k_nn_config(**overrides), seed_id=seed)
    )


def make_fixed_dynamic_k_nn_env(num_agents, seed=0, **overrides):
    overrides = dict(overrides)
    overrides["num_agents_pool"] = [num_agents]
    return NeighborSelectionFlockingEnv(
        config_to_env_input(dynamic_k_nn_config(**overrides), seed_id=seed)
    )


def deterministic_reset(env, num_agents):
    positions = np.zeros((num_agents, 2), dtype=np.float64)
    positions[:, 0] = 3.0 * np.arange(num_agents)
    # Equal-distance candidates around ego 0, plus a farther candidate.
    positions[1] = [1.0, 0.0]
    positions[2] = [-1.0, 0.0]
    positions[3] = [2.0, 0.0]
    headings = np.zeros((num_agents, 1), dtype=np.float64)
    velocities = np.zeros((num_agents, 2), dtype=np.float64)
    velocities[:, 0] = env.config.control.speed
    return env.custom_reset(
        positions,
        velocities,
        headings,
        num_agents_max=TRAINING_SWARM_SIZE,
        comm_range=None,
    )


def torch_obs(obs):
    result = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
    result["local_agent_infos"] = result["local_agent_infos"].float()
    result["absolute_headings"] = result["absolute_headings"].float()
    return result


def make_dynamic_k_nn_model(env):
    distribution_class, num_outputs = ModelCatalog.get_action_dist(
        env.action_space, {}, framework="torch"
    )
    model = DynamicKNNPPORLlib(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_outputs=num_outputs,
        model_config={"custom_model_config": model_config()},
        name="dynamic_k_nn_test_model",
    )
    model.eval()
    return model, distribution_class, num_outputs


class DynamicKNNEnvironmentTest(unittest.TestCase):
    def test_action_and_observation_spaces_are_dynamic_for_supported_sizes(self):
        for num_agents in (10, 20, 40):
            env = make_fixed_dynamic_k_nn_env(num_agents, seed=num_agents)
            self.assertEqual(env.action_space.shape, (num_agents,))
            np.testing.assert_array_equal(
                env.action_space.nvec,
                np.full(num_agents, num_agents),
            )
            self.assertEqual(
                env.observation_space["local_agent_infos"].shape,
                (num_agents, num_agents, 4),
            )
            observation = env.reset()
            pointer = np.arange(num_agents, dtype=np.int64)
            binary = env.cutoff_indices_to_binary_action(pointer)
            self.assertEqual(binary.shape, (num_agents, num_agents))
            next_observation, reward, _, info = env.step(pointer)
            self.assertEqual(next_observation["padding_mask"].shape, (num_agents,))
            self.assertTrue(np.isfinite(reward))
            self.assertNotIn("binary_action", info)
            self.assertNotIn("control_inputs", info)

    def test_fixed_action_space_uniform_swarm_pool_and_invariants(self):
        env = make_dynamic_k_nn_env(seed=123)
        np.testing.assert_array_equal(
            env.action_space.nvec,
            np.full(TRAINING_SWARM_SIZE, TRAINING_SWARM_SIZE),
        )

        counts = Counter()
        for _ in range(90):
            env.reset()
            counts[int(env.num_agents)] += 1
        self.assertEqual(set(counts), {10, 15, 20})
        # A broad deterministic smoke bound catches accidental non-uniform or
        # range sampling without making the test statistically fragile.
        self.assertTrue(all(15 <= count <= 45 for count in counts.values()), counts)

        bad_config = dynamic_k_nn_config(comm_range=100.0)
        with self.assertRaisesRegex(ValueError, "comm_range=None"):
            NeighborSelectionFlockingEnv(config_to_env_input(bad_config, seed_id=0))

        env.config.env.comm_range = 1.0
        with self.assertRaisesRegex(ValueError, "comm_range=None"):
            env.reset()

    def test_self_external_tie_padding_and_action_history(self):
        env = make_dynamic_k_nn_env(seed=2, get_action_hist=True)
        obs = deterministic_reset(env, 10)
        pointer = np.arange(TRAINING_SWARM_SIZE, dtype=np.int64)

        self_action = env.cutoff_indices_to_binary_action(pointer)
        np.testing.assert_array_equal(
            np.flatnonzero(self_action[0]), np.array([0])
        )

        pointer[0] = 1
        tied_action = env.cutoff_indices_to_binary_action(pointer)
        np.testing.assert_array_equal(
            np.flatnonzero(tied_action[0]), np.array([0, 1, 2])
        )
        # The mask is directed: other egos still made their self choices.
        np.testing.assert_array_equal(
            np.flatnonzero(tied_action[1]), np.array([1])
        )

        pointer[0] = 3
        farther_action = env.cutoff_indices_to_binary_action(pointer)
        np.testing.assert_array_equal(
            np.flatnonzero(farther_action[0]), np.array([0, 1, 2, 3])
        )

        pointer[0] = 19
        padding_choice = env.cutoff_indices_to_binary_action(pointer)
        np.testing.assert_array_equal(
            np.flatnonzero(padding_choice[0]), np.array([0])
        )
        self.assertFalse(padding_choice[10:, :].any())
        self.assertFalse(padding_choice[:, 10:].any())

        pointer[0] = 1
        expected = env.cutoff_indices_to_binary_action(pointer)
        next_obs, reward, done, info = env.step(pointer)
        np.testing.assert_array_equal(env.action_hist[0], expected.astype(bool))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertEqual(next_obs["padding_mask"].sum(), 10)
        self.assertIn("original_reward", info)

        active_pairs = obs["padding_mask"][:, None] & obs["padding_mask"][None, :]
        np.testing.assert_array_equal(obs["neighbor_masks"], active_pairs)


class DynamicKNNModelTest(unittest.TestCase):
    def test_n20_state_dict_strict_loads_and_steps_at_10_20_40(self):
        source_env = make_fixed_dynamic_k_nn_env(20, seed=20)
        source_model, _, _ = make_dynamic_k_nn_model(source_env)
        source_state = source_model.state_dict()

        for num_agents in (10, 20, 40):
            env = make_fixed_dynamic_k_nn_env(num_agents, seed=num_agents)
            model, _, num_outputs = make_dynamic_k_nn_model(env)
            model.load_state_dict(source_state, strict=True)
            observation = env.reset()
            with torch.no_grad():
                logits, _ = model.forward({"obs": torch_obs(observation)}, [], None)
            self.assertEqual(num_outputs, num_agents ** 2)
            self.assertEqual(tuple(logits.shape), (1, num_agents ** 2))
            pointer = logits.reshape(1, num_agents, num_agents).argmax(-1)[0]
            env.step(pointer.cpu().numpy().astype(np.int64))

    def test_candidate_masks_distribution_and_dynamic_actor(self):
        env = make_dynamic_k_nn_env(seed=3)
        model, distribution_class, num_outputs = make_dynamic_k_nn_model(env)
        self.assertEqual(num_outputs, TRAINING_SWARM_SIZE ** 2)
        self.assertIsInstance(model.actor.generator, RawAttentionScoreGenerator)

        for num_agents in (10, 15, 20):
            obs = deterministic_reset(env, num_agents)
            obs_t = torch_obs(obs)
            with torch.no_grad():
                logits, _ = model.forward({"obs": obs_t}, [], None)
                value = model.value_function()
            self.assertEqual(tuple(logits.shape), (1, 400))
            self.assertEqual(tuple(value.shape), (1,))

            matrix = logits.reshape(1, 20, 20)[0]
            # Distance never limits candidates: every active candidate has a
            # finite logit for every active ego, including the farthest agent.
            self.assertTrue(torch.all(matrix[:num_agents, :num_agents] > -1e8))
            if num_agents < 20:
                self.assertTrue(torch.all(matrix[:, num_agents:] <= -1e8))

            distribution = distribution_class(logits, model)
            sampled = distribution.sample()[0]
            self.assertEqual(tuple(sampled.shape), (20,))
            self.assertTrue(obs_t["padding_mask"][0, sampled].bool().all())
            if num_agents < 20:
                self.assertTrue(torch.all(sampled[num_agents:] == 0))

            # The Transformer/pointer core itself derives N from its tensors;
            # RLlib alone fixes this training interface at N_max=20.
            compact_obs = {
                "local_agent_infos": obs_t["local_agent_infos"][:, :num_agents, :num_agents],
                "neighbor_masks": obs_t["neighbor_masks"][:, :num_agents, :num_agents],
                "padding_mask": obs_t["padding_mask"][:, :num_agents],
                "is_from_my_env": obs_t["is_from_my_env"],
            }
            with torch.no_grad():
                compact_scores, compact_context, _, _ = model.actor(compact_obs)
            self.assertEqual(tuple(compact_scores.shape), (1, num_agents, num_agents))
            self.assertEqual(tuple(compact_context.shape), (1, 1, 16))

    def test_padding_is_excluded_from_observation_context_and_critic(self):
        env = make_dynamic_k_nn_env(seed=4)
        obs = deterministic_reset(env, 10)
        self.assertFalse(obs["local_agent_infos"][10:, :, :].any())
        self.assertFalse(obs["local_agent_infos"][:, 10:, :].any())
        self.assertFalse(obs["neighbor_masks"][10:, :].any())
        self.assertFalse(obs["neighbor_masks"][:, 10:].any())

        model, _, _ = make_dynamic_k_nn_model(env)
        original = torch_obs(obs)
        perturbed = {key: value.clone() for key, value in original.items()}
        perturbed["local_agent_infos"][:, 10:, :, :] = 12345.0
        perturbed["local_agent_infos"][:, :, 10:, :] = -54321.0

        with torch.no_grad():
            logits_a, _ = model.forward({"obs": original}, [], None)
            value_a = model.value_function().clone()
            logits_b, _ = model.forward({"obs": perturbed}, [], None)
            value_b = model.value_function().clone()

        torch.testing.assert_close(logits_a, logits_b)
        torch.testing.assert_close(value_a, value_b)


class LegacyBinaryRegressionTest(unittest.TestCase):
    def test_binary_environment_model_and_baseline_still_run(self):
        config = Config(**load_dict("envs/default_env_config.yaml"))
        config.env.action_type = "binary_vector"
        config.env.num_agents_pool = [5]
        config.env.max_time_steps = 2
        env = NeighborSelectionFlockingEnv(config_to_env_input(config, seed_id=5))
        obs = env.reset()

        action = RandomNeighborSelection(seed=5)(obs)
        env.step(action)

        _, num_outputs = ModelCatalog.get_action_dist(
            env.action_space, {}, framework="torch"
        )
        model = NeighborSelectionPPORLlib(
            env.observation_space,
            env.action_space,
            num_outputs,
            {"custom_model_config": model_config()},
            "legacy_binary_test_model",
        )
        with torch.no_grad():
            logits, _ = model.forward({"obs": torch_obs(obs)}, [], None)
        self.assertEqual(tuple(logits.shape), (1, 2 * 5 * 5))


class DynamicKNNPPORolloutTest(unittest.TestCase):
    def test_short_ppo_rollout(self):
        import ray
        from ray.rllib.algorithms.ppo import PPO
        from ray.tune.registry import register_env

        suffix = str(os.getpid())
        env_name = "dynamic_k_nn_rollout_" + suffix
        model_name = "dynamic_k_nn_rollout_model_" + suffix
        env_config = config_to_env_input(
            dynamic_k_nn_config(is_training=True), seed_id=7
        )
        register_env(env_name, lambda cfg: NeighborSelectionFlockingEnv(cfg))
        ModelCatalog.register_custom_model(model_name, DynamicKNNPPORLlib)

        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            local_mode=True,
            num_cpus=1,
            include_dashboard=False,
            log_to_driver=False,
        )
        algorithm = None
        try:
            algorithm = PPO(config={
                "env": env_name,
                "env_config": env_config,
                "framework": "torch",
                "model": {
                    "custom_model": model_name,
                    "custom_model_config": model_config(),
                },
                "num_workers": 0,
                "num_gpus": 0,
                "rollout_fragment_length": 4,
                "train_batch_size": 8,
                "sgd_minibatch_size": 4,
                "num_sgd_iter": 1,
                "simple_optimizer": True,
                "disable_env_checking": False,
                "log_level": "ERROR",
            })
            result = algorithm.train()
            self.assertGreaterEqual(result["timesteps_total"], 8)

            env = make_dynamic_k_nn_env(seed=8)
            obs = env.reset()
            action = algorithm.compute_single_action(obs)
            self.assertEqual(action.shape, (20,))
            active = obs["padding_mask"]
            self.assertTrue(active[action[:env.num_agents]].all())
            env.step(action)
        finally:
            if algorithm is not None:
                algorithm.stop()
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
