"""Focused contract tests for the compact sensitivity rollout runner."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from eval.artifacts import atomic_save_npz, run_fingerprint
from eval.defaults import DEFAULT_CHECKPOINT
from eval.policies import DynamicKNNInferencePolicy
from eval.sensitivity import analysis as sensitivity_analysis
from eval.sensitivity.config import (
    OAT_CONFIG_PATH,
    REFINED_NEAR_ZERO_CONFIG_PATH,
    load_sensitivity_config,
)
from eval.sensitivity.runner import (
    PROTOCOL_ID,
    SUITE_ID,
    SparseAccumulator,
    _expected_meta,
    _summary_from_payload,
    build_sensitivity_env_config,
    episode_path,
    run_sensitivity,
    validate_sensitivity_bundle,
    validate_sensitivity_episode,
)


class _FakeEnv:
    def __init__(self, setting, first_position=0.0):
        states = np.zeros((setting.num_agents, 5), dtype=np.float64)
        states[:, 0] = np.arange(setting.num_agents, dtype=np.float64)
        states[0, 0] = first_position
        states[:, 2] = setting.speed
        self.state = {
            "agent_states": states,
            "padding_mask": np.ones(setting.num_agents, dtype=bool),
        }


def _metadata(setting, horizon, policy="learned_deterministic", seed=0):
    return {
        "run_id": "unit",
        "suite": SUITE_ID,
        "protocol_id": PROTOCOL_ID,
        "fingerprint": "fingerprint",
        "checkpoint_id": "checkpoint",
        "checkpoint_hash": "package",
        "setting_id": setting.setting_id,
        "factor": setting.factor,
        "factor_value": setting.factor_value,
        "memberships": list(setting.memberships),
        "mode": setting.mode,
        "policy": policy,
        "seed": seed,
        "action_seed": 123,
        "horizon": horizon,
        "policy_interval_seconds": 0.1,
        "substeps_per_policy_interval": setting.substeps_per_policy_interval,
        "dynamics_dt": setting.dynamics_dt,
        "physics_steps": horizon * setting.substeps_per_policy_interval,
        "num_agents": setting.num_agents,
        "speed": setting.speed,
        "minimum_turn_radius": setting.minimum_turn_radius,
        "max_turn_rate": setting.max_turn_rate,
        "interaction_radius": setting.interaction_radius,
        "acs_gain_multiplier": setting.acs_gain_multiplier,
        "acs_lambda": setting.acs_lambda,
        "acs_sigma": setting.acs_sigma,
        "initial_position_bound": setting.initial_position_bound,
        "backend": "cpu",
        "batch_size": 1,
        "config_sha256": "config",
    }


def _one_macro_artifact(setting, *, policy="learned_deterministic",
                        first_position=0.0):
    env = _FakeEnv(setting, first_position=first_position)
    accumulator = SparseAccumulator(env, setting, horizon=1)
    substeps, agents = setting.substeps_per_policy_interval, setting.num_agents
    masks = np.tile(np.eye(agents, dtype=bool), (substeps, 1, 1))
    controls = np.zeros((substeps, agents), dtype=np.float64)
    accumulator.append_macro(0, np.full(substeps, -1.0 / substeps), masks, controls)
    meta = _metadata(setting, horizon=1, policy=policy)
    return accumulator.payload(meta), meta


class SensitivityRunnerTest(unittest.TestCase):
    def test_actual_three_policy_run_resumes_from_sparse_cache(self):
        source = load_sensitivity_config(OAT_CONFIG_PATH)
        config = replace(
            source,
            settings=(source.setting_map()["baseline"],),
            seeds=(0, 1),
            macro_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "sensitivity"
            manifest = run_sensitivity(
                DEFAULT_CHECKPOINT, bundle, "unit-smoke", config,
                workers=1, batch_size=2,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["episode_count"], 6)
            paths = tuple(
                episode_path(bundle, "baseline", policy, seed)
                for policy in config.policies
                for seed in config.seeds
            )
            self.assertTrue(all(path.is_file() for path in paths))

            with mock.patch.object(
                    sensitivity_analysis, "verify_summary_tables",
                    wraps=sensitivity_analysis.verify_summary_tables) as verify:
                validated = validate_sensitivity_bundle(bundle)
            verify.assert_called_once()
            self.assertEqual(validated["status"], "pass")
            self.assertEqual(validated["episodes"], 6)

            with self.assertRaisesRegex(ValueError, "another run ID"):
                run_sensitivity(
                    DEFAULT_CHECKPOINT, bundle, "other-run", config,
                    workers=1, batch_size=2,
                )

            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            with mock.patch(
                    "eval.sensitivity.runner.DynamicKNNInferencePolicy",
                    side_effect=AssertionError("cached run reloaded the model")):
                resumed = run_sensitivity(
                    DEFAULT_CHECKPOINT, bundle, "unit-smoke", config,
                    workers=1, batch_size=2,
                )
            self.assertEqual(resumed["episode_count"], 6)
            self.assertEqual(mtimes, {
                path: path.stat().st_mtime_ns for path in paths
            })

            missing = episode_path(
                bundle, "baseline", "learned_deterministic", 1)
            cached_peer = episode_path(
                bundle, "baseline", "learned_deterministic", 0)
            cached_mtime = cached_peer.stat().st_mtime_ns
            missing.unlink()
            observed_batch_sizes = []
            original_actions = DynamicKNNInferencePolicy.actions

            def recording_actions(instance, observations, *args, **kwargs):
                observed_batch_sizes.append(len(observations))
                return original_actions(instance, observations, *args, **kwargs)

            with mock.patch.object(
                    DynamicKNNInferencePolicy, "actions", new=recording_actions):
                repaired = run_sensitivity(
                    DEFAULT_CHECKPOINT, bundle, "unit-smoke", config,
                    workers=1, batch_size=2,
                )
            self.assertEqual(repaired["episode_count"], 6)
            self.assertEqual(observed_batch_sizes, [2, 2])
            self.assertTrue(missing.is_file())
            self.assertEqual(cached_peer.stat().st_mtime_ns, cached_mtime)

    def test_programmatic_runner_rejects_malformed_device(self):
        config = load_sensitivity_config(
            OAT_CONFIG_PATH, settings=("baseline",), seeds=(0,))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "device must be"):
                run_sensitivity(
                    DEFAULT_CHECKPOINT, Path(temporary), "invalid-device", config,
                    device="cudafoo", dry_run=True,
                )

    def test_standard_and_refined_settings_project_to_physics_config(self):
        standard = load_sensitivity_config(
            OAT_CONFIG_PATH,
            settings=("minimum_turn_radius_14p0625",),
        ).settings[0]
        refined = load_sensitivity_config(
            REFINED_NEAR_ZERO_CONFIG_PATH,
            settings=("refined_minimum_turn_radius_14p0625",),
        ).settings[0]

        standard_cfg = build_sensitivity_env_config(standard, 7, "r0_log")
        refined_cfg = build_sensitivity_env_config(refined, 7, "r0_log")

        self.assertEqual(standard_cfg.env.max_time_steps, 7)
        self.assertEqual(refined_cfg.env.max_time_steps, 14)
        self.assertAlmostEqual(standard_cfg.env.dt, 0.1)
        self.assertAlmostEqual(refined_cfg.env.dt, 0.05)
        self.assertEqual(standard_cfg.env.num_agents_pool, [20])
        self.assertEqual(refined_cfg.env.num_agents_pool, [20])
        for cfg, setting in ((standard_cfg, standard), (refined_cfg, refined)):
            self.assertEqual(cfg.env.action_type, "dynamic_k_nn")
            self.assertTrue(cfg.env.evaluation_diagnostics)
            self.assertTrue(cfg.env.use_fixed_episode_length)
            self.assertEqual(cfg.env.obs_position_scale, "r0_log")
            self.assertAlmostEqual(cfg.env.entropy_p_goal,
                                   0.7 * setting.interaction_radius)
            self.assertAlmostEqual(cfg.control.speed, setting.speed)
            self.assertAlmostEqual(cfg.control.max_turn_rate,
                                   setting.max_turn_rate)
            self.assertAlmostEqual(cfg.control.r0, setting.interaction_radius)
            self.assertAlmostEqual(cfg.control.lam, setting.acs_lambda)
            self.assertAlmostEqual(cfg.control.sig, setting.acs_sigma)
            self.assertAlmostEqual(cfg.control.initial_position_bound,
                                   setting.initial_position_bound)

    def test_accumulator_sums_macro_reward_and_aggregates_diagnostics(self):
        setting = load_sensitivity_config(
            REFINED_NEAR_ZERO_CONFIG_PATH,
            settings=("refined_minimum_turn_radius_14p0625",),
        ).settings[0]
        env = _FakeEnv(setting)
        initial = env.state["agent_states"].copy()
        accumulator = SparseAccumulator(env, setting, horizon=1)
        env.state["agent_states"][:, 0] += 0.25

        agents = setting.num_agents
        sparse = np.eye(agents, dtype=bool)
        sparse[np.arange(agents), (np.arange(agents) + 1) % agents] = True
        masks = np.stack((sparse, np.ones((agents, agents), dtype=bool)))
        rate = setting.max_turn_rate
        controls = np.stack((np.full(agents, rate), np.full(agents, -rate / 2)))
        accumulator.append_macro(
            0, rewards=np.asarray([-1.25, -2.75]), masks=masks, controls=controls)

        self.assertTrue(np.isnan(accumulator.series["reward"][0]))
        self.assertEqual(accumulator.series["reward"][1], -4.0)
        self.assertAlmostEqual(accumulator.steps["mean_selected_neighbors"][0], 10.0)
        self.assertAlmostEqual(accumulator.steps["mean_abs_control"][0],
                               0.75 * rate, places=6)
        self.assertAlmostEqual(accumulator.steps["max_abs_control"][0], rate,
                               places=6)
        self.assertAlmostEqual(
            accumulator.steps["control_saturation_fraction"][0], 0.5)
        np.testing.assert_array_equal(accumulator.initial_state, initial)
        self.assertFalse(np.shares_memory(
            accumulator.initial_state, env.state["agent_states"]))

    def test_refined_c2_is_judged_on_ten_hz_macro_boundaries(self):
        setting = load_sensitivity_config(
            REFINED_NEAR_ZERO_CONFIG_PATH,
            settings=("refined_minimum_turn_radius_0p477464829275686",),
        ).settings[0]
        horizon = 300
        payload = {
            "phi": np.full(horizon + 1, 0.99),
            "s_ent": np.full(horizon + 1, 40.0),
            "v_ent": np.zeros(horizon + 1),
            "n_comp_r0": np.ones(horizon + 1, dtype=np.int16),
            "reward": np.r_[np.nan, np.full(horizon, -2.0)],
            "mean_selected_neighbors": np.zeros(horizon),
            "mean_abs_control": np.zeros(horizon),
            "max_abs_control": np.zeros(horizon),
            "control_saturation_fraction": np.zeros(horizon),
        }
        summary = _summary_from_payload(_metadata(setting, horizon), payload)

        self.assertEqual(setting.substeps_per_policy_interval, 59)
        self.assertEqual(summary["t_fire"], 299)
        self.assertAlmostEqual(summary["t_fire_seconds"], 29.9)
        self.assertEqual(summary["J"], 2.0 * 299)

    def test_sparse_validation_accepts_artifact_and_rejects_tampering(self):
        setting = load_sensitivity_config(
            OAT_CONFIG_PATH, settings=("baseline",),
        ).settings[0]
        (payload, expected_summary), meta = _one_macro_artifact(setting)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "episode.npz"
            atomic_save_npz(path, payload)
            actual, initial_hash = validate_sensitivity_episode(path, meta)
            for key, value in expected_summary.items():
                if isinstance(value, float) and np.isnan(value):
                    self.assertTrue(np.isnan(actual[key]))
                else:
                    self.assertEqual(actual[key], value)
            self.assertEqual(initial_hash, json.loads(payload["meta"].item())[
                "initial_state_sha256"])

            cases = []
            initial = {name: np.asarray(value).copy() for name, value in payload.items()}
            initial["initial_agent_states"][0, 0] += 1.0
            cases.append(("initial state", "initial-state hash mismatch", initial))

            top_level = {name: np.asarray(value).copy() for name, value in payload.items()}
            top_level["success"] = np.asarray(1, dtype=np.int8)
            cases.append((
                "top-level outcome", "stored top-level success mismatch", top_level))

            for field, value in (("v_ent", 123.0), ("reward", -999.0)):
                evidence = {
                    name: np.asarray(item).copy() for name, item in payload.items()
                }
                evidence[field][1] = value
                cases.append((field, "sparse evidence hash mismatch", evidence))

            for field, value, message in (
                ("mean_selected_neighbors", -1.0, "invalid selected-neighbor series"),
                ("control_saturation_fraction", 1.1, "invalid saturation series"),
                ("mean_abs_control", -0.1, "invalid control series"),
                ("max_abs_control", setting.max_turn_rate * 1.01,
                 "invalid control series"),
            ):
                changed = {
                    name: np.asarray(item).copy() for name, item in payload.items()
                }
                changed[field][0] = value
                cases.append((field, message, changed))

            for index, (label, message, changed) in enumerate(cases):
                changed_path = root / "tampered-{}.npz".format(index)
                atomic_save_npz(changed_path, changed)
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, message):
                        validate_sensitivity_episode(changed_path, meta)

    def test_bundle_rejects_initial_state_mismatch_across_policies(self):
        config = load_sensitivity_config(
            OAT_CONFIG_PATH,
            settings=("baseline",),
            policies=("learned_deterministic", "pure_acs"),
            seeds=(0,),
        )
        setting = config.settings[0]
        resolved = config.fingerprint_input()
        resolved["macro_steps"] = 1
        spec = {
            "config": resolved,
            "backend": "cpu",
            "batch_size": 1,
            "checkpoint_package_sha256": "package",
            "effective_config_sha256_by_setting": {"baseline": "config"},
        }
        fingerprint = run_fingerprint(spec)
        manifest = {
            "run_id": "unit",
            "suite": SUITE_ID,
            "status": "completed",
            "fingerprint": fingerprint,
            "episode_count": 2,
            "checkpoint": {
                "id": "checkpoint",
                "archive": "checkpoint",
                "package_sha256": "package",
            },
            "spec": spec,
        }
        job = {
            "run_id": "unit",
            "fingerprint": fingerprint,
            "checkpoint_id": "checkpoint",
            "checkpoint_hash": "package",
            "horizon": 1,
            "policy_interval_seconds": 0.1,
            "device": "cpu",
            "batch_size": 1,
        }

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            for policy, first_position in (
                    ("learned_deterministic", 0.0), ("pure_acs", -1.0)):
                expected = _expected_meta(job, setting, policy, 0, "config")
                env = _FakeEnv(setting, first_position=first_position)
                accumulator = SparseAccumulator(env, setting, horizon=1)
                agents = setting.num_agents
                accumulator.append_macro(
                    0,
                    rewards=np.asarray([-1.0]),
                    masks=np.eye(agents, dtype=bool)[None, :, :],
                    controls=np.zeros((1, agents)),
                )
                meta = dict(expected)
                meta["criterion_of_record"] = False
                meta["official_c2_complete_horizon"] = False
                payload, _ = accumulator.payload(meta)
                atomic_save_npz(
                    episode_path(bundle, setting.setting_id, policy, 0), payload)

            with mock.patch(
                    "eval.sensitivity.runner.checkpoint_identity",
                    return_value={"package_sha256": "package"}):
                with self.assertRaisesRegex(
                        ValueError, "initial state mismatch across policies"):
                    validate_sensitivity_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
