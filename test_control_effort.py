"""Unit tests for full-artifact main-C2 control-effort analysis."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.analysis.control_effort import (
    HORIZON_METRICS,
    SUCCESS_METRICS,
    analyze_episode,
    compute_effort_metrics,
    discover_bundle_episodes,
    event_aligned,
    paired_vs_pure_acs,
    run_from_bundle,
)
from eval.artifacts import FULL_SCHEMA_VERSION


def _episode_arrays(horizon=299, n_agents=2):
    states = np.zeros((horizon + 1, n_agents, 5), dtype=np.float64)
    states[:, :, 0] = np.arange(n_agents, dtype=np.float64)
    states[:, :, 2] = 15.0
    controls = np.ones((horizon, n_agents), dtype=np.float64)
    pointer = np.tile(np.arange(n_agents, dtype=np.int16), (horizon, 1))
    binary = np.tile(np.eye(n_agents, dtype=bool), (horizon, 1, 1))
    spatial = np.sqrt(np.var(states[:, :, :2], axis=1).sum(axis=1))
    velocity = np.sqrt(np.var(states[:, :, 2:4], axis=1).sum(axis=1))
    phi = np.ones(horizon + 1, dtype=np.float32)
    components = np.ones(horizon + 1, dtype=np.float32)
    reward = np.concatenate([
        np.asarray([np.nan], dtype=np.float32),
        np.full(horizon, -1.0, dtype=np.float32),
    ])
    return states, controls, pointer, binary, spatial, velocity, phi, components, reward


def _write_new(path: Path, policy="pure_acs", seed=7):
    states, controls, pointer, binary, spatial, velocity, phi, components, reward = _episode_arrays()
    meta = {
        "N": 2, "policy": policy, "seed": seed, "horizon": 299,
        "dt": 0.2, "speed": 3.0, "r0": 60.0,
    }
    np.savez_compressed(
        path,
        schema_version=np.asarray(FULL_SCHEMA_VERSION),
        meta=np.asarray(json.dumps(meta)),
        agent_states=states,
        pointer_actions=pointer,
        binary_actions=binary,
        control_inputs=controls,
        s_ent=spatial,
        v_ent=velocity,
        phi=phi,
        n_comp_r0=components,
        reward=reward,
    )


def _write_legacy(path: Path, policy="best_deterministic", seed=7):
    states, controls, pointer, binary, spatial, velocity, phi, components, reward = _episode_arrays()
    np.savez_compressed(
        path,
        schema_version=np.asarray("1.1"),
        episode_index=np.asarray(0),
        seed=np.asarray(seed),
        action_seed=np.asarray(seed + 1),
        num_agents=np.asarray(2),
        horizon=np.asarray(299),
        policy_variant=np.asarray(policy),
        checkpoint_kind=np.asarray("best_01"),
        action_mode=np.asarray("deterministic"),
        dt=np.asarray(0.2),
        speed=np.asarray(3.0),
        r0=np.asarray(60.0),
        agent_states=states,
        pointer_actions=pointer,
        binary_actions=binary,
        control_inputs=controls,
        spatial_entropy=spatial[1:],
        velocity_entropy=velocity[1:],
        phi=phi[1:],
        proximity_components=components[1:],
        original_rewards=reward[1:],
    )


class ControlEffortMetricTest(unittest.TestCase):
    def test_l1_l2_pre_post_and_event_alignment(self):
        control = np.asarray([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0], [9.0, 9.0]])
        metrics, temporal = compute_effort_metrics(control, t_fire=3, dt=0.5, speed=2.0)
        self.assertEqual(metrics["l1_integral_to_c2"], 0.5 * (1.0 + 1.0 + 2.0))
        self.assertEqual(metrics["env_l1_cost_to_c2"], 4.0)
        self.assertEqual(metrics["l2_energy_to_c2"], 0.5 * (1.0 + 2.0 + 5.0))
        self.assertEqual(metrics["pre100_mean_abs"], 1.0)
        self.assertEqual(metrics["post100_mean_abs"], 5.5)

        aligned = event_aligned(temporal["l1_rate"], 3, np.asarray([-2, -1, 0, 1, 2]))
        np.testing.assert_allclose(aligned[:4], [1.0, 1.0, 2.0, 9.0])
        self.assertTrue(np.isnan(aligned[4]))

    def test_new_and_legacy_full_recompute_main_c2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_path, legacy_path = root / "new.npz", root / "legacy.npz"
            _write_new(new_path)
            _write_legacy(legacy_path)

            new = analyze_episode(new_path)
            legacy = analyze_episode(legacy_path)
            for item in (new, legacy):
                self.assertEqual(item["row"]["t_fire"], 299)
                self.assertEqual(item["row"]["success"], 1)
                self.assertEqual(item["row"]["J"], 299.0)
                self.assertAlmostEqual(item["row"]["env_l1_cost_to_c2"], 179.4)
                self.assertEqual(item["row"]["official_c2_complete_horizon"], 0)
            self.assertEqual(new["row"]["legacy"], 0)
            self.assertEqual(legacy["row"]["legacy"], 1)

    def test_manifest_and_episodes_csv_take_discovery_priority(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episodes" / "population" / "ck" / "N2" / "pure_acs" / "seed_00007.npz"
            episode.parent.mkdir(parents=True)
            _write_new(episode)
            (root / "ignored.npz").write_bytes(b"not-an-npz")
            (root / "manifest.json").write_text(
                json.dumps({"schema_version": "bundle", "spec": {"dt": 0.2, "speed": 3.0}}),
                encoding="utf-8")
            with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path"])
                writer.writeheader()
                writer.writerow({"path": str(episode.relative_to(root))})

            found_root, manifest, paths = discover_bundle_episodes(root)
            self.assertEqual(found_root, root.resolve())
            self.assertEqual(manifest["spec"]["dt"], 0.2)
            self.assertEqual(paths, [episode.resolve()])

            manifest_path = root / "manifest.json"
            running = json.loads(manifest_path.read_text(encoding="utf-8"))
            running["status"] = "running"
            manifest_path.write_text(json.dumps(running), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not 'completed'"):
                discover_bundle_episodes(root)

    def test_seed_pairing_requires_pure_acs_and_same_initial_state(self):
        base = {
            "N": 20, "seed": 3, "policy": "pure_acs", "success": 1,
            "initial_state_sha256": "same",
        }
        learned = {
            "N": 20, "seed": 3, "policy": "deterministic", "success": 1,
            "initial_state_sha256": "same",
        }
        for metric in SUCCESS_METRICS + HORIZON_METRICS:
            base[metric] = 2.0
            learned[metric] = 1.0
        rows = paired_vs_pure_acs([base, learned])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta_env_l1_cost_to_c2"], -1.0)
        self.assertEqual(rows[0]["percent_change_env_l1_cost_to_c2"], -50.0)

        learned["initial_state_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "initial states are not paired"):
            paired_vs_pure_acs([base, learned])

    def test_run_from_bundle_writes_csv_plots_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            checkpoint_id = "checkpoint_1"
            episode_rows = []
            for policy in ("pure_acs", "deterministic"):
                episode = (root / "episodes" / "population" / checkpoint_id
                           / "N2" / policy / "seed_00007.npz")
                episode.parent.mkdir(parents=True, exist_ok=True)
                _write_new(episode, policy=policy, seed=7)
                episode_rows.append({"path": str(episode.relative_to(root))})
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "evaluation-bundle-2.0",
                "suite": "population_v1",
                "status": "completed",
                "run_id": "test-run",
                "fingerprint": "abc123",
                "spec": {
                    "num_agents": [2], "policies": ["pure_acs", "deterministic"],
                    "seeds": [7], "horizon": 299, "dt": 0.2,
                    "speed": 3.0, "r0": 60.0,
                },
            }), encoding="utf-8")
            with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path"])
                writer.writeheader()
                writer.writerows(episode_rows)

            output = root / "custom-analysis"
            result = run_from_bundle(root, output=output)

            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["pairs"], 1)
            self.assertEqual(result["animations"], 0)
            self.assertTrue((output / "episode_control_effort.csv").is_file())
            self.assertTrue((output / "paired_vs_pure_acs.csv").is_file())
            self.assertTrue((output / "event_aligned.csv").is_file())
            self.assertTrue((output / "control_effort_manifest.json").is_file())
            self.assertTrue((output / "plots" / "effort_to_main_c2.png").is_file())
            self.assertTrue((output / "plots" / "paired_vs_pure_acs.png").is_file())
            self.assertTrue((output / "plots" / "pre_post_main_c2.png").is_file())
            self.assertTrue((output / "plots" / "event_aligned_control.png").is_file())
            analysis_manifest = json.loads(
                (output / "control_effort_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(analysis_manifest["protocol"]["id"], "main_c2_v1")
            self.assertFalse(Path(analysis_manifest["source_bundle"]).is_absolute())
            self.assertTrue(all(
                not Path(value).is_absolute()
                for key, value in analysis_manifest["outputs"].items()
                if key.endswith("_csv")
            ))
            with (output / "episode_control_effort.csv").open(
                    newline="", encoding="utf-8") as stream:
                first = next(csv.DictReader(stream))
            self.assertFalse(Path(first["path"]).is_absolute())
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
