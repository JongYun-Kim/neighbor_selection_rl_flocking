"""Unit tests for population scheduling and Dynamic-k trace invariants."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.artifacts import (
    FULL_SCHEMA_VERSION, SCALAR_SERIES, atomic_save_npz, canonical_json,
    json_compatible, load_episode,
)
from eval.population import (
    PROTOCOL_ID,
    SUITE_ID,
    _derived_full_series,
    _nanmedian,
    _state_hash,
    _verify_summary_files,
    derive_action_seed,
    parse_int_range,
    pure_acs_pointer,
    reconstruct_cutoff_mask,
    summarize,
    validate_population_episode,
)
from eval.protocol import judge_episode_c2


class _FakeEnv:
    num_agents_max = 4
    state = {"padding_mask": np.asarray([True, True, True, False])}
    rel_state = {
        "rel_agent_dists": np.asarray([
            [0.0, 2.0, 5.0, 0.0],
            [2.0, 0.0, 3.0, 0.0],
            [5.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }


class PopulationTest(unittest.TestCase):
    @staticmethod
    def canonical_episode(path: Path):
        horizon, n_agents = 2, 2
        dt, speed, rho, r0 = 0.1, 15.0, 1.0, 60.0
        states = np.zeros((horizon + 1, n_agents, 5), dtype=np.float64)
        states[0, :, 0] = [0.0, 10.0]
        states[:, :, 2] = speed
        for step in range(horizon):
            states[step + 1, :, :2] = (
                states[step, :, :2] + states[step, :, 2:4] * dt)
        pointers = np.tile(np.arange(n_agents, dtype=np.int16), (horizon, 1))
        binary = np.tile(np.eye(n_agents, dtype=bool), (horizon, 1, 1))
        controls = np.zeros((horizon, n_agents), dtype=np.float64)
        derived = _derived_full_series(
            states, binary, controls, dt=dt, speed=speed, rho=rho, r0=r0)
        series = {
            name: np.asarray(derived[name], dtype=np.float32)
            for name in SCALAR_SERIES
        }
        judgment = judge_episode_c2(
            series["phi"], series["s_ent"], series["n_comp_r0"],
            series["reward"])
        meta = {
            "run_id": "unit", "suite": SUITE_ID,
            "protocol_id": PROTOCOL_ID, "fingerprint": "fingerprint",
            "checkpoint_id": "checkpoint", "checkpoint_hash": "package",
            "num_agents": n_agents, "bound": 250.0, "seed": 3,
            "action_seed": derive_action_seed(3, n_agents),
            "policy": "deterministic", "horizon": horizon,
            "backend": "cpu", "batch_size": 1, "config_sha256": "config",
            "dt": dt, "speed": speed, "r0": r0, "rho": rho,
        }
        summary = {
            "run_id": meta["run_id"], "suite": meta["suite"],
            "protocol_id": meta["protocol_id"],
            "checkpoint_id": meta["checkpoint_id"],
            "checkpoint_hash": meta["checkpoint_hash"],
            "num_agents": n_agents, "bound": meta["bound"],
            "seed": meta["seed"], "action_seed": meta["action_seed"],
            "policy": meta["policy"], "horizon": horizon,
            **judgment,
            "phi_ss": _nanmedian(series["phi"][-300:]),
            "sigma_p_ss": _nanmedian(series["s_ent"][-300:]),
            "min_pair": float(np.nanmin(series["min_pair"])),
            "deg_ss": _nanmedian(series["deg_mean"][-300:]),
            "churn_ss": _nanmedian(series["churn"][-300:]),
            "n_comp_end": float(series["n_comp_r0"][-1]),
        }
        meta.update({
            "summary": json_compatible(summary),
            "initial_state_sha256": _state_hash(states[0]),
        })
        payload = {
            "schema_version": np.asarray(FULL_SCHEMA_VERSION),
            "meta": np.asarray(canonical_json(meta)),
            "agent_states": states, "pointer_actions": pointers,
            "binary_actions": binary, "control_inputs": controls,
            "deg_agents": derived["deg_agents"],
            "t_fire": np.asarray(judgment["t_fire"], dtype=np.int32),
            "success": np.asarray(judgment["success"], dtype=np.int8),
            "J": np.asarray(judgment["J"], dtype=np.float64),
            **series,
        }
        atomic_save_npz(path, payload)
        return payload, meta

    def test_seed_parser_and_action_seed_are_stable_and_n_specific(self):
        self.assertEqual(parse_int_range("0-3"), [0, 1, 2, 3])
        self.assertEqual(parse_int_range("9,2,5"), [9, 2, 5])
        with self.assertRaises(ValueError):
            parse_int_range("3-1")
        with self.assertRaises(ValueError):
            parse_int_range("1,1")
        self.assertEqual(derive_action_seed(4, 20), derive_action_seed(4, 20))
        self.assertNotEqual(derive_action_seed(4, 20), derive_action_seed(4, 40))

    def test_summary_validation_is_read_only_and_detects_tampering(self):
        rows = [
            {
                "num_agents": 20,
                "policy": policy,
                "seed": 0,
                "success": 1,
                "t_fire": 300 + index,
                "J": 10.0 + index,
                "path": "episodes/{}/seed_00000.npz".format(policy),
            }
            for index, policy in enumerate(("deterministic", "pure_acs"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            summarize(rows, bundle)
            paths = sorted((bundle / "summaries").glob("*.csv"))
            before = {path: path.read_bytes() for path in paths}

            _verify_summary_files(rows, bundle)
            self.assertEqual(
                before, {path: path.read_bytes() for path in paths}
            )

            aggregate = bundle / "summaries" / "aggregate.csv"
            aggregate.write_bytes(aggregate.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "differs from validated"):
                _verify_summary_files(rows, bundle)

            summarize(rows, bundle)
            _verify_summary_files(rows, bundle)

    def test_pure_acs_chooses_each_ego_farthest_active_agent(self):
        pointer = pure_acs_pointer(_FakeEnv())
        np.testing.assert_array_equal(pointer[:3], [2, 2, 0])
        # Padding entries are ignored by the environment; keeping zero is stable.
        self.assertEqual(pointer[3], 0)

    def test_mask_reconstruction_uses_pre_step_state_and_self_semantics(self):
        # Agent 1 and 2 are tied at radius 1 from agent 0. Selecting either
        # must include both; selecting ego itself must include ego only.
        states = np.zeros((2, 3, 5), dtype=np.float64)
        states[0, :, :2] = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
        states[1, :, :2] = [[0.0, 0.0], [100.0, 0.0], [-100.0, 0.0]]
        pointer = np.asarray([[1, 1, 0]], dtype=np.int16)

        mask = reconstruct_cutoff_mask(states, pointer)
        np.testing.assert_array_equal(mask[0, 0], [True, True, True])
        np.testing.assert_array_equal(mask[0, 1], [False, True, False])
        np.testing.assert_array_equal(mask[0, 2], [True, False, True])

    def test_mask_reconstruction_validates_shapes_and_indices(self):
        states = np.zeros((2, 2, 5))
        with self.assertRaisesRegex(ValueError, "out of range"):
            reconstruct_cutoff_mask(states, np.asarray([[0, 2]]))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            reconstruct_cutoff_mask(states[:-1], np.asarray([[0, 1]]))

    def test_deep_validation_rejects_summary_and_physical_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "episode.npz"
            payload, meta = self.canonical_episode(path)
            expected = {
                key: meta[key] for key in (
                    "run_id", "suite", "protocol_id", "fingerprint",
                    "checkpoint_id", "checkpoint_hash", "num_agents", "bound",
                    "seed", "action_seed", "policy", "horizon", "backend",
                    "batch_size", "config_sha256", "dt", "speed", "r0", "rho",
                )
            }
            validated = validate_population_episode(
                load_episode(path), expected, deep=True)
            self.assertEqual(validated["success"], 0)

            fake_meta = dict(meta)
            fake_meta["summary"] = dict(meta["summary"])
            fake_meta["summary"]["success"] = 1
            fake = dict(payload)
            fake["meta"] = np.asarray(canonical_json(fake_meta))
            fake_path = root / "fake-summary.npz"
            atomic_save_npz(fake_path, fake)
            with self.assertRaisesRegex(ValueError, "stored summary success"):
                validate_population_episode(load_episode(fake_path), expected, deep=True)

            physical = dict(payload)
            physical["control_inputs"] = payload["control_inputs"].copy()
            physical["control_inputs"][0, 0] = 1.0
            physical_path = root / "fake-control.npz"
            atomic_save_npz(physical_path, physical)
            with self.assertRaisesRegex(ValueError, "physical reward mismatch"):
                validate_population_episode(
                    load_episode(physical_path), expected, deep=True)


if __name__ == "__main__":
    unittest.main()
