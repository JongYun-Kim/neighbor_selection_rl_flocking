"""Tests for versioned full-trajectory artifacts and legacy readers."""

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.artifacts import (
    FULL_SCHEMA_VERSION,
    atomic_save_npz,
    atomic_write_json,
    json_compatible,
    load_episode,
    make_staging_directory,
    validate_full_episode,
)


def canonical_payload(horizon=2, n_agents=2):
    states = np.zeros((horizon + 1, n_agents, 5), dtype=np.float64)
    states[:, 1, 0] = 1.0
    pointer = np.tile(np.arange(n_agents, dtype=np.int16), (horizon, 1))
    binary = np.tile(np.eye(n_agents, dtype=bool), (horizon, 1, 1))
    meta = {
        "run_id": "unit", "policy": "deterministic", "seed": 7,
        "horizon": horizon, "num_agents": n_agents, "dt": 0.1,
        "r0": 60.0, "speed": 15.0,
    }
    payload = {
        "schema_version": np.asarray(FULL_SCHEMA_VERSION),
        "meta": np.asarray(json.dumps(meta)),
        "agent_states": states,
        "pointer_actions": pointer,
        "binary_actions": binary,
        "control_inputs": np.zeros((horizon, n_agents), dtype=np.float64),
        "s_ent": np.ones(horizon + 1, dtype=np.float32),
        "v_ent": np.zeros(horizon + 1, dtype=np.float32),
        "phi": np.ones(horizon + 1, dtype=np.float32),
        "n_comp_r0": np.ones(horizon + 1, dtype=np.int16),
        "reward": np.r_[np.nan, np.zeros(horizon)].astype(np.float32),
        "deg_agents": np.zeros((horizon, n_agents), dtype=np.int16),
        "t_fire": np.asarray(-1, dtype=np.int32),
        "success": np.asarray(0, dtype=np.int8),
        "J": np.asarray(np.nan, dtype=np.float64),
    }
    for name in ("s_ent_env", "v_ent_env"):
        payload[name] = np.r_[np.nan, np.zeros(horizon)].astype(np.float32)
    for name in ("nnd_mean", "nnd_max", "min_pair", "diam", "radius"):
        payload[name] = np.zeros(horizon + 1, dtype=np.float32)
    for name in ("n_comp_sel", "churn", "deg_mean"):
        payload[name] = np.full(horizon + 1, np.nan, dtype=np.float32)
    return payload


class ArtifactTest(unittest.TestCase):
    def test_atomic_outputs_follow_umask_and_preserve_existing_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "manifest.json"
            npz_path = root / "episode.npz"
            previous_umask = os.umask(0o002)
            try:
                atomic_write_json(json_path, {"status": "new"})
                atomic_save_npz(npz_path, {"value": np.asarray([1])})
                staging = make_staging_directory(root, ".stage.")
            finally:
                os.umask(previous_umask)

            self.assertEqual(stat.S_IMODE(json_path.stat().st_mode), 0o664)
            self.assertEqual(stat.S_IMODE(npz_path.stat().st_mode), 0o664)
            self.assertEqual(stat.S_IMODE(staging.stat().st_mode), 0o775)

            os.chmod(json_path, 0o640)
            atomic_write_json(json_path, {"status": "updated"})
            self.assertEqual(stat.S_IMODE(json_path.stat().st_mode), 0o640)

    def test_atomic_canonical_round_trip_and_expected_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "episode.npz"
            atomic_save_npz(path, canonical_payload())
            view = load_episode(path)

            self.assertFalse(view.legacy)
            self.assertTrue(view.full)
            self.assertEqual(view.schema_version, FULL_SCHEMA_VERSION)
            result = validate_full_episode(
                view, expected={"run_id": "unit", "seed": 7})
            self.assertEqual(result["horizon"], 2)
            self.assertEqual(result["n_agents"], 2)
            self.assertFalse(any(path.parent.glob(".tmp-*")))

    def test_validation_rejects_out_of_range_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            payload = canonical_payload()
            payload["pointer_actions"][0, 0] = 2
            atomic_save_npz(path, payload)
            with self.assertRaisesRegex(ValueError, "out of range"):
                validate_full_episode(load_episode(path))

    def test_legacy_full_aliases_are_read_only_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.npz"
            payload = canonical_payload()
            payload.pop("meta")
            payload["schema_version"] = np.asarray("1.1")
            payload["spatial_entropy"] = payload.pop("s_ent")[1:]
            payload["velocity_entropy"] = payload.pop("v_ent")[1:]
            payload["proximity_components"] = payload.pop("n_comp_r0")[1:]
            payload.pop("reward")
            payload["original_rewards"] = np.asarray([-1.0, -2.0])
            payload.update({
                "seed": np.asarray(11),
                "num_agents": np.asarray(2),
                "horizon": np.asarray(2),
                "policy_variant": np.asarray("deterministic"),
                "action_mode": np.asarray("deterministic"),
            })
            atomic_save_npz(path, payload)

            view = load_episode(path)
            self.assertTrue(view.legacy)
            self.assertTrue(view.full)
            self.assertEqual(view.meta["protocol_id"], "legacy-40-window")
            self.assertFalse(view.meta["official_c2_complete_horizon"])
            self.assertEqual(view.meta["seed"], 11)
            self.assertEqual(view["s_ent"].shape, (3,))
            self.assertEqual(view["reward"].shape, (3,))
            self.assertTrue(np.isnan(view["reward"][0]))
            np.testing.assert_array_equal(view["reward"][1:], [-1.0, -2.0])

    def test_sparse_main_artifact_is_not_mislabelled_full(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sparse.npz"
            atomic_save_npz(path, {
                "meta": np.asarray('{"seed":3}'),
                "phi": np.ones(4),
            })
            view = load_episode(path)
            self.assertFalse(view.full)
            self.assertEqual(view.meta["seed"], 3)

    def test_nonfinite_metadata_has_explicit_null_representation(self):
        converted = json_compatible({
            "J": np.nan, "limit": np.inf, "ok": np.float32(1.5),
            "nested": [np.int64(3)],
        })
        self.assertEqual(
            converted,
            {"J": None, "limit": None, "ok": 1.5, "nested": [3]},
        )
        self.assertNotIn("NaN", json.dumps(converted, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
