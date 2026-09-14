"""Contract tests for pointer-less (mask) full-trajectory bundles."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.artifacts import (
    FULL_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    atomic_save_npz,
    load_episode,
    validate_full_episode,
)
from eval.analysis.core import (
    coerce_episode,
    compute_cutoff_radii,
    compute_mask_radii,
    rank_episode_cutoff_radii,
)

from test_eval_artifacts import canonical_payload


def mask_payload(horizon=2, n_agents=2):
    """The canonical payload with the pointer dropped and the mask id set."""
    payload = canonical_payload(horizon=horizon, n_agents=n_agents)
    payload.pop("pointer_actions")
    payload["schema_version"] = np.asarray(MASK_SCHEMA_VERSION)
    return payload


class MaskSchemaTest(unittest.TestCase):
    def test_mask_payload_is_canonical_and_validates_without_a_pointer(self):
        payload = mask_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed_00000.npz"
            atomic_save_npz(path, payload)
            view = load_episode(path)
            self.assertEqual(view.schema_version, MASK_SCHEMA_VERSION)
            self.assertTrue(view.full)
            self.assertFalse(view.legacy)
            self.assertNotIn("pointer_actions", view)
            report = validate_full_episode(view)
            self.assertEqual(report["schema_version"], MASK_SCHEMA_VERSION)
            self.assertEqual(report["n_agents"], 2)

    def test_pointer_payload_keeps_its_full_classification(self):
        payload = canonical_payload()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed_00000.npz"
            atomic_save_npz(path, payload)
            view = load_episode(path)
            self.assertEqual(view.schema_version, FULL_SCHEMA_VERSION)
            self.assertTrue(view.full)
            self.assertFalse(view.legacy)
            report = validate_full_episode(view)
            self.assertEqual(report["schema_version"], FULL_SCHEMA_VERSION)

    def test_legacy_and_sparse_classification_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = {
                "agent_states": np.zeros((3, 2, 5), dtype=np.float64),
                "pointer_actions": np.zeros((2, 2), dtype=np.int16),
                "binary_actions": np.tile(np.eye(2, dtype=bool), (2, 1, 1)),
                "control_inputs": np.zeros((2, 2), dtype=np.float64),
                "meta": np.asarray(json.dumps({"seed": 1})),
            }
            legacy_path = root / "legacy.npz"
            atomic_save_npz(legacy_path, legacy)
            legacy_view = load_episode(legacy_path)
            self.assertTrue(legacy_view.legacy)
            self.assertTrue(legacy_view.full)

            sparse_path = root / "sparse.npz"
            atomic_save_npz(sparse_path, {
                "phi": np.ones(3, dtype=np.float32),
                "meta": np.asarray(json.dumps({"seed": 1})),
            })
            sparse_view = load_episode(sparse_path)
            self.assertFalse(sparse_view.full)
            self.assertFalse(sparse_view.legacy)

    def test_mask_episode_coerces_without_a_pointer(self):
        payload = mask_payload()
        # Derive the recorded series instead of asserting the stub values.
        for name in ("s_ent", "v_ent", "phi"):
            payload.pop(name)
        episode = coerce_episode(payload, require_metrics=False)
        self.assertIsNone(episode.pointer_actions)
        self.assertEqual(episode.horizon, 2)
        self.assertEqual(episode.num_agents, 2)
        ranked = rank_episode_cutoff_radii(
            episode.states, episode.pointer_actions, episode.binary_actions
        )
        # Self-only masks select nobody, so every radius is the zero convention.
        self.assertTrue(np.array_equal(
            ranked["cutoff_radii"], np.zeros((2, 2))
        ))

    def test_mask_radii_match_pointer_radii_on_a_pointer_episode(self):
        rng = np.random.default_rng(11)
        horizon, n_agents = 7, 5
        states = np.zeros((horizon + 1, n_agents, 5), dtype=np.float64)
        states[:, :, :2] = rng.normal(scale=40.0, size=(horizon + 1, n_agents, 2))
        pointer = rng.integers(0, n_agents, size=(horizon, n_agents)).astype(np.int16)
        positions = states[:-1, :, :2]
        distances = np.linalg.norm(
            positions[:, :, None, :] - positions[:, None, :, :], axis=3
        )
        cutoff = np.take_along_axis(distances, pointer[:, :, None], axis=2)
        mask = distances <= cutoff
        ego = np.arange(n_agents)[None, :]
        self_rows = np.where(pointer == ego)
        mask[self_rows[0], self_rows[1], :] = False
        mask[self_rows[0], self_rows[1], self_rows[1]] = True

        pointer_radii = compute_cutoff_radii(states, pointer)
        mask_radii = compute_mask_radii(states, mask)
        self_selected = pointer == ego
        self.assertTrue(np.array_equal(
            mask_radii[~self_selected], pointer_radii[~self_selected]
        ))
        # A self pointer is a zero radius under both definitions.
        self.assertTrue(np.all(mask_radii[self_selected] == 0.0))
        self.assertTrue(np.all(pointer_radii[self_selected] == 0.0))


if __name__ == "__main__":
    unittest.main()
