"""Tests for staged C2 checkpoint identity and promotion safeguards."""

import json
import tempfile
import unittest
from pathlib import Path

from eval.c2_suite import candidates_from_json, evaluate_c2_lane, explicit_candidate


def make_checkpoint(root: Path) -> Path:
    trial = root / "trial"
    checkpoint = trial / "checkpoint_000008"
    policy = checkpoint / "policies" / "default_policy"
    policy.mkdir(parents=True)
    (policy / "policy_state.pkl").write_bytes(b"state")
    (trial / "params.json").write_text(json.dumps({
        "model": {"custom_model": "dynamic_k_nn_neighbor_selector_rl",
                  "custom_model_config": {"scale_factor": 1.0}},
        "env_config": {"config": {"env": {
            "action_type": "dynamic_k_nn", "obs_position_scale": "legacy",
        }}},
    }), encoding="utf-8")
    return checkpoint


class C2SuiteTest(unittest.TestCase):
    def test_selection_json_revalidates_complete_checkpoint_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root)
            candidate = explicit_candidate(checkpoint)
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "checkpoints": [{
                    "resolved_path": str(checkpoint), "iteration": 8,
                    "checkpoint_tree_sha256": candidate["tree_sha256"],
                    "checkpoint_params_sha256": candidate["params_sha256"],
                    "checkpoint_state_sha256": candidate["state_sha256"],
                    "checkpoint_package_sha256": candidate["package_sha256"],
                    "roles": ["top_01"], "metrics": {},
                }],
            }), encoding="utf-8")
            loaded = candidates_from_json(selection)
            self.assertEqual(loaded[0]["package_sha256"], candidate["package_sha256"])

            params = checkpoint.parent / "params.json"
            params.write_text(params.read_text(encoding="utf-8") + "\n",
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "params_sha256 changed"):
                candidates_from_json(selection)

    def test_confirmation_rejects_nonexplicit_or_multiple_candidates(self):
        candidate = {
            "checkpoint": Path("missing"), "screen_roles": ["top_01"],
        }
        with self.assertRaisesRegex(ValueError, "explicitly chosen"):
            evaluate_c2_lane([candidate], "confirm", "unused", "run")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_c2_lane([candidate, candidate], "confirm", "unused", "run")


if __name__ == "__main__":
    unittest.main()
