"""Tests for the canonical checkpoint defaults exposed by evaluation CLIs."""

import argparse
import unittest
from pathlib import Path

from eval.__main__ import _device, _parser
from eval.defaults import DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_PACKAGE


class EvalDefaultsTest(unittest.TestCase):
    def test_default_checkpoint_has_the_loader_layout(self):
        self.assertEqual(DEFAULT_CHECKPOINT.parent, DEFAULT_CHECKPOINT_PACKAGE)
        self.assertTrue(DEFAULT_CHECKPOINT.is_dir())
        self.assertTrue((DEFAULT_CHECKPOINT_PACKAGE / "params.json").is_file())
        self.assertTrue(
            (DEFAULT_CHECKPOINT / "policies" / "default_policy"
             / "policy_state.pkl").is_file()
        )

    def test_unified_c2_uses_default_without_a_source_option(self):
        args = _parser().parse_args([
            "c2", "--lane", "dev", "--run-id", "default-c2",
        ])
        self.assertIsNone(args.candidates)
        self.assertEqual(args.checkpoint, DEFAULT_CHECKPOINT)

    def test_unified_population_uses_default_but_allows_override(self):
        default_args = _parser().parse_args([
            "population", "--run-id", "default-population",
        ])
        self.assertEqual(default_args.checkpoint, DEFAULT_CHECKPOINT)

        override = Path("another/checkpoint_000008")
        override_args = _parser().parse_args([
            "population", "--run-id", "override-population",
            "--checkpoint", str(override),
        ])
        self.assertEqual(override_args.checkpoint, override)

    def test_sensitivity_run_uses_default_checkpoint(self):
        args = _parser().parse_args([
            "sensitivity", "run", "--config", "sensitivity.yaml",
            "--run-id", "default-sensitivity",
        ])
        self.assertEqual(args.sensitivity_command, "run")
        self.assertEqual(args.checkpoint, DEFAULT_CHECKPOINT)

    def test_device_values_are_normalized_and_strict(self):
        population = _parser().parse_args([
            "population", "--run-id", "device-population",
            "--device", " GPU ",
        ])
        sensitivity = _parser().parse_args([
            "sensitivity", "run", "--config", "sensitivity.yaml",
            "--run-id", "device-sensitivity", "--device", "CUDA:2",
        ])
        self.assertEqual(population.device, "cuda")
        self.assertEqual(sensitivity.device, "cuda:2")
        with self.assertRaises(argparse.ArgumentTypeError):
            _device("cudafoo")


if __name__ == "__main__":
    unittest.main()
