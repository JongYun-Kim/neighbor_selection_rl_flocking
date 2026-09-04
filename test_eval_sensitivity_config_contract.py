"""Focused contract tests for the sensitivity configuration layer."""

import json
import tempfile
import unittest
from pathlib import Path

from eval.sensitivity.config import (
    OAT_CONFIG_PATH,
    OAT_FACTORS,
    POLICY_NAMES,
    REFINED_NEAR_ZERO_CONFIG_PATH,
    SensitivityConfigError,
    load_sensitivity_config,
)


class SensitivityConfigContractTest(unittest.TestCase):
    def test_oat_example_resolves_twenty_settings_with_one_baseline(self):
        config = load_sensitivity_config(OAT_CONFIG_PATH)

        self.assertEqual(config.suite, "oat")
        self.assertEqual(config.factors, OAT_FACTORS)
        self.assertEqual(config.policies, POLICY_NAMES)
        self.assertEqual(config.seeds, tuple(range(100)))
        self.assertEqual(config.macro_steps, 6000)
        self.assertEqual(config.policy_interval_seconds, 0.1)
        self.assertEqual(len(config.settings), 20)
        self.assertEqual(config.setting_ids.count("baseline"), 1)
        self.assertEqual(config.settings[0].memberships, OAT_FACTORS)
        self.assertEqual(
            {setting.factor for setting in config.settings[1:]},
            set(OAT_FACTORS),
        )

    def test_oat_physical_couplings_are_resolved(self):
        settings = load_sensitivity_config(OAT_CONFIG_PATH).setting_map()

        turn = settings["minimum_turn_radius_14p0625"]
        self.assertAlmostEqual(turn.max_turn_rate, turn.speed / 14.0625)
        interaction = settings["interaction_radius_120"]
        self.assertEqual(interaction.interaction_radius, 120.0)
        self.assertEqual(interaction.r0, 120.0)
        self.assertEqual(interaction.c2_proximity_radius, 120.0)
        gain = settings["acs_gain_multiplier_2"]
        self.assertEqual(gain.acs_lambda, 10.0)
        self.assertEqual(gain.acs_sigma, 2.0)
        self.assertEqual(gain.interaction_radius, 60.0)

    def test_refined_example_has_six_radii_and_expected_substeps(self):
        config = load_sensitivity_config(REFINED_NEAR_ZERO_CONFIG_PATH)

        self.assertEqual(config.suite, "refined_near_zero")
        self.assertEqual(config.factors, ("minimum_turn_radius",))
        self.assertEqual(len(config.settings), 6)
        self.assertEqual(
            tuple(item.substeps_per_policy_interval for item in config.settings),
            (2, 4, 8, 16, 32, 59),
        )
        for setting in config.settings:
            self.assertEqual(setting.mode, "refined")
            self.assertAlmostEqual(
                setting.dynamics_dt,
                0.1 / setting.substeps_per_policy_interval,
            )
            self.assertEqual(setting.max_turn_rate, setting.speed / setting.minimum_turn_radius)

    def test_filters_are_strict_subsets_and_use_configuration_order(self):
        config = load_sensitivity_config(
            OAT_CONFIG_PATH,
            factors=("minimum_turn_radius",),
            policies=("pure_acs", "learned_deterministic"),
            seeds=(99, 0),
        )

        self.assertEqual(config.factors, ("minimum_turn_radius",))
        self.assertEqual(config.setting_ids[0], "baseline")
        self.assertEqual(len(config.settings), 5)
        self.assertEqual(config.settings[0].memberships, OAT_FACTORS)
        self.assertEqual(
            config.policies, ("learned_deterministic", "pure_acs"))
        self.assertEqual(config.seeds, (0, 99))

        exact = load_sensitivity_config(
            OAT_CONFIG_PATH,
            factors=("minimum_turn_radius",),
            settings=("minimum_turn_radius_14p0625",),
        )
        self.assertEqual(exact.setting_ids, ("minimum_turn_radius_14p0625",))

        inferred = load_sensitivity_config(
            OAT_CONFIG_PATH,
            settings=("baseline", "minimum_turn_radius_14p0625"),
        )
        self.assertEqual(inferred.factors, ("minimum_turn_radius",))
        self.assertEqual(inferred.settings[0].memberships, OAT_FACTORS)

        baseline_only = load_sensitivity_config(
            OAT_CONFIG_PATH,
            settings=("baseline",),
        )
        self.assertEqual(baseline_only.factors, OAT_FACTORS)
        self.assertEqual(baseline_only.settings[0].memberships, OAT_FACTORS)

    def test_unknown_or_duplicate_filters_fail(self):
        invalid = (
            {"factors": ("unknown",)},
            {"settings": ("unknown",)},
            {"policies": ("unknown",)},
            {"seeds": (100,)},
            {"factors": ("num_agents", "num_agents")},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(SensitivityConfigError):
                    load_sensitivity_config(OAT_CONFIG_PATH, **arguments)

    def test_fingerprint_input_is_resolved_stable_and_path_independent(self):
        original = load_sensitivity_config(OAT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "renamed.yaml"
            copied.write_text(OAT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            replay = load_sensitivity_config(copied)

        self.assertEqual(original.fingerprint_input(), replay.fingerprint_input())
        json.dumps(original.fingerprint_input(), sort_keys=True, allow_nan=False)
        filtered = load_sensitivity_config(OAT_CONFIG_PATH, seeds=(0,))
        self.assertNotEqual(original.fingerprint_input(), filtered.fingerprint_input())

    def test_yaml_schema_rejects_duplicate_unknown_and_inconsistent_values(self):
        oat = OAT_CONFIG_PATH.read_text(encoding="utf-8")
        refined = REFINED_NEAR_ZERO_CONFIG_PATH.read_text(encoding="utf-8")
        cases = (
            oat.replace("suite: oat", "suite: oat\nsuite: oat", 1),
            oat.replace("suite: oat", "suite: oat\nunknown: true", 1),
            oat.replace(
                "num_agents: [10, 20, 40, 80]",
                "num_agents: [10, 40, 80]",
                1,
            ),
            refined.replace(
                "max_heading_change_per_substep: 0.05333333333333334",
                "max_heading_change_per_substep: 0.0",
                1,
            ),
            oat.replace("acs_lambda: 5.0", "acs_lambda: 1.0e308", 1),
            refined.replace("    - 14.0625", "    - 1.0e-308", 1),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, content in enumerate(cases):
                path = Path(temporary) / "invalid-{}.yaml".format(index)
                path.write_text(content, encoding="utf-8")
                with self.subTest(index=index):
                    with self.assertRaises(SensitivityConfigError):
                        load_sensitivity_config(path)


if __name__ == "__main__":
    unittest.main()
