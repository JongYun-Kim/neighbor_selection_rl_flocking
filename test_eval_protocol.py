"""Regression tests for the main criterion-of-record C2 protocol."""
from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from eval.protocol import MAIN_C2_V1, first_fire_c2, judge_episode_c2


class MainC2ProtocolTest(unittest.TestCase):
    @staticmethod
    def passing_series(length=300):
        phi = np.full(length, 0.99, dtype=np.float64)
        spatial = np.full(length, 40.0, dtype=np.float64)
        components = np.ones(length, dtype=np.int64)
        return phi, spatial, components

    def test_main_spec_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            MAIN_C2_V1.phi_goal = 0.97

    def test_eval_c2_keeps_historical_public_aliases(self):
        from eval import eval_c2

        self.assertIs(eval_c2.t_fire_c2, first_fire_c2)
        self.assertEqual(eval_c2.PHI_GOAL, MAIN_C2_V1.phi_goal)
        self.assertEqual(eval_c2.W_A, MAIN_C2_V1.alignment_window)
        self.assertEqual(eval_c2.W, MAIN_C2_V1.stability_window)
        self.assertEqual(eval_c2.EPS, MAIN_C2_V1.spatial_band_epsilon)
        phi, spatial, components = self.passing_series()
        self.assertEqual(
            eval_c2.t_fire_c2(phi=phi, s=spatial, comp=components), 299)

    def test_strict_phi_and_spatial_band_boundaries(self):
        phi, spatial, components = self.passing_series()
        phi[-1] = MAIN_C2_V1.phi_goal
        self.assertEqual(first_fire_c2(phi, spatial, components), -1)

        phi[:] = 0.99
        # 150 values at 39 and 150 at 41 give mean=40 and p2p/mean=0.05
        # exactly, which must fail the strict '< 0.05' boundary.
        spatial[:150] = 39.0
        spatial[150:] = 41.0
        self.assertEqual(first_fire_c2(phi, spatial, components), -1)

    def test_alignment_uses_last_50_samples(self):
        phi, spatial, components = self.passing_series(length=301)
        phi[250] = MAIN_C2_V1.phi_goal
        # At index 299 the bad value remains inside [250, 299].  At index 300,
        # the 50-sample window is [251, 300], so the criterion first fires.
        self.assertEqual(first_fire_c2(phi, spatial, components), 300)

    def test_stability_uses_last_300_samples(self):
        phi, spatial, components = self.passing_series(length=301)
        components[0] = 2
        self.assertEqual(first_fire_c2(phi, spatial, components), 300)

    def test_t_plus_one_indexing_has_earliest_fire_at_299(self):
        phi, spatial, components = self.passing_series(length=299)
        self.assertEqual(first_fire_c2(phi, spatial, components), -1)

        phi, spatial, components = self.passing_series(length=300)
        self.assertEqual(first_fire_c2(phi, spatial, components), 299)

    def test_episode_j_uses_steps_one_through_fire_inclusive(self):
        phi, spatial, components = self.passing_series(length=301)
        reward = np.full(301, -2.0, dtype=np.float32)
        reward[0] = np.nan
        reward[300] = -1000.0  # after t_fire=299; must not contribute

        judged = judge_episode_c2(phi, spatial, components, reward)
        self.assertEqual(judged["t_fire"], 299)
        self.assertEqual(judged["success"], 1)
        self.assertEqual(judged["J"], 2.0 * 299)

        components[:] = 2
        failed = judge_episode_c2(phi, spatial, components, reward)
        self.assertEqual(failed["t_fire"], -1)
        self.assertEqual(failed["success"], 0)
        self.assertTrue(np.isnan(failed["J"]))


if __name__ == "__main__":
    unittest.main()
