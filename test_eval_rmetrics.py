"""Unit tests for the post-hoc R1-R6 metrics in eval.rmetrics."""

import unittest

import numpy as np

from eval.rmetrics import (boundary_tolerance, selection_arrays, self_checks,
                           window_slices, window_summary,
                           _selection_is_rank_faithful)


def _states(positions):
    """Lift a (T+1, N, 2) position series onto the stored (T+1, N, 5) layout."""
    positions = np.asarray(positions, dtype=np.float64)
    steps, agents = positions.shape[0], positions.shape[1]
    states = np.zeros((steps, agents, 5), dtype=np.float64)
    states[:, :, 0:2] = positions
    return states


def _line(offsets, steps):
    """Hold a fixed 1-D formation for ``steps`` action steps."""
    row = np.zeros((len(offsets), 2), dtype=np.float64)
    row[:, 0] = offsets
    return np.repeat(row[None, :, :], steps + 1, axis=0)


class HandBuiltEpisodeTest(unittest.TestCase):
    """Four agents on a line with a mask whose values are worked out by hand.

    Positions A=0, B=1, C=3, D=7 on the x axis.  The mask is

        A -> {B, D}     B -> {A}     C -> {}     D -> {C, B, A}
    """

    def setUp(self):
        self.states = _states(_line([0.0, 1.0, 3.0, 7.0], 1))
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, 0, [1, 3]] = True
        mask[0, 1, 0] = True
        mask[0, 3, [0, 1, 2]] = True
        self.mask = mask
        self.arrays, self.degrees, self.gap = selection_arrays(
            self.states, mask, r0=10.0)

    def test_degrees(self):
        np.testing.assert_array_equal(self.degrees[0], [2, 1, 0, 3])

    def test_r1_is_the_farthest_selected_neighbour(self):
        radius = self.arrays["r1"][0]
        np.testing.assert_allclose(radius[[0, 1, 3]], [7.0, 1.0, 7.0])
        self.assertTrue(np.isnan(radius[2]))
        np.testing.assert_allclose(self.arrays["r1_norm"][0, 0], 0.7)

    def test_r3_counts_the_unselected_agents_inside_the_disk(self):
        # A selects 2 of the 3 agents within 7.0; B and D select every agent
        # inside their own radius.
        nu = self.arrays["r3"][0]
        np.testing.assert_allclose(nu[[0, 1, 3]], [2.0 / 3.0, 1.0, 1.0])
        self.assertTrue(np.isnan(nu[2]))

    def test_r4_counts_selections_outside_the_k_nearest(self):
        # A picks D (rank 3) while skipping C (rank 2): one of two is outside.
        deviation = self.arrays["r4"][0]
        np.testing.assert_allclose(deviation[[0, 1, 3]], [0.5, 0.0, 0.0])
        self.assertTrue(np.isnan(deviation[2]))

    def test_r5_counts_by_distance_rank(self):
        np.testing.assert_array_equal(self.arrays["r5_sel_count"][0], [3, 1, 2])
        np.testing.assert_array_equal(self.arrays["r5_ref_count"][0], [3, 2, 1])

    def test_r6_asymmetry_and_non_reciprocity(self):
        # Edges A-B and A-D are mutual; B-D and C-D are one-directional.
        self.assertAlmostEqual(float(self.arrays["r6_asym"][0]), 0.5)
        nonrecip = self.arrays["r6_nonrecip"][0]
        np.testing.assert_allclose(nonrecip[[0, 1, 3]], [0.0, 0.0, 2.0 / 3.0])
        self.assertTrue(np.isnan(nonrecip[2]))

    def test_derived_dtypes_match_the_storage_contract(self):
        for key in ("r1", "r1_norm", "r3", "r4", "r6_asym", "r6_nonrecip"):
            self.assertEqual(self.arrays[key].dtype, np.float64, key)
        for key in ("r5_sel_count", "r5_ref_count"):
            self.assertEqual(self.arrays[key].dtype, np.int16, key)


class PointerLikeSelectionTest(unittest.TestCase):
    """A cutoff mask is rank-faithful: nu == 1, rank_dev == 0, sel == ref."""

    def setUp(self):
        rng = np.random.default_rng(20260914)
        steps, agents = 40, 7
        positions = rng.uniform(-50.0, 50.0, size=(steps + 1, agents, 2))
        self.states = _states(positions)
        distances = np.linalg.norm(
            positions[:steps, :, None, :] - positions[:steps, None, :, :],
            axis=-1)
        for index in range(agents):
            distances[:, index, index] = np.inf
        # every agent points at a rotating target (never itself, so the
        # cutoff is a real distance) and takes the closed disk
        offsets = 1 + (np.arange(steps) % (agents - 1))
        targets = (np.arange(agents)[None, :] + offsets[:, None]) % agents
        cutoff = np.take_along_axis(distances, targets[:, :, None], axis=-1)
        self.mask = distances <= cutoff
        self.arrays, self.degrees, self.gap = selection_arrays(
            self.states, self.mask, r0=60.0)

    def test_every_selection_is_a_nearest_set(self):
        chosen = self.degrees >= 1
        self.assertTrue(chosen.any())
        np.testing.assert_array_equal(self.arrays["r3"][chosen], 1.0)
        np.testing.assert_array_equal(self.arrays["r4"][chosen], 0.0)

    def test_rank_selection_matches_the_same_k_projection(self):
        np.testing.assert_array_equal(self.arrays["r5_sel_count"],
                                      self.arrays["r5_ref_count"])

    def test_r1_equals_the_cutoff_distance(self):
        from eval.analysis.core import compute_cutoff_radii

        steps, agents = self.mask.shape[0], self.mask.shape[1]
        offsets = 1 + (np.arange(steps) % (agents - 1))
        targets = (np.arange(agents)[None, :] + offsets[:, None]) % agents
        expected = compute_cutoff_radii(self.states, targets.astype(np.int16))
        np.testing.assert_array_equal(self.arrays["r1"], expected)


class FullCouplingTest(unittest.TestCase):
    """An all-ones mask is the ACS-FC degenerate point."""

    def setUp(self):
        rng = np.random.default_rng(7)
        steps, agents = 12, 5
        self.states = _states(rng.uniform(-30, 30, size=(steps + 1, agents, 2)))
        self.mask = np.ones((steps, agents, agents), dtype=bool)
        self.arrays, self.degrees, self.gap = selection_arrays(
            self.states, self.mask, r0=60.0)

    def test_degenerate_values(self):
        np.testing.assert_array_equal(self.degrees, 4)
        np.testing.assert_array_equal(self.arrays["r3"], 1.0)
        np.testing.assert_array_equal(self.arrays["r4"], 0.0)
        np.testing.assert_array_equal(self.arrays["r6_asym"], 0.0)
        np.testing.assert_array_equal(self.arrays["r6_nonrecip"], 0.0)


class EmptySelectionTest(unittest.TestCase):
    """k = 0 is NaN in the table convention, and no edge at all is NaN asym."""

    def setUp(self):
        self.states = _states(_line([0.0, 1.0, 3.0], 1))
        self.mask = np.zeros((1, 3, 3), dtype=bool)
        # the diagonal is set by the env and must be ignored everywhere
        self.mask[0, np.arange(3), np.arange(3)] = True
        self.arrays, self.degrees, self.gap = selection_arrays(
            self.states, self.mask, r0=60.0)

    def test_diagonal_is_ignored(self):
        np.testing.assert_array_equal(self.degrees[0], [0, 0, 0])

    def test_per_agent_metrics_are_nan(self):
        for key in ("r1", "r1_norm", "r3", "r4", "r6_nonrecip"):
            self.assertTrue(np.all(np.isnan(self.arrays[key][0])), key)

    def test_asymmetry_is_nan_without_edges(self):
        self.assertTrue(np.isnan(self.arrays["r6_asym"][0]))

    def test_rank_counts_are_zero(self):
        np.testing.assert_array_equal(self.arrays["r5_sel_count"][0], [0, 0])
        np.testing.assert_array_equal(self.arrays["r5_ref_count"][0], [0, 0])


class WindowAggregationTest(unittest.TestCase):
    def test_window_bounds(self):
        windows = window_slices(6000)
        self.assertEqual((windows["early"].start, windows["early"].stop), (0, 300))
        self.assertEqual((windows["ss"].start, windows["ss"].stop), (5700, 6000))
        self.assertEqual((windows["full"].start, windows["full"].stop), (0, 6000))

    def test_short_episode_windows_stay_inside(self):
        windows = window_slices(100)
        self.assertEqual((windows["early"].start, windows["early"].stop), (0, 100))
        self.assertEqual((windows["ss"].start, windows["ss"].stop), (0, 100))

    def test_summary_pools_agent_steps_and_skips_nan(self):
        steps, agents = 700, 4
        states = _states(_line([0.0, 1.0, 3.0, 7.0], steps))
        mask = np.zeros((steps, agents, agents), dtype=bool)
        mask[:, 0, [1, 3]] = True
        mask[:, 1, 0] = True
        mask[:, 3, [0, 1, 2]] = True
        arrays, degrees, _ = selection_arrays(states, mask, r0=10.0)
        summary = window_summary(arrays, degrees, agents)
        # k = [2, 1, 0, 3] every step, so the mean degree is 1.5 of 3
        self.assertAlmostEqual(summary["k_full"], 1.5)
        self.assertAlmostEqual(summary["k_norm_full"], 0.5)
        self.assertAlmostEqual(summary["k0_rate_full"], 0.25)
        # NaN agents drop out of the pooled mean: r1 over A, B, D only
        self.assertAlmostEqual(summary["r1_full"], (7.0 + 1.0 + 7.0) / 3.0)
        self.assertAlmostEqual(summary["r3_full"], (2.0 / 3.0 + 1.0 + 1.0) / 3.0)
        self.assertAlmostEqual(summary["r4_full"], 0.5 / 3.0)
        self.assertAlmostEqual(summary["r6_asym_full"], 0.5)
        for window in ("early", "ss"):
            self.assertAlmostEqual(summary[f"k_{window}"], summary["k_full"])


class BoundaryToleranceTest(unittest.TestCase):
    """Only a disagreement at the precision boundary is excused."""

    def _episode(self, skipped_at, selected_at):
        """A selects the agent at ``selected_at`` and skips the nearer one.

        B and C take everyone, so only A can violate rank faithfulness.
        """
        states = _states(_line([0.0, skipped_at, selected_at], 1))
        mask = np.zeros((1, 3, 3), dtype=bool)
        mask[0, 0, 2] = True            # A -> C only, skipping the nearer B
        mask[0, 1, [0, 2]] = True
        mask[0, 2, [0, 1]] = True
        arrays, degrees, gap = selection_arrays(states, mask, r0=60.0)
        return arrays, degrees, gap

    def test_a_near_tie_is_excused(self):
        # B is 1e-9 m nearer than the agent A actually selected: below the
        # resolution at which the env decides its own cutoff.
        arrays, degrees, gap = self._episode(10.0, 10.0 + 1e-9)
        self.assertAlmostEqual(float(arrays["r3"][0, 0]), 0.5)
        self.assertLess(float(gap[0, 0]),
                        float(boundary_tolerance(arrays['r1'][0, 0])))
        checks = self_checks(arrays, degrees, {}, True, 3, gap)
        self.assertTrue(checks["check_r3_one"])
        self.assertTrue(checks["check_r4_zero"])
        self.assertTrue(checks["check_r5_ref"])

    def test_a_real_violation_is_still_reported(self):
        # B is 5 m nearer and is genuinely skipped: no precision excuse.
        arrays, degrees, gap = self._episode(1.0, 6.0)
        self.assertAlmostEqual(float(arrays["r3"][0, 0]), 0.5)
        self.assertGreater(float(gap[0, 0]),
                           float(boundary_tolerance(arrays['r1'][0, 0])))
        checks = self_checks(arrays, degrees, {}, True, 3, gap)
        self.assertFalse(checks["check_r3_one"])
        self.assertFalse(checks["check_r4_zero"])
        self.assertFalse(checks["check_r5_ref"])

    def test_boundary_gap_is_infinite_without_a_selection(self):
        states = _states(_line([0.0, 1.0, 3.0], 1))
        mask = np.zeros((1, 3, 3), dtype=bool)
        _, degrees, gap = selection_arrays(states, mask, r0=60.0)
        np.testing.assert_array_equal(degrees[0], [0, 0, 0])
        self.assertTrue(np.all(np.isinf(gap[0])))


class RankFaithfulnessDispatchTest(unittest.TestCase):
    def test_arms_that_select_a_nearest_set_by_construction(self):
        self.assertTrue(_selection_is_rank_faithful("deterministic", True, None))
        self.assertTrue(_selection_is_rank_faithful("pure_acs", False, None))
        self.assertTrue(_selection_is_rank_faithful("knn12", False, None))
        self.assertTrue(_selection_is_rank_faithful("k19_L250_N20", False, 19))

    def test_a_learned_mask_makes_no_such_claim(self):
        self.assertFalse(
            _selection_is_rank_faithful("deterministic", False, None))
        self.assertFalse(_selection_is_rank_faithful("stochastic", False, None))


class ShapeValidationTest(unittest.TestCase):
    def test_mismatched_state_and_mask_lengths_are_rejected(self):
        states = _states(_line([0.0, 1.0], 5))
        mask = np.zeros((3, 2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            selection_arrays(states, mask, r0=60.0)

    def test_non_square_masks_are_rejected(self):
        states = _states(_line([0.0, 1.0], 1))
        with self.assertRaises(ValueError):
            selection_arrays(states, np.zeros((1, 2, 3), dtype=bool), r0=60.0)


if __name__ == "__main__":
    unittest.main()
