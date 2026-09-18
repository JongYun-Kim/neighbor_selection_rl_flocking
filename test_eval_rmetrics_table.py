"""Table, figure, and stats-kit paths of eval.rmetrics / eval.stats.

These paths had no coverage before the round-1 audit (S5_audit_findings.md
F1-1 .. F1-6, F3-2).  Every test builds its own tiny derived tree.
"""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eval import rmetrics
from eval.stats import cvar10, mcnemar, paired_dj


def _row(policy, n, seed, **values):
    base = {
        "kind": "population", "arm": "x", "run_id": "r", "num_agents": str(n),
        "policy": policy, "seed": str(seed),
        "source_path": f"/src/N{n}/{policy}/seed_{seed:05d}.npz",
        "t_fire": "100", "success": "1", "J": "1.0", "min_pair_min": "0.5",
        "min_pair_min_post_c2": "5.0", "r0": "60.0", "dt": "0.1",
        "speed": "15.0", "u_max_used": "0.5333",
    }
    for window in rmetrics.WINDOW_NAMES:
        for name in rmetrics.R_COLUMNS:
            base[f"{name}_{window}"] = "0.25"
    for name in rmetrics.EFFORT_COLUMNS:
        base[name] = "2.0"
    base.update({"rms_to_c2_or_horizon": "1.0", "sat_frac_to_c2": "0.0",
                 "sat_frac_full": "0.0"})
    for name in rmetrics.CHECK_COLUMNS:
        base[name] = "True"
    base.update({k: str(v) for k, v in values.items()})
    return base


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


class MedianIqr(unittest.TestCase):
    def test_drops_none_and_nan_and_counts_the_rest(self):
        median, q1, q3, n = rmetrics._median_iqr([3.0, None, 1.0, float("nan"), 2.0])
        self.assertEqual((median, q1, q3, n), (2.0, 1.5, 2.5, 3))

    def test_empty_is_nan_with_zero_count(self):
        median, q1, q3, n = rmetrics._median_iqr([None, float("nan")])
        self.assertTrue(all(math.isnan(v) for v in (median, q1, q3)))
        self.assertEqual(n, 0)

    def test_a_failure_sentinel_is_finite_and_therefore_not_dropped_here(self):
        # the gate lives in _cell_value, not here
        self.assertEqual(rmetrics._median_iqr([-1.0, 5.0])[3], 2)


class TableMetrics(unittest.TestCase):
    def test_identifiers_constants_flags_and_success_are_not_metrics(self):
        metrics = rmetrics._table_metrics([_row("a", 20, 0)])
        for name in ("num_agents", "r0", "dt", "speed", "u_max_used", "success",
                     "kind", "arm", "seed", "policy", "source_path", "run_id",
                     *rmetrics.CHECK_COLUMNS):
            self.assertNotIn(name, metrics)
        for name in ("t_fire", "J", "min_pair_min", "min_pair_min_post_c2",
                     "r4_full", "l2_energy_to_c2"):
            self.assertIn(name, metrics)

    def test_t_fire_is_gated_on_success(self):
        failed = _row("a", 20, 0, t_fire=-1, success=0)
        self.assertTrue(math.isnan(rmetrics._cell_value(failed, "t_fire")))
        self.assertEqual(rmetrics._cell_value(failed, "J"), 1.0)
        self.assertEqual(rmetrics._cell_value(_row("a", 20, 0), "t_fire"), 100.0)


class RunTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _table(self, arms, pairs=()):
        out = self.root / "table.csv"
        result = rmetrics.run_table(arms, list(pairs), out)
        with out.open() as stream:
            return result, list(csv.DictReader(stream))

    def test_one_cell_per_arm_and_paired_rows_keyed_by_n_and_seed(self):
        left = _write_csv(self.root / "l.csv", [
            _row("det", 20, s, r4_full=0.3 + 0.01 * s) for s in range(4)])
        right = _write_csv(self.root / "r.csv", [
            _row("acs", 20, s, r4_full=0.0) for s in range(4)] )
        result, rows = self._table({"L": left, "R": right}, [("L", "R")])
        self.assertEqual(result["matched_pairs"], {"L:R": 4})
        self.assertEqual(result["cells"], {"L": ("det", "20"), "R": ("acs", "20")})
        paired = {r["metric"]: r for r in rows if r["comparison"] == "paired_difference"}
        self.assertAlmostEqual(float(paired["r4_full"]["median"]), 0.315)
        self.assertEqual(paired["r4_full"]["n"], "4")
        self.assertNotIn("num_agents", paired)

    def test_an_arm_spanning_several_cells_is_rejected(self):
        mixed = _write_csv(self.root / "m.csv", [
            _row("det", 20, 0), _row("acs", 20, 0), _row("det", 40, 0)])
        with self.assertRaisesRegex(ValueError, "3 \\(policy, num_agents\\) cells"):
            self._table({"M": mixed})

    def test_repeated_seeds_across_n_do_not_collapse_when_sliced(self):
        # same seed in two N cells: each slice pairs on its own N
        a20 = _write_csv(self.root / "a20.csv", [_row("det", 20, s, k_full=2) for s in range(3)])
        b20 = _write_csv(self.root / "b20.csv", [_row("det", 20, s, k_full=5) for s in range(3)])
        b40 = _write_csv(self.root / "b40.csv", [_row("det", 40, s, k_full=9) for s in range(3)])
        result, rows = self._table({"A": a20, "B": b20}, [("A", "B")])
        self.assertEqual(result["matched_pairs"], {"A:B": 3})
        with self.assertRaisesRegex(ValueError, "no shared \\(num_agents, seed\\)"):
            self._table({"A": a20, "B": b40}, [("A", "B")])

    def test_disjoint_seed_ranges_fail_instead_of_writing_nan(self):
        pop = _write_csv(self.root / "p.csv", [_row("det", 20, s) for s in range(3)])
        ref = _write_csv(self.root / "k.csv", [_row("k12", 20, s) for s in range(1000, 1003)])
        with self.assertRaises(ValueError):
            self._table({"P": pop, "K": ref}, [("P", "K")])
        self.assertFalse((self.root / "table.csv").exists())

    def test_failed_episodes_leave_t_fire_but_keep_min_pair(self):
        arm = _write_csv(self.root / "f.csv", [
            _row("k12", 20, 0, t_fire=515, success=1, min_pair_min=0.6),
            _row("k12", 20, 1, t_fire=-1, success=0, min_pair_min=0.4),
            _row("k12", 20, 2, t_fire=490, success=1, min_pair_min=0.8)])
        _, rows = self._table({"K": arm})
        by = {r["metric"]: r for r in rows}
        self.assertEqual((float(by["t_fire"]["median"]), by["t_fire"]["n"]), (502.5, "2"))
        self.assertEqual((float(by["min_pair_min"]["median"]), by["min_pair_min"]["n"]), (0.6, "3"))
        self.assertNotIn("success", by)

    def test_unknown_pair_arm_is_rejected(self):
        arm = _write_csv(self.root / "a.csv", [_row("det", 20, 0)])
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            self._table({"A": arm}, [("A", "Z")])


class WriteSummary(unittest.TestCase):
    def test_columns_and_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_row("stochastic", 40, 9), _row("deterministic", 10, 10),
                    _row("deterministic", 10, 2)]
            for row in rows:
                row["num_agents"] = int(row["num_agents"])
            path = rmetrics.write_summary(rows, Path(tmp) / "s.csv")
            with path.open() as stream:
                written = list(csv.DictReader(stream))
        header = list(written[0])
        self.assertEqual(header[:16], [
            "kind", "arm", "run_id", "num_agents", "policy", "seed", "source_path",
            "t_fire", "success", "J", "min_pair_min", "min_pair_min_post_c2",
            "r0", "dt", "speed", "u_max_used"])
        self.assertEqual(header[-len(rmetrics.CHECK_COLUMNS):], list(rmetrics.CHECK_COLUMNS))
        # (policy, int N, str seed): "10" < "2" as strings
        self.assertEqual([(r["policy"], r["seed"]) for r in written],
                         [("deterministic", "10"), ("deterministic", "2"), ("stochastic", "9")])


class PostC2MinPair(unittest.TestCase):
    def test_window_starts_at_the_declaration_state(self):
        min_pair = np.array([0.03, 0.5, 9.0, 7.0, 8.0])
        self.assertEqual(rmetrics.post_c2_min_pair(min_pair, 2, 1), 7.0)
        self.assertEqual(rmetrics.post_c2_min_pair(min_pair, 0, 1), 0.03)

    def test_failures_are_nan(self):
        self.assertTrue(math.isnan(rmetrics.post_c2_min_pair(np.ones(3), -1, 0)))
        self.assertTrue(math.isnan(rmetrics.post_c2_min_pair(np.ones(3), 1, 0)))


def _derived_tree(root, cells, horizon=6):
    """cells: list of (policy, n, seed, sel_weight).  Returns rows for a CSV."""
    rows = []
    for policy, n, seed, weight in cells:
        row = _row(policy, n, seed)
        path = root / "episodes" / f"N{n}" / policy / f"seed_{seed:05d}.rmetrics.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        sel = np.full((horizon, n - 1), weight * n, dtype=np.float64)
        ref = np.full((horizon, n - 1), 1.0 * n, dtype=np.float64)
        meta = {"num_agents": n, "horizon": horizon, "source_path": row["source_path"]}
        np.savez(path, r5_sel_count=sel, r5_ref_count=ref, meta=json.dumps(meta))
        rows.append(row)
    return rows


class RankCurves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = _derived_tree(self.root, [
            ("det", 4, 0, 0.2), ("det", 4, 1, 0.4),
            ("acs", 4, 0, 1.0), ("det", 6, 0, 0.9)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_rows_of_the_arm_csv_are_pooled(self):
        det4 = [r for r in self.rows if r["policy"] == "det" and r["num_agents"] == "4"]
        csv_path = _write_csv(self.root / "slice_det_N4.csv", det4)
        sel, ref = rmetrics._rank_curves(csv_path, "full")
        self.assertEqual(sel.shape, (3,))
        np.testing.assert_allclose(sel, 0.3)   # mean of 0.2 and 0.4, acs excluded
        np.testing.assert_allclose(ref, 1.0)

    def test_mixed_n_in_one_arm_is_an_error(self):
        csv_path = _write_csv(self.root / "all.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "mix num_agents"):
            rmetrics._rank_curves(csv_path, "full")

    def test_a_slice_outside_the_tree_is_an_error_not_a_missing_curve(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            csv_path = _write_csv(Path(elsewhere) / "slice.csv", self.rows[:2])
            with self.assertRaisesRegex(ValueError, "none of its 2 episodes"):
                rmetrics._rank_curves(csv_path, "full")


class ArmStyle(unittest.TestCase):
    def test_every_report_arm_name_lands_on_its_palette_family(self):
        expected = {
            "dknn": "#2E7D32", "full_NN": "#C62828", "k12": "#B07A06",
            "k19": "#37474F", "ACS_FC": "#37474F", "dknn_sto": "#2E7D32",
            "fnn_acs": "#37474F", "fnn_knn12": "#B07A06",
        }
        for name, color in expected.items():
            style = rmetrics.arm_style(name)
            self.assertEqual(style["color"], color, name)
            self.assertEqual(style["suffix"], "", name)

    def test_only_stochastic_decoding_is_dashed(self):
        for name in ("dknn", "full_NN", "k19", "fnn_acs", "fnn_knn12"):
            self.assertEqual(rmetrics.arm_style(name)["linestyle"], "-", name)
        for name in ("dknn_sto", "fnn_sto", "stochastic_N20"):
            self.assertEqual(rmetrics.arm_style(name)["linestyle"], "--", name)

    def test_an_unmapped_arm_is_grey_and_says_so(self):
        style = rmetrics.arm_style("mystery")
        self.assertEqual(style["color"], rmetrics.UNMAPPED_ARM_COLOR)
        self.assertEqual(style["family"], "unmapped")
        self.assertEqual(style["suffix"], " (unmapped)")


class EffortPoints(unittest.TestCase):
    def test_episodes_that_never_reached_c2_are_counted_not_dropped(self):
        rows = [_row("det", 20, seed) for seed in range(5)]
        for row in rows[:2]:
            row.update({"t_fire": "-1", "l2_energy_to_c2": "nan"})
        t_fire, energy, drawn, total = rmetrics._effort_points(rows)
        self.assertEqual((drawn, total), (3, 5))
        self.assertEqual(t_fire.size, 3)
        self.assertTrue(np.isfinite(energy).all())


class RunFigures(unittest.TestCase):
    def test_writes_the_four_figures_from_sliced_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = _derived_tree(root, [("det", 4, 0, 0.2), ("det", 4, 1, 0.4),
                                        ("acs", 4, 0, 1.0), ("acs", 4, 1, 1.0)])
            arms = {}
            for policy in ("det", "acs"):
                arms[policy] = _write_csv(
                    root / f"slice_{policy}_N4.csv",
                    [r for r in rows if r["policy"] == policy])
            result = rmetrics.run_figures(arms, root / "figs")
            names = sorted(Path(p).name for p in result["figures"])
        self.assertEqual(names, sorted([
            "r5_rank_selection.png", "r1_r2_early_vs_ss.png",
            "effort_vs_t_fire.png", "k0_rate.png"]))


class StatsKit(unittest.TestCase):
    def test_cvar10_is_the_mean_of_the_worst_ceil_ten_percent(self):
        j = np.arange(1, 24, dtype=float)          # 23 values -> ceil(2.3) = 3
        self.assertEqual(cvar10(j), (21 + 22 + 23) / 3)
        self.assertEqual(cvar10([4.0]), 4.0)
        self.assertTrue(math.isnan(cvar10([])))

    def test_mcnemar_counts_discordant_pairs_in_the_kit_convention(self):
        pol = pd.DataFrame({"success": [1, 0, 0, 1, 1]})
        ref = pd.DataFrame({"success": [1, 1, 1, 0, 1]})
        b, c, p = mcnemar(pol, ref)
        self.assertEqual((b, c), (2, 1))       # b = pol fails where ref succeeds
        self.assertAlmostEqual(p, 1.0)

    def test_paired_dj_uses_co_success_pairs_only(self):
        pol = pd.DataFrame({"success": [1] * 8, "J": [12, 11, 13, 14, 15, 16, 17, 1e9]})
        ref = pd.DataFrame({"success": [1] * 7 + [0], "J": [10] * 8})
        out = paired_dj(pol, ref)
        self.assertEqual(out["n_pair"], 7)
        self.assertEqual(out["med"], 4.0)
        self.assertEqual(out["worse"], 7)
        self.assertLess(out["sign_p"], 0.05)

    def test_paired_dj_survives_identical_arms(self):
        # knn12 == pure_acs at N=10 gives an all-zero difference vector
        pol = pd.DataFrame({"success": [1] * 10, "J": np.arange(10.0)})
        out = paired_dj(pol, pol.copy())
        self.assertEqual((out["n_pair"], out["med"], out["worse"]), (10, 0.0, 0))
        self.assertTrue(math.isnan(out["wilcox_p"]) and math.isnan(out["sign_p"]))

    def test_paired_dj_too_few_pairs(self):
        pol = pd.DataFrame({"success": [1, 1, 0], "J": [1.0, 2.0, 3.0]})
        self.assertEqual(paired_dj(pol, pol)["n_pair"], 2)


if __name__ == "__main__":
    unittest.main()
