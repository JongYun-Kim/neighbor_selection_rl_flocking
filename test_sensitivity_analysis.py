"""Focused tests for compact sensitivity summaries and figures."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eval.sensitivity.analysis import (
    FACTOR_VALUE_COLUMNS,
    PLOT_FILENAMES,
    _plot_view,
    analyze,
    verify_summary_tables,
    write_analysis,
    write_summary_tables,
)
from eval.stats import wilson


def _row(setting, mode, policy, seed, success, fire, value, factor="minimum_turn_radius"):
    return {
        "setting_id": setting,
        "factor": factor,
        "factor_value": value,
        "mode": mode,
        "policy": policy,
        "seed": seed,
        "horizon": 9,
        "success": int(success),
        "t_fire": fire if success else -1,
        "t_fire_seconds": fire / 10.0 if success else np.nan,
        "J": fire if success else np.nan,
        "path": "episodes/{}/{}/{}.npz".format(setting, policy, seed),
        "memberships": (
            '["num_agents","minimum_turn_radius","interaction_radius",'
            '"acs_gain_multiplier","initial_position_bound"]'
            if factor == "baseline" else factor
        ),
        "num_agents": 20,
        "minimum_turn_radius": 28.125 if factor == "baseline" else value,
        "interaction_radius": 60.0,
        "acs_gain_multiplier": 1.0,
        "initial_position_bound": 250.0,
    }


def _episodes():
    rows = []
    standard = {
        "pure_acs": ((True, 8), (False, -1)),
        "learned_deterministic": ((True, 5), (True, 9)),
        "learned_stochastic": ((True, 9), (False, -1)),
    }
    refined = {
        "pure_acs": ((True, 4), (True, 5)),
        "learned_deterministic": ((True, 6), (True, 7)),
        "learned_stochastic": ((True, 3), (True, 4)),
    }
    for policy, outcomes in standard.items():
        for seed, (success, fire) in enumerate(outcomes):
            rows.append(_row("turn_r14", "standard", policy, seed, success, fire, 14.0))
    for policy, outcomes in refined.items():
        for seed, (success, fire) in enumerate(outcomes):
            rows.append(_row("turn_r1", "refined", policy, seed, success, fire, 1.0))
    for policy, outcomes in standard.items():
        for seed, (success, fire) in enumerate(outcomes):
            rows.append(_row(
                "baseline", "standard", policy, seed, success, fire, 1.0,
                factor="baseline",
            ))
    return rows


class SensitivityAnalysisTest(unittest.TestCase):
    def test_summary_uses_wilson_and_horizon_plus_one_censoring(self):
        source = pd.DataFrame(_episodes())
        before = source.copy(deep=True)
        result = analyze(source)
        pd.testing.assert_frame_equal(source, before)

        aggregate = result.aggregate.set_index(["setting_id", "policy"])
        pure = aggregate.loc[("turn_r14", "pure_acs")]
        self.assertEqual(pure.success_count, 1)
        self.assertEqual(pure.failure_count, 1)
        self.assertAlmostEqual(pure.success_rate, 0.5)
        self.assertAlmostEqual(pure.restricted_time_mean_seconds, 0.9)
        failure_low, failure_high = wilson(1, 2)
        self.assertAlmostEqual(pure.success_wilson_95_low, 1.0 - failure_high)
        self.assertAlmostEqual(pure.success_wilson_95_high, 1.0 - failure_low)

        failed = result.episodes.loc[
            (result.episodes.setting_id == "turn_r14")
            & (result.episodes.policy == "pure_acs")
            & (result.episodes.seed == 1)
        ].iloc[0]
        self.assertAlmostEqual(failed.restricted_time_seconds, 1.0)

    def test_pairing_and_dominance_are_setting_seed_matched(self):
        result = analyze(_episodes())
        pair = result.paired.loc[
            (result.paired.setting_id == "turn_r14")
            & (result.paired.policy == "learned_deterministic")
            & (result.paired.seed == 1)
        ].iloc[0]
        self.assertAlmostEqual(pair.delta_restricted_time_seconds, -0.1)
        self.assertEqual(pair.outcome, "win")

        dominance = result.dominance.set_index(["setting_id", "policy"])
        self.assertTrue(
            bool(dominance.loc[("turn_r14", "learned_deterministic")].point_estimate_weak_dominance)
        )
        self.assertFalse(
            bool(dominance.loc[("turn_r14", "learned_stochastic")].point_estimate_weak_dominance)
        )
        self.assertFalse(
            bool(dominance.loc[("turn_r1", "learned_deterministic")].point_estimate_weak_dominance)
        )
        self.assertTrue(
            bool(dominance.loc[("turn_r1", "learned_stochastic")].point_estimate_weak_dominance)
        )

        overall = result.overall_dominance.set_index(["mode", "policy"])
        self.assertEqual(
            overall.loc[("standard", "learned_deterministic")].dominant_setting_count,
            2,
        )
        self.assertEqual(
            overall.loc[("standard", "learned_deterministic")].setting_count, 2
        )
        self.assertEqual(
            overall.loc[("refined", "learned_deterministic")].dominant_setting_count,
            0,
        )

        expanded = _plot_view(result.aggregate)
        baseline = expanded.loc[expanded.setting_id == "baseline"]
        self.assertEqual(set(baseline.factor), {"minimum_turn_radius"})
        self.assertEqual(len(baseline), 3)

        all_factors = _plot_view(result.aggregate, tuple(FACTOR_VALUE_COLUMNS))
        all_baseline = all_factors.loc[all_factors.setting_id == "baseline"]
        self.assertEqual(set(all_baseline.factor), set(FACTOR_VALUE_COLUMNS))
        self.assertEqual(len(all_baseline), 3 * len(FACTOR_VALUE_COLUMNS))

    def test_write_analysis_accepts_run_directory_and_writes_four_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            summaries = run / "summaries"
            summaries.mkdir(parents=True)
            pd.DataFrame(_episodes()).to_csv(summaries / "episodes.csv", index=False)

            output = write_analysis(run)
            analysis = run / "analysis"
            self.assertEqual(Path(output["output_dir"]), analysis)
            self.assertEqual(set(path.name for path in analysis.glob("*.png")),
                             set(PLOT_FILENAMES))
            self.assertTrue(all((analysis / name).stat().st_size > 0
                                for name in PLOT_FILENAMES))
            self.assertEqual(
                set(path.name for path in analysis.glob("*.csv")),
                {"aggregate.csv", "paired_vs_acs.csv", "dominance.csv",
                 "overall_dominance.csv"},
            )

    def test_manifest_factors_limit_canonical_baseline_plot_membership(self):
        rows = [row for row in _episodes() if row["setting_id"] == "baseline"]
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            summaries = run / "summaries"
            summaries.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(summaries / "episodes.csv", index=False)
            (run / "manifest.json").write_text(json.dumps({
                "spec": {"config": {"factors": ["minimum_turn_radius"]}},
            }), encoding="utf-8")

            result = analyze(run)
            self.assertEqual(result.factors, ("minimum_turn_radius",))
            expanded = _plot_view(result.aggregate, result.factors)
            self.assertEqual(set(expanded.factor), {"minimum_turn_radius"})

    def test_partial_summary_is_allowed_but_plot_requires_complete_pairs(self):
        rows = _episodes()
        unpaired = [
            row for row in rows
            if not (row["setting_id"] == "turn_r14"
                    and row["policy"] == "pure_acs" and row["seed"] == 1)
        ]
        result = analyze(unpaired)
        self.assertFalse(result.aggregate.empty)
        self.assertFalse(any(result.dominance.setting_id == "turn_r14"))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "complete identical-seed"):
                write_analysis(unpaired, output_dir=temporary)

    def test_summary_writer_and_read_only_verifier(self):
        rows = _episodes()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "run"
            paths = write_summary_tables(rows, bundle)
            self.assertEqual(set(paths), {
                "settings", "episodes", "aggregate", "paired_vs_acs", "dominance"
            })
            self.assertEqual(paths, verify_summary_tables(rows, bundle))
            aggregate = bundle / "summaries" / "aggregate.csv"
            aggregate.write_bytes(aggregate.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_summary_tables(rows, bundle)

    def test_rejects_duplicate_rows(self):
        rows = _episodes()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(rows + [dict(rows[0])])

    def test_rejects_time_that_is_not_on_policy_boundary(self):
        rows = _episodes()
        rows[0]["t_fire_seconds"] = 0.75
        with self.assertRaisesRegex(ValueError, "policy interval"):
            analyze(rows)


if __name__ == "__main__":
    unittest.main()
