"""Sparse control-effort analysis of the sensitivity lane.

The sensitivity bundles store per-step control diagnostics instead of the
``control_inputs`` matrix, so these tests pin three things: the L1 numbers are
on the same scale as the population lane's (``PopulationParity``), every
contract violation raises instead of warning (``EpisodeGuards``), and no L2 or
RMS column can reappear anywhere (``NoL2Columns``).  ``UnifiedCliWiring`` pins
that ``python -m eval sensitivity effort`` reaches the same functions as the
module's own CLI.  Each test builds its own tiny bundle; the two checks against
the recorded lanes skip when the archive is absent.
"""
import contextlib
import csv
import io
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.analysis import control_effort, sensitivity_effort as se
from eval.artifacts import canonical_json, load_episode
from eval.protocol import judge_episode_c2


HORIZON = 400
U_MAX = 0.5333333333333333
POPULATION_EPISODE = Path(
    "test_results/evaluation/r1-fullnn-pop/population/episodes/population_v1"
    "/checkpoint_000100/N20/deterministic/seed_00000.npz")
POPULATION_BUNDLE = Path("test_results/evaluation/r1-fullnn-pop/population")
RECORDED_OUTPUTS = (
    Path("test_results/evaluation/r1-fullnn-sens/sensitivity/analysis/control_effort"),
    Path("test_results/evaluation/s1night-dknn-sens/sensitivity/analysis/control_effort"),
    Path("test_results/evaluation/s7-sens-effort"),
)


def _read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _state_series(success):
    """C2 series that fire at index 299 (success) or never (failure)."""
    phi = np.full(HORIZON + 1, 0.99 if success else 0.10, dtype=np.float32)
    s_ent = np.full(HORIZON + 1, 40.0, dtype=np.float32)
    v_ent = np.zeros(HORIZON + 1, dtype=np.float32)
    n_comp = np.ones(HORIZON + 1, dtype=np.int16)
    reward = np.full(HORIZON + 1, -0.5, dtype=np.float64)
    reward[0] = np.nan
    return {"phi": phi, "s_ent": s_ent, "v_ent": v_ent, "n_comp_r0": n_comp,
            "reward": reward}


def _meta(**overrides):
    meta = {
        "run_id": "unit-run", "suite": "sensitivity_v1",
        "protocol_id": "main_c2_v1", "checkpoint_id": "checkpoint_000001",
        "checkpoint_hash": "0" * 64, "setting_id": "baseline",
        "factor": "baseline", "factor_value": 1.0,
        "memberships": ["num_agents", "minimum_turn_radius"],
        "mode": "standard", "policy": "learned_deterministic", "seed": 0,
        "horizon": HORIZON, "policy_interval_seconds": 0.1,
        "substeps_per_policy_interval": 1, "dynamics_dt": 0.1,
        "num_agents": 20, "speed": 15.0, "max_turn_rate": U_MAX,
        "minimum_turn_radius": 28.125, "interaction_radius": 60.0,
        "acs_gain_multiplier": 1.0, "initial_position_bound": 250.0,
        "initial_state_sha256": "a" * 64,
    }
    meta.update(overrides)
    return meta


def _write_episode(path, *, success=True, level=0.2, meta=None,
                   schema=se.SPARSE_SCHEMA, outcome=None, arrays=None):
    meta = _meta() if meta is None else meta
    series = _state_series(success)
    payload = dict(series)
    payload.update({
        "mean_abs_control": np.full(HORIZON, level, dtype=np.float32),
        "max_abs_control": np.full(HORIZON, level * 1.5, dtype=np.float32),
        "control_saturation_fraction": np.full(HORIZON, 0.25, dtype=np.float32),
        "initial_agent_states": np.zeros((int(meta["num_agents"]), 5),
                                         dtype=np.float64),
    })
    payload.update(arrays or {})
    judgment = judge_episode_c2(payload["phi"], payload["s_ent"],
                               payload["n_comp_r0"], payload["reward"])
    judgment.update(outcome or {})
    payload.update({
        "schema_version": np.asarray(schema),
        "meta": np.asarray(canonical_json(meta)),
        "t_fire": np.asarray(judgment["t_fire"], dtype=np.int32),
        "success": np.asarray(judgment["success"], dtype=np.int8),
        "J": np.asarray(judgment["J"], dtype=np.float64),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path, meta, payload


def _build_bundle(root, *, run_id="unit-run", checkpoint="checkpoint_000001",
                  checkpoint_hash="0" * 64, settings=("baseline", "num_agents_40"),
                  failing=("num_agents_40", "pure_acs"), seeds=(0, 1, 2)):
    """A two-setting bundle whose PureACS cell of one setting never converges."""
    root = Path(root)
    rows = []
    for setting_id in settings:
        for policy in ("learned_deterministic", "pure_acs"):
            for seed in seeds:
                success = not (setting_id == failing[0] and policy == failing[1])
                n_agents = 40 if setting_id == "num_agents_40" else 20
                meta = _meta(
                    run_id=run_id, checkpoint_id=checkpoint,
                    checkpoint_hash=checkpoint_hash, setting_id=setting_id,
                    factor=("baseline" if setting_id == "baseline"
                            else "num_agents"),
                    factor_value=(1.0 if setting_id == "baseline" else 40.0),
                    memberships=(["num_agents", "minimum_turn_radius"]
                                 if setting_id == "baseline" else ["num_agents"]),
                    policy=policy, seed=seed, num_agents=n_agents,
                    initial_state_sha256=f"{setting_id}-{seed}")
                path = (root / "episodes" / se.SUITE_ID / setting_id / policy
                        / f"seed_{seed:05d}.npz")
                _, _, payload = _write_episode(
                    path, success=success, level=0.2 + 0.05 * seed, meta=meta)
                rows.append({
                    "run_id": run_id, "setting_id": setting_id,
                    "policy": policy, "seed": seed,
                    "t_fire": int(payload["t_fire"]),
                    "success": int(payload["success"]),
                    "mean_abs_control": repr(float(
                        np.mean(payload["mean_abs_control"]))),
                    "max_abs_control": repr(float(
                        np.max(payload["max_abs_control"]))),
                    "control_saturation_fraction": repr(float(
                        np.mean(payload["control_saturation_fraction"]))),
                    "path": str(path.relative_to(root)),
                })
    (root / "summaries").mkdir(parents=True, exist_ok=True)
    with (root / "summaries" / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": se.BUNDLE_SCHEMA, "suite": se.SUITE_ID,
        "status": "completed", "run_id": run_id, "fingerprint": "f" * 64,
        "checkpoint": {"id": checkpoint, "package_sha256": checkpoint_hash},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class SparseEffortMetrics(unittest.TestCase):
    def setUp(self):
        self.m1 = np.full(HORIZON, 0.2, dtype=np.float32)
        self.peak = np.full(HORIZON, 0.4, dtype=np.float32)
        self.sat = np.full(HORIZON, 0.25, dtype=np.float32)

    def test_integrals_use_the_policy_interval_and_stop_at_c2(self):
        metrics, temporal = se.compute_sparse_effort_metrics(
            self.m1, self.peak, self.sat, t_fire=300, dt=0.1, speed=15.0,
            u_max=U_MAX)
        self.assertAlmostEqual(metrics["l1_integral_to_c2"], 6.0, places=6)
        self.assertAlmostEqual(metrics["env_l1_cost_to_c2"], 90.0, places=5)
        self.assertAlmostEqual(metrics["mean_abs_to_c2"], 0.2, places=7)
        self.assertAlmostEqual(metrics["sat_seconds_to_c2"], 7.5, places=6)
        self.assertAlmostEqual(metrics["sat_frac_to_c2"], 0.25, places=7)
        self.assertAlmostEqual(metrics["peak_abs_to_c2"], 0.4, places=7)
        self.assertEqual(sorted(temporal), ["l1_rate", "peak_rate", "sat_rate"])

    def test_whole_episode_reductions_keep_the_stored_dtype(self):
        # the bundle's own summary is float(np.mean(float32 series)); matching
        # it bit for bit is what makes the CSV bridge exact
        noisy = (np.arange(HORIZON, dtype=np.float32) % 7) / 100.0
        metrics, _ = se.compute_sparse_effort_metrics(
            noisy, self.peak, self.sat, t_fire=-1, dt=0.1, speed=15.0,
            u_max=U_MAX)
        self.assertEqual(metrics["full_mean_abs"], float(np.mean(noisy)))
        self.assertNotEqual(metrics["full_mean_abs"],
                            float(np.mean(noisy.astype(np.float64))))

    def test_a_nonconverged_episode_keeps_only_the_horizon_metrics(self):
        metrics, _ = se.compute_sparse_effort_metrics(
            self.m1, self.peak, self.sat, t_fire=-1, dt=0.1, speed=15.0,
            u_max=U_MAX)
        for name in se.SUCCESS_METRICS:
            if name in metrics:
                self.assertTrue(math.isnan(metrics[name]), name)
        self.assertAlmostEqual(metrics["l1_integral_to_c2_or_horizon"], 8.0,
                               places=6)
        self.assertAlmostEqual(metrics["env_l1_cost_to_c2_or_horizon"], 120.0,
                               places=4)
        self.assertAlmostEqual(metrics["sat_frac_full"], 0.25, places=7)

    def test_no_l2_or_rms_metric_is_produced(self):
        metrics, temporal = se.compute_sparse_effort_metrics(
            self.m1, self.peak, self.sat, t_fire=300, dt=0.1, speed=15.0,
            u_max=U_MAX)
        for name in list(metrics) + list(temporal) + list(se.METRICS):
            self.assertIsNone(re.match(r"^(l2_|rms_)", name), name)
            self.assertNotIn("l2", name)

    def test_control_above_the_cap_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exceeds the configured cap"):
            se.compute_sparse_effort_metrics(
                self.m1, self.peak + 1.0, self.sat, t_fire=300, dt=0.1,
                speed=15.0, u_max=U_MAX)

    def test_invalid_series_and_constants_are_refused(self):
        with self.assertRaises(ValueError):
            se.compute_sparse_effort_metrics(
                self.m1, self.peak[:10], self.sat, 300, 0.1, 15.0, U_MAX)
        with self.assertRaises(ValueError):
            se.compute_sparse_effort_metrics(
                self.m1, self.peak, self.sat, 300, 0.0, 15.0, U_MAX)
        with self.assertRaises(ValueError):
            se.compute_sparse_effort_metrics(
                self.m1, self.peak, self.sat, HORIZON + 1, 0.1, 15.0, U_MAX)


@unittest.skipUnless(POPULATION_EPISODE.is_file(),
                     "recorded population episode is unavailable")
class PopulationParity(unittest.TestCase):
    """The sparse L1 numbers must be the population lane's L1 numbers."""

    def test_sparse_reduction_reproduces_control_effort_exactly(self):
        manifest = json.loads((POPULATION_BUNDLE / "manifest.json").read_text())
        analysis = control_effort.analyze_episode(POPULATION_EPISODE, manifest)
        row = analysis["row"]
        controls = np.asarray(load_episode(POPULATION_EPISODE)["control_inputs"],
                              dtype=np.float64)
        absolute = np.abs(controls)
        metrics, _ = se.compute_sparse_effort_metrics(
            np.mean(absolute, axis=1), np.max(absolute, axis=1),
            np.mean(np.isclose(absolute, U_MAX, rtol=1e-6, atol=1e-8), axis=1),
            row["t_fire"], dt=row["dt"], speed=row["speed"], u_max=U_MAX)
        for name in ("l1_integral_to_c2", "env_l1_cost_to_c2", "mean_abs_to_c2",
                     "pre100_mean_abs", "post100_mean_abs",
                     "l1_integral_to_c2_or_horizon", "full_mean_abs"):
            self.assertEqual(metrics[name], row[name], name)


class EpisodeGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_a_well_formed_episode_is_accepted(self):
        path, _, _ = _write_episode(self.root / "ok.npz")
        row = se.analyze_sparse_episode(path, arm="unit")["row"]
        self.assertEqual((row["t_fire"], row["success"], row["nonconverged"]),
                         (299, 1, 0))
        self.assertEqual(row["arm"], "unit")
        self.assertEqual(row["policy_short"], "deterministic")
        self.assertEqual(row["N"], row["num_agents"])
        self.assertAlmostEqual(row["t_fire_seconds"], 29.9, places=9)

    def test_multiple_substeps_are_refused(self):
        path, _, _ = _write_episode(
            self.root / "substeps.npz",
            meta=_meta(substeps_per_policy_interval=2))
        with self.assertRaisesRegex(ValueError, "substeps_per_policy_interval"):
            se.analyze_sparse_episode(path)

    def test_a_policy_interval_that_is_not_the_dynamics_step_is_refused(self):
        path, _, _ = _write_episode(self.root / "dt.npz",
                                    meta=_meta(dynamics_dt=0.05))
        with self.assertRaisesRegex(ValueError, "dynamics_dt"):
            se.analyze_sparse_episode(path)

    def test_control_above_the_setting_cap_is_refused(self):
        path, _, _ = _write_episode(
            self.root / "cap.npz",
            arrays={"max_abs_control": np.full(HORIZON, 0.9, dtype=np.float32)})
        with self.assertRaisesRegex(ValueError, "exceeds the configured cap"):
            se.analyze_sparse_episode(path)

    def test_a_full_schema_episode_is_refused(self):
        path, _, _ = _write_episode(self.root / "full.npz",
                                    schema="main-c2-full-2.0")
        with self.assertRaisesRegex(ValueError, "requires .* episodes"):
            se.analyze_sparse_episode(path)

    def test_a_stored_outcome_the_judge_does_not_reproduce_is_refused(self):
        path, _, _ = _write_episode(self.root / "judge.npz",
                                    outcome={"t_fire": 123})
        with self.assertRaisesRegex(ValueError, "disagrees with the stored"):
            se.analyze_sparse_episode(path)
        row = se.analyze_sparse_episode(path, recompute_judgment=False)["row"]
        self.assertEqual(row["t_fire"], 123)

    def test_a_truncated_series_is_refused(self):
        path, _, _ = _write_episode(
            self.root / "short.npz",
            arrays={"mean_abs_control": np.full(HORIZON - 1, 0.2, np.float32)})
        with self.assertRaisesRegex(ValueError, "horizon values"):
            se.analyze_sparse_episode(path)


class SettingSummary(unittest.TestCase):
    """Non-convergence is a column, never a silent drop."""

    def _records(self):
        records = []
        for policy in ("learned_deterministic", "pure_acs"):
            for seed in range(5):
                success = policy == "learned_deterministic"
                row = {
                    "arm": "unit", "run_id": "r", "checkpoint_id": "c",
                    "checkpoint_hash": "h", "mode": "standard",
                    "factor": "minimum_turn_radius", "factor_value": 56.25,
                    "setting_id": "minimum_turn_radius_56p25",
                    "memberships": '["minimum_turn_radius"]', "N": 20,
                    "num_agents": 20, "minimum_turn_radius": 56.25,
                    "interaction_radius": 60.0, "acs_gain_multiplier": 1.0,
                    "initial_position_bound": 250.0, "policy": policy,
                    "policy_short": se._policy_short(policy), "seed": seed,
                    "success": int(success), "nonconverged": int(not success),
                    "t_fire": 300 if success else -1,
                    "t_fire_seconds": 30.0 if success else float("nan"),
                }
                for metric in se.METRICS:
                    if metric in row:
                        continue
                    row[metric] = (float(seed + 1) if success
                                   or metric in se.HORIZON_METRICS
                                   else float("nan"))
                records.append(row)
        return records

    def test_an_all_failing_cell_keeps_its_horizon_metrics_and_its_count(self):
        rows = {row["policy"]: row
                for row in se.summarize_by_setting(self._records())}
        acs = rows["pure_acs"]
        self.assertEqual((acs["episode_count"], acs["success_count"],
                          acs["nonconverged_count"], acs["success_rate"]),
                         (5, 0, 5, 0.0))
        self.assertEqual(acs["l1_integral_to_c2_n"], 0)
        self.assertTrue(math.isnan(acs["l1_integral_to_c2_median"]))
        self.assertEqual(acs["l1_integral_to_c2_or_horizon_n"], 5)
        self.assertEqual(acs["l1_integral_to_c2_or_horizon_median"], 3.0)
        learned = rows["learned_deterministic"]
        self.assertEqual((learned["success_count"],
                          learned["nonconverged_count"]), (5, 0))
        self.assertEqual(learned["l1_integral_to_c2_n"], 5)

    def test_the_t_fire_failure_sentinel_never_enters_an_aggregate(self):
        rows = {row["policy"]: row
                for row in se.summarize_by_setting(self._records())}
        self.assertEqual(rows["pure_acs"]["t_fire_n"], 0)
        self.assertTrue(math.isnan(rows["pure_acs"]["t_fire_median"]))


class BundleRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = _build_bundle(Path(self.tmp.name) / "bundle")
        self.out = Path(self.tmp.name) / "out"
        self.result = se.run_from_bundle(self.root, self.out, arm="unit")

    def test_counts_and_outputs(self):
        self.assertEqual(self.result["episodes"], 12)
        self.assertEqual(self.result["nonconverged"], 3)
        self.assertEqual(self.result["setting_rows"], 4)
        self.assertEqual(self.result["pairs"], 6)
        for name in ("episode_sensitivity_effort.csv", "setting_effort.csv",
                     "paired_vs_pure_acs.csv", "event_aligned.csv",
                     "sensitivity_effort_manifest.json"):
            self.assertTrue((self.out / name).is_file(), name)

    def test_the_manifest_states_that_l2_is_not_recoverable(self):
        manifest = json.loads(
            (self.out / "sensitivity_effort_manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], se.SCHEMA_VERSION)
        self.assertIs(manifest["l2_recoverable"], False)
        self.assertEqual(manifest["substeps_per_policy_interval"], 1)
        self.assertEqual(manifest["episode_count"], 12)
        self.assertEqual(manifest["nonconverged_count"], 3)

    def test_event_aligned_keeps_the_population_columns_in_order(self):
        with (self.out / "event_aligned.csv").open() as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            metrics = {row["metric"] for row in reader}
        pinned = ["N", "policy", "metric", "offset_steps", "offset_seconds",
                  "n", "mean", "q25", "median", "q75"]
        self.assertEqual([name for name in columns if name in pinned], pinned)
        self.assertEqual(metrics, set(se.EVENT_METRICS))
        self.assertNotIn("l2_rate", metrics)

    def test_event_alignment_covers_successes_only(self):
        rows = _read_rows(self.out / "event_aligned.csv")
        cells = {(row["setting_id"], row["policy"]) for row in rows}
        self.assertNotIn(("num_agents_40", "pure_acs"), cells)
        self.assertEqual(len(cells), 3)

    def test_the_episode_bridge_refuses_a_disagreeing_summary(self):
        target = self.root / "summaries" / "episodes.csv"
        rows = _read_rows(target)
        rows[0]["mean_abs_control"] = "0.123"
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "summaries/episodes.csv"):
            se.run_from_bundle(self.root, self.out / "again", arm="unit")

    def test_paired_rows_assert_the_shared_initial_state(self):
        rows = _read_rows(self.out / "paired_vs_pure_acs.csv")
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["pure_acs_policy"], "pure_acs")
        failed = [row for row in rows if row["setting_id"] == "num_agents_40"]
        self.assertTrue(all(row["both_success"] == "0" for row in failed))
        self.assertTrue(all(math.isnan(float(row["delta_l1_integral_to_c2"]))
                            for row in failed))
        self.assertTrue(all(math.isfinite(
            float(row["delta_l1_integral_to_c2_or_horizon"])) for row in failed))

    def test_an_incomplete_bundle_is_refused(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["status"] = "running"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not 'completed'"):
            se.run_from_bundle(self.root, self.out / "again")


class CompareGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.arms = {}
        for name, checkpoint, settings in (
                ("full_NN", "checkpoint_000100", ("baseline",)),
                ("dknn", "checkpoint_000592", ("baseline", "num_agents_40"))):
            root = _build_bundle(base / name, run_id=f"{name}-run",
                                 checkpoint=checkpoint,
                                 checkpoint_hash=name.lower().ljust(64, "0"),
                                 settings=settings)
            result = se.run_from_bundle(root, base / f"{name}-out", arm=name)
            self.arms[name] = Path(result["outputs"]["episodes_csv"])
        self.out = base / "compare"

    def test_an_arm_spanning_two_checkpoints_is_refused(self):
        pooled = Path(self.tmp.name) / "pooled.csv"
        rows = []
        for path in self.arms.values():
            rows.extend(_read_rows(path))
        with pooled.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "checkpoint hashes"):
            se.run_compare({"pooled": pooled}, self.out)

    def test_a_setting_missing_from_one_arm_is_excluded_not_deleted(self):
        result = se.run_compare(self.arms, self.out, exclude_settings=())
        excluded = _read_rows(self.out / "excluded_settings.csv")
        self.assertEqual([row["setting_id"] for row in excluded],
                         ["num_agents_40"])
        self.assertTrue(excluded[0]["reason"].startswith("absent_in_arm:"))
        self.assertEqual(excluded[0]["arm_absent"], "full_NN")
        cross = _read_rows(self.out / "cross_arm_setting_effort.csv")
        self.assertTrue(cross)
        self.assertNotIn("num_agents_40",
                         {row["setting_id"] for row in cross})
        self.assertEqual({row["arm"] for row in cross}, {"dknn", "full_NN"})
        # the dknn episode CSV still carries the excluded setting
        dknn = _read_rows(self.arms["dknn"])
        self.assertIn("num_agents_40", {row["setting_id"] for row in dknn})
        self.assertIs(result["controlled_comparison"], False)
        self.assertTrue(result["reason"])

    def test_no_cross_arm_delta_or_percent_change_column_is_emitted(self):
        se.run_compare(self.arms, self.out)
        for path in sorted(self.out.glob("*.csv")):
            with path.open() as stream:
                columns = next(csv.reader(stream))
            for name in columns:
                self.assertFalse(name.startswith("delta_"), (path.name, name))
                self.assertFalse(name.startswith("percent_change_"),
                                 (path.name, name))
        manifest = json.loads((self.out / "compare_manifest.json").read_text())
        self.assertIs(manifest["controlled_comparison"], False)
        self.assertIn("checkpoint", manifest["reason"])

    def test_every_cross_arm_row_carries_its_provenance(self):
        se.run_compare(self.arms, self.out)
        rows = _read_rows(self.out / "cross_arm_setting_effort.csv")
        for row in rows:
            self.assertTrue(row["arm"] and row["run_id"]
                            and row["checkpoint_hash"])
        provenance = _read_rows(self.out / "arm_provenance.csv")
        self.assertEqual({row["arm"] for row in provenance},
                         {"dknn", "full_NN"})

    def test_the_default_exclusion_list_is_honoured(self):
        self.assertEqual(se.DEFAULT_COMPARE_EXCLUSIONS, ("num_agents_80",))
        se.run_compare(self.arms, self.out,
                       exclude_settings=("baseline", "num_agents_80"))
        excluded = {row["setting_id"]: row["reason"]
                    for row in _read_rows(self.out / "excluded_settings.csv")}
        self.assertEqual(excluded["baseline"], "excluded_by_request")
        self.assertTrue(excluded["num_agents_40"].startswith("absent_in_arm:"))


class UnifiedCliWiring(unittest.TestCase):
    """``python -m eval sensitivity effort`` must reach the same functions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = _build_bundle(self.base / "bundle")

    @staticmethod
    def _run(argv):
        import eval.__main__ as unified

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = unified.main(argv)
        return code, json.loads(stream.getvalue())

    def test_the_subcommands_are_registered_beside_run_validate_plot(self):
        import eval.__main__ as unified

        args = unified._parser().parse_args([
            "sensitivity", "effort", "--run", "bundle", "--arm", "dknn",
            "--settings", "baseline,num_agents_40", "--policies", "pure_acs",
            "--seeds", "0-4", "--no-event-aligned"])
        self.assertEqual(args.sensitivity_command, "effort")
        self.assertEqual(args.settings, ["baseline", "num_agents_40"])
        self.assertEqual(args.policies, ["pure_acs"])
        self.assertTrue(args.no_event_aligned)
        compare = unified._parser().parse_args([
            "sensitivity", "effort-compare", "--arm", "dknn=/tmp/a.csv",
            "--out", "/tmp/out"])
        self.assertEqual(compare.arm, [("dknn", Path("/tmp/a.csv"))])

    def test_an_arm_without_name_equals_path_is_rejected(self):
        import eval.__main__ as unified

        with self.assertRaises(SystemExit):
            unified._parser().parse_args([
                "sensitivity", "effort-compare", "--arm", "/tmp/a.csv",
                "--out", "/tmp/out"])

    def test_effort_and_effort_compare_reach_the_module(self):
        out = self.base / "cli-out"
        code, result = self._run([
            "sensitivity", "effort", "--run", str(self.root),
            "--output", str(out), "--arm", "unit", "--seeds", "0,1,2"])
        self.assertEqual(code, 0)
        self.assertEqual((result["episodes"], result["nonconverged"]), (12, 3))
        self.assertEqual(result["schema_version"], se.SCHEMA_VERSION)
        direct = se.run_from_bundle(self.root, self.base / "direct", arm="unit")
        self.assertEqual(
            (out / "episode_sensitivity_effort.csv").read_text(),
            (self.base / "direct" / "episode_sensitivity_effort.csv").read_text())
        self.assertEqual(direct["pairs"], result["pairs"])

        code, compared = self._run([
            "sensitivity", "effort-compare", "--arm",
            f"unit={out / 'episode_sensitivity_effort.csv'}",
            "--out", str(self.base / "cli-compare")])
        self.assertEqual(code, 0)
        self.assertIs(compared["controlled_comparison"], False)
        self.assertEqual(compared["excluded_settings"], [])
        self.assertTrue(
            (self.base / "cli-compare" / "arm_provenance.csv").is_file())


class NoL2Columns(unittest.TestCase):
    """No emitted table may grow an L2/RMS column, in any lane."""

    FORBIDDEN = re.compile(r"^(l2_|rms_)")

    def _assert_clean(self, directory):
        seen = 0
        for path in sorted(Path(directory).rglob("*.csv")):
            with path.open() as stream:
                reader = csv.reader(stream)
                columns = next(reader)
            for name in columns:
                self.assertIsNone(self.FORBIDDEN.match(name), (path, name))
                for prefix in ("learned_", "pure_acs_", "delta_",
                               "percent_change_"):
                    self.assertIsNone(
                        self.FORBIDDEN.match(name[len(prefix):])
                        if name.startswith(prefix) else None, (path, name))
            if path.name == "event_aligned.csv":
                with path.open() as stream:
                    metrics = {row["metric"] for row in csv.DictReader(stream)}
                self.assertNotIn("l2_rate", metrics)
            seen += 1
        return seen

    def test_a_freshly_written_bundle_has_no_l2_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_bundle(Path(tmp) / "bundle")
            out = Path(tmp) / "out"
            se.run_from_bundle(root, out, arm="unit")
            se.run_compare(
                {"unit": out / "episode_sensitivity_effort.csv"},
                Path(tmp) / "compare", exclude_settings=())
            self.assertGreaterEqual(self._assert_clean(out), 4)
            self.assertGreaterEqual(self._assert_clean(Path(tmp) / "compare"), 3)

    def test_the_recorded_outputs_have_no_l2_column(self):
        present = [path for path in RECORDED_OUTPUTS if path.is_dir()]
        if not present:
            self.skipTest("no recorded sensitivity-effort output on disk")
        for directory in present:
            self.assertGreater(self._assert_clean(directory), 0, directory)


if __name__ == "__main__":
    unittest.main()
