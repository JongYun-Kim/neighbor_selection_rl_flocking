"""Focused regression tests for dynamic-k full-artifact analysis."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval.analysis.core import (
    centroid_distance_ranking,
    coerce_episode,
    compute_cutoff_radii,
    mean_heading_deviation_ranking,
    pointer_actions_to_mask,
    rank_episode_cutoff_radii,
    state_entropy_series,
    state_polarization,
    time_window_steps,
    validate_pointer_mask,
)
from eval.analysis.bundle import deterministic_paths
from eval.analysis.heatmaps import run_from_bundle as run_heatmaps
from eval.analysis.population import OnlineMoments, PopulationBuilder, figure_filename
from eval.analysis.radius import frame_indices
from eval.analysis.radii import _candidate_paths, run_from_bundle as run_radii
from eval.artifacts import FULL_SCHEMA_VERSION


def synthetic_mapping(
    num_agents=4,
    horizon=3,
    seed=0,
    dt=0.5,
    policy="deterministic",
    include_r0=True,
):
    states = np.zeros((horizon + 1, num_agents, 5), dtype=np.float64)
    agent = np.arange(num_agents, dtype=np.float64)
    for step in range(horizon + 1):
        headings = -0.6 + 1.2 * agent / max(1, num_agents - 1) + 0.03 * step
        states[step, :, 0] = agent * (2.0 + 0.1 * step)
        states[step, :, 1] = (agent % 3) * 1.5 - 0.2 * step * agent
        states[step, :, 2] = np.cos(headings)
        states[step, :, 3] = np.sin(headings)
        states[step, :, 4] = headings
    pointer = np.empty((horizon, num_agents), dtype=np.int16)
    for step in range(horizon):
        pointer[step] = (np.arange(num_agents) + step + 1) % num_agents
    binary = pointer_actions_to_mask(states, pointer)
    position_entropy, velocity_entropy = state_entropy_series(states)
    result = {
        "agent_states": states,
        "pointer_actions": pointer,
        "binary_actions": binary,
        "phi": state_polarization(states).astype(np.float32),
        "s_ent": position_entropy.astype(np.float32),
        "v_ent": velocity_entropy.astype(np.float32),
        "meta": {
            "num_agents": num_agents,
            "horizon": horizon,
            "seed": seed,
            "policy": policy,
            "dt": dt,
        },
    }
    if include_r0:
        result["meta"]["r0"] = 8.0
    return result


def write_episode(
    path: Path,
    num_agents: int,
    seed: int,
    dt=0.5,
    policy="deterministic",
    include_r0=True,
):
    episode = synthetic_mapping(
        num_agents=num_agents,
        horizon=3,
        seed=seed,
        dt=dt,
        policy=policy,
        include_r0=include_r0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(FULL_SCHEMA_VERSION),
        meta=np.asarray(json.dumps(episode["meta"])),
        agent_states=episode["agent_states"],
        pointer_actions=episode["pointer_actions"],
        binary_actions=episode["binary_actions"],
        phi=episode["phi"],
        s_ent=episode["s_ent"],
        v_ent=episode["v_ent"],
    )


class CutoffGeometryTest(unittest.TestCase):
    def test_cutoff_uses_pre_step_state(self):
        states = np.zeros((2, 3, 5), dtype=np.float64)
        states[0, :, 0] = [0.0, 3.0, 10.0]
        states[1, :, 0] = [100.0, 101.0, 102.0]
        pointer = np.asarray([[1, 2, 0]])
        np.testing.assert_allclose(
            compute_cutoff_radii(states, pointer), [[3.0, 7.0, 10.0]]
        )

    def test_equal_distance_ties_and_self_pointer_rule(self):
        states = np.zeros((2, 4, 5), dtype=np.float64)
        states[:, 1, 0] = 1.0
        states[:, 2, 0] = -1.0
        # Agent 3 is co-located with ego 0, but a self pointer still means self only.
        pointer = np.asarray([[1, 1, 2, 3]])
        mask = pointer_actions_to_mask(states, pointer)
        np.testing.assert_array_equal(np.flatnonzero(mask[0, 0]), [0, 1, 2, 3])
        np.testing.assert_array_equal(np.flatnonzero(mask[0, 3]), [3])

    def test_environment_float32_near_tie_is_reproduced(self):
        states = np.zeros((2, 3, 5), dtype=np.float64)
        states[:, 1, :2] = [108.01147760, 37.45521362]
        states[:, 2, :2] = [98.83914879, 57.44732948]
        pointer = np.asarray([[1, 1, 2]])
        self.assertGreater(
            np.linalg.norm(states[0, 2, :2]),
            np.linalg.norm(states[0, 1, :2]),
        )
        mask = pointer_actions_to_mask(states, pointer)
        np.testing.assert_array_equal(np.flatnonzero(mask[0, 0]), [0, 1, 2])


class RankingAndAlignmentTest(unittest.TestCase):
    def test_centroid_and_circular_heading_rank_nearest_first_stably(self):
        centered = np.asarray([[[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]]])
        distances, order = centroid_distance_ranking(centered)
        np.testing.assert_array_equal(order, [[0, 1, 2]])
        np.testing.assert_allclose(np.take_along_axis(distances, order, axis=1), [[1, 1, 3]])

        states = np.zeros((1, 4, 5), dtype=np.float64)
        states[0, :, 4] = [0.0, 0.0, 0.4, -0.4]
        _, deviation, heading_order = mean_heading_deviation_ranking(states)
        ranked = np.take_along_axis(deviation, heading_order, axis=1)
        self.assertTrue(np.all(np.diff(ranked, axis=1) >= -1e-12))
        np.testing.assert_array_equal(heading_order[0, :2], [0, 1])

    def test_legacy_post_step_metric_aliases_are_aligned_to_t_plus_one(self):
        mapping = synthetic_mapping()
        legacy = {
            "states": mapping["agent_states"],
            "pointer": mapping["pointer_actions"],
            "binary": mapping["binary_actions"],
            "polarization": mapping["phi"][1:],
            "spatial_entropy": mapping["s_ent"][1:],
            "velocity_entropy": mapping["v_ent"][1:],
            "meta": mapping["meta"],
        }
        episode = coerce_episode(legacy)
        self.assertEqual(episode.phi.shape, (episode.horizon + 1,))
        self.assertEqual(episode.position_entropy.shape, (episode.horizon + 1,))
        self.assertEqual(validate_pointer_mask(episode)["mismatch_count"], 0)

    def test_ranked_radius_rows_are_monotone_in_their_context_metric(self):
        mapping = synthetic_mapping()
        ranked = rank_episode_cutoff_radii(
            mapping["agent_states"], mapping["pointer_actions"]
        )
        self.assertTrue(
            np.all(np.diff(ranked["centroid_ranked_distances"], axis=1) >= -1e-12)
        )
        self.assertTrue(
            np.all(np.diff(ranked["heading_ranked_deviation"], axis=1) >= -1e-12)
        )

    def test_physical_time_window_requires_exact_cell_boundary(self):
        self.assertEqual(time_window_steps(1000, 0.1, 30.0), 300)
        with self.assertRaises(ValueError):
            time_window_steps(1000, 0.1, 30.05)


class StreamingPopulationTest(unittest.TestCase):
    def test_online_mean_and_population_std_match_numpy(self):
        samples = np.asarray(
            [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]], [[5.0, 7.0], [9.0, 11.0]]]
        )
        moments = OnlineMoments()
        for sample in samples:
            moments.update(sample)
        mean, std = moments.finalize()
        np.testing.assert_allclose(mean, np.mean(samples, axis=0))
        np.testing.assert_allclose(std, np.std(samples, axis=0, ddof=0))

    def test_population_builder_streams_validated_episodes(self):
        builder = PopulationBuilder(4)
        first = coerce_episode(synthetic_mapping(seed=0), source=Path("a.npz"))
        second_mapping = synthetic_mapping(seed=1)
        second_mapping["agent_states"] = second_mapping["agent_states"].copy()
        second_mapping["agent_states"][:, :, :2] *= 1.2
        # Recompute all persisted fields affected by the modified states.
        second_mapping["binary_actions"] = pointer_actions_to_mask(
            second_mapping["agent_states"], second_mapping["pointer_actions"]
        )
        position, velocity = state_entropy_series(second_mapping["agent_states"])
        second_mapping["s_ent"] = position
        second_mapping["v_ent"] = velocity
        second_mapping["phi"] = state_polarization(second_mapping["agent_states"])
        second = coerce_episode(second_mapping, source=Path("b.npz"))
        builder.add(first)
        builder.add(second)
        aggregate = builder.finalize(expected_count=2)
        self.assertEqual(aggregate.episode_count, 2)
        self.assertEqual(aggregate.centroid_mean.shape, (3, 4))
        self.assertTrue(np.all(aggregate.centroid_std >= 0.0))

    def test_names_keep_four_figure_set_members_distinct(self):
        names = {
            figure_filename(n, ranking, max_time_seconds=30.0, with_entropy_panels=True)
            for n in (20, 40)
            for ranking in ("centroid_distance", "mean_heading_difference")
        }
        self.assertEqual(len(names), 4)
        self.assertTrue(all("tmax30s_with_entropies" in name for name in names))

    def test_frame_indices_always_include_final_action(self):
        self.assertEqual(frame_indices(0, 11, 4), [0, 4, 8, 10])


class BundlePathSelectionTest(unittest.TestCase):
    def test_policy_filter_ignores_deterministic_text_in_run_ancestors(self):
        root = Path("/tmp/deterministic-ablation-run/episodes/checkpoint/N20")
        stochastic = root / "stochastic" / "seed_00007.npz"
        deterministic = root / "deterministic" / "seed_00007.npz"
        self.assertEqual(
            deterministic_paths([stochastic, deterministic]), [deterministic]
        )

    def test_radius_candidate_names_are_anchored_and_fallback_keeps_n(self):
        base = Path("/tmp/run/episodes/checkpoint")
        canonical = base / "N20" / "deterministic" / "seed_00007.npz"
        legacy = base / "N20" / "best_deterministic" / "episode_00007.npz"
        prefix_collision = (
            base / "N20" / "deterministic" / "seed_000070.npz"
        )
        wrong_seed = base / "N20" / "deterministic" / "episode_00008.npz"
        wrong_n = base / "N40" / "deterministic" / "seed_00007.npz"
        paths = [canonical, legacy, prefix_collision, wrong_seed, wrong_n]
        self.assertEqual(_candidate_paths(paths, 20, 7), [canonical, legacy])
        self.assertEqual(
            _candidate_paths([wrong_seed, wrong_n], 20, 7), [wrong_seed]
        )


class BundleAdapterIntegrationTest(unittest.TestCase):
    def test_heatmap_bundle_adapter_writes_exactly_four_full_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "population"
            for n_agents in (20, 40):
                write_episode(
                    root
                    / "episodes"
                    / "population_v1"
                    / "checkpoint"
                    / "N{}".format(n_agents)
                    / "deterministic"
                    / "seed_00000.npz",
                    n_agents,
                    0,
                )
            manifest = {
                "schema_version": "evaluation-bundle-2.0",
                "suite": "population_v1",
                "run_id": "synthetic",
                "fingerprint": "test",
                "status": "completed",
                "spec": {
                    "num_agents": [20, 40],
                    "seeds": [0],
                    "policies": ["deterministic"],
                    "dt": 0.5,
                },
            }
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(directory) / "analysis"
            result = run_heatmaps(root, output=output)
            self.assertEqual(result["figure_count"], 4)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 4)
            self.assertTrue(
                (output / "derived" / "N20_population_ranked_cutoff_radius.npz").is_file()
            )
            self.assertTrue(
                (output / "derived" / "N40_population_ranked_cutoff_radius.npz").is_file()
            )
            persisted = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["analysis_output_root"], ".")
            self.assertEqual(persisted["output_dir"], ".")
            self.assertEqual(persisted["source_path_base"], "source_bundle")
            self.assertFalse(Path(persisted["source_bundle"]).is_absolute())
            source_root = (output / persisted["source_bundle"]).resolve()
            self.assertEqual(source_root, root.resolve())
            self.assertTrue(
                all(
                    not Path(item["path"]).is_absolute()
                    and (output / item["path"]).is_file()
                    for item in persisted["figures"]
                )
            )
            for population in persisted["populations"].values():
                derived_path = Path(population["derived_path"])
                self.assertFalse(derived_path.is_absolute())
                self.assertTrue((output / derived_path).is_file())
                self.assertTrue(
                    all(
                        not Path(path).is_absolute()
                        and (source_root / path).is_file()
                        for path in population["source_paths"]
                    )
                )
                with np.load(output / derived_path, allow_pickle=False) as archive:
                    self.assertEqual(str(archive["source_path_base"]), "source_bundle")
                    self.assertTrue(
                        all(
                            not Path(str(path)).is_absolute()
                            and (source_root / str(path)).is_file()
                            for path in archive["source_paths"]
                        )
                    )
            entropy_output = Path(directory) / "analysis_entropy"
            entropy_result = run_heatmaps(
                root,
                output=entropy_output,
                t_max_seconds=1.0,
                with_entropies=True,
            )
            self.assertEqual(entropy_result["figure_count"], 4)
            self.assertTrue(
                all(
                    figure["panel_count_excluding_colorbar"] == 3
                    and figure["entropy_positive_direction"] == "left"
                    for figure in entropy_result["figures"]
                )
            )

            # With no explicit output, each view owns a separate manifest and
            # derived directory so successive invocations cannot overwrite it.
            default_full = run_heatmaps(root)
            default_detail = run_heatmaps(root, t_max_seconds=1.0)
            default_entropy = run_heatmaps(
                root, t_max_seconds=1.0, with_entropies=True
            )
            expected_outputs = {
                root / "analysis" / "ranked_heatmaps" / "full",
                root / "analysis" / "ranked_heatmaps" / "tmax1s",
                root
                / "analysis"
                / "ranked_heatmaps"
                / "tmax1s_with_entropies",
            }
            actual_outputs = {
                Path(default_full["output_dir"]),
                Path(default_detail["output_dir"]),
                Path(default_entropy["output_dir"]),
            }
            self.assertEqual(actual_outputs, expected_outputs)
            self.assertTrue(
                all((directory / "manifest.json").is_file() for directory in expected_outputs)
            )

            # Default analysis lives inside the bundle.  Moving that bundle
            # must not invalidate any persisted source/output reference.
            moved_root = Path(directory) / "population_moved"
            root.rename(moved_root)
            moved_output = (
                moved_root / "analysis" / "ranked_heatmaps" / "full"
            )
            moved_manifest = json.loads(
                (moved_output / "manifest.json").read_text(encoding="utf-8")
            )
            moved_source = (
                moved_output / moved_manifest["source_bundle"]
            ).resolve()
            self.assertEqual(moved_source, moved_root.resolve())
            self.assertTrue(
                all(
                    (moved_output / item["path"]).is_file()
                    for item in moved_manifest["figures"]
                )
            )
            for population in moved_manifest["populations"].values():
                self.assertTrue(
                    (moved_output / population["derived_path"]).is_file()
                )
                self.assertTrue(
                    all(
                        (moved_source / path).is_file()
                        for path in population["source_paths"]
                    )
                )

    def test_radii_bundle_adapter_writes_ranked_circle_previews(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "population"
            write_episode(
                root
                / "episodes"
                / "population_v1"
                / "checkpoint"
                / "N20"
                / "deterministic"
                / "seed_00007.npz",
                20,
                7,
            )
            manifest = {
                "schema_version": "evaluation-bundle-2.0",
                "suite": "population_v1",
                "run_id": "synthetic",
                "fingerprint": "test",
                "status": "completed",
                "spec": {
                    "num_agents": [20],
                    "seeds": [7],
                    "policies": ["deterministic"],
                    "dt": 0.5,
                },
            }
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(directory) / "radius"
            result = run_radii(root, 20, 7, output=output, animation=False)
            self.assertEqual(result["mask_validation"]["mismatch_count"], 0)
            self.assertEqual(len(result["previews"]), 3)
            self.assertEqual(len(list((output / "previews").glob("*.png"))), 3)
            persisted = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["analysis_output_root"], ".")
            self.assertEqual(persisted["output_dir"], ".")
            self.assertFalse(Path(persisted["source_bundle"]).is_absolute())
            self.assertFalse(Path(persisted["source_episode"]).is_absolute())
            source_root = (output / persisted["source_bundle"]).resolve()
            self.assertTrue(
                (source_root / persisted["source_episode"]).is_file()
            )
            self.assertTrue(
                all(
                    not Path(item["path"]).is_absolute()
                    and (output / item["path"]).is_file()
                    for item in persisted["previews"]
                )
            )

    def test_radii_skips_stochastic_sibling_and_inherits_legacy_manifest_r0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "deterministic-text-in-run-id"
            stochastic = (
                root
                / "episodes"
                / "checkpoint"
                / "N20"
                / "policy_a"
                / "seed_00007.npz"
            )
            deterministic = (
                root
                / "episodes"
                / "checkpoint"
                / "N20"
                / "policy_b"
                / "episode_00007.npz"
            )
            write_episode(
                stochastic,
                20,
                7,
                policy="stochastic",
                include_r0=False,
            )
            write_episode(
                deterministic,
                20,
                7,
                policy="deterministic",
                include_r0=False,
            )
            manifest = {
                "schema_version": "legacy-evaluation",
                "run_id": "deterministic-text-in-run-id",
                "status": "completed",
                "spec": {"num_agents": [20], "seeds": [7], "dt": 0.5},
                "evaluation": {
                    "config_by_num_agents": {
                        "20": {"control": {"r0": 23.5}}
                    }
                },
            }
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "explicit analysis output directory"
            ):
                run_radii(root, 20, 7, animation=False)
            with self.assertRaisesRegex(ValueError, "must be outside"):
                run_radii(
                    root,
                    20,
                    7,
                    output=root / "analysis" / "legacy-radius",
                    animation=False,
                )
            self.assertFalse((root / "analysis").exists())
            output = Path(directory) / "radius_legacy"
            result = run_radii(root, 20, 7, output=output, animation=False)
            self.assertEqual(Path(result["source_episode"]), deterministic)
            self.assertEqual(result["r0_m"], 23.5)
            self.assertEqual(result["mask_validation"]["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
