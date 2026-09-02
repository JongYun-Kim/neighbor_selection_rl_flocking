"""Streaming population aggregation and ranked cutoff-radius figures."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval.artifacts import atomic_save_npz, atomic_write_json

from .core import (
    EpisodeData,
    coerce_episode,
    rank_episode_cutoff_radii,
    require_deterministic_episode,
    resolve_dt,
    time_window_steps,
    validate_pointer_mask,
)


ANALYSIS_SCHEMA_VERSION = "dynamic-k-ranked-population-v1"
RANKING_TYPES = ("centroid_distance", "mean_heading_difference")


class OnlineMoments:
    """Numerically stable element-wise population moments (Welford, ddof=0)."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None

    def update(self, sample: np.ndarray) -> None:
        value = np.asarray(sample, dtype=np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError("population sample contains a non-finite value")
        if self.mean is None:
            self.mean = value.copy()
            self._m2 = np.zeros_like(value)
            self.count = 1
            return
        if value.shape != self.mean.shape:
            raise ValueError(
                "population sample shape {} differs from {}".format(
                    value.shape, self.mean.shape
                )
            )
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 1 or self.mean is None or self._m2 is None:
            raise ValueError("cannot finalize empty population moments")
        return self.mean.copy(), np.sqrt(self._m2 / self.count)


@dataclass(frozen=True)
class PopulationAggregate:
    num_agents: int
    episode_count: int
    horizon: int
    dt: float
    seeds: Tuple[Optional[int], ...]
    source_paths: Tuple[str, ...]
    centroid_mean: np.ndarray
    centroid_std: np.ndarray
    heading_mean: np.ndarray
    heading_std: np.ndarray
    centroid_metric_mean: np.ndarray
    centroid_metric_std: np.ndarray
    heading_metric_mean: np.ndarray
    heading_metric_std: np.ndarray
    position_entropy_mean: np.ndarray
    position_entropy_std: np.ndarray
    velocity_entropy_mean: np.ndarray
    velocity_entropy_std: np.ndarray
    max_metric_errors: Mapping[str, float]

    def cutoff_values(self, ranking_type: str, statistic: str = "mean") -> np.ndarray:
        if ranking_type not in RANKING_TYPES:
            raise ValueError("unknown ranking type {!r}".format(ranking_type))
        if statistic not in {"mean", "std"}:
            raise ValueError("statistic must be 'mean' or 'std'")
        prefix = "centroid" if ranking_type == "centroid_distance" else "heading"
        return np.asarray(getattr(self, "{}_{}".format(prefix, statistic)))


class PopulationBuilder:
    """Consume one full episode at a time without stacking the population."""

    _MOMENT_NAMES = (
        "centroid",
        "heading",
        "centroid_metric",
        "heading_metric",
        "position_entropy",
        "velocity_entropy",
    )

    def __init__(
        self,
        num_agents: int,
        *,
        dt_override: Optional[float] = None,
        require_deterministic: bool = True,
        reject_duplicate_seeds: bool = True,
    ) -> None:
        self.num_agents = int(num_agents)
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.dt_override = dt_override
        self.require_deterministic = bool(require_deterministic)
        self.reject_duplicate_seeds = bool(reject_duplicate_seeds)
        self._moments = {name: OnlineMoments() for name in self._MOMENT_NAMES}
        self._horizon: Optional[int] = None
        self._dt: Optional[float] = None
        self._seeds: List[Optional[int]] = []
        self._seen_seeds = set()
        self._sources: List[str] = []
        self._max_errors: Dict[str, float] = {}

    @property
    def count(self) -> int:
        return self._moments["centroid"].count

    def add(self, episode: EpisodeData) -> None:
        if episode.num_agents != self.num_agents:
            raise ValueError(
                "expected N={}, got N={}".format(self.num_agents, episode.num_agents)
            )
        if self.require_deterministic:
            require_deterministic_episode(episode)
        validate_pointer_mask(episode)
        dt = resolve_dt(episode, self.dt_override)
        if self._horizon is None:
            self._horizon = episode.horizon
            self._dt = dt
        elif episode.horizon != self._horizon:
            raise ValueError("population episodes have inconsistent horizons")
        elif not np.isclose(dt, self._dt, rtol=0.0, atol=1e-12):
            raise ValueError("population episodes have inconsistent dt")

        seed_value = episode.meta.get("seed")
        seed = None if seed_value is None else int(seed_value)
        if (
            seed is not None
            and self.reject_duplicate_seeds
            and seed in self._seen_seeds
        ):
            raise ValueError(
                "duplicate seed {} in N={} population".format(seed, self.num_agents)
            )
        if seed is not None:
            self._seen_seeds.add(seed)

        ranked = rank_episode_cutoff_radii(
            episode.states, episode.pointer_actions
        )
        self._moments["centroid"].update(ranked["centroid_ranked_radii"])
        self._moments["heading"].update(ranked["heading_ranked_radii"])
        self._moments["centroid_metric"].update(
            ranked["centroid_ranked_distances"]
        )
        self._moments["heading_metric"].update(
            ranked["heading_ranked_deviation"]
        )
        self._moments["position_entropy"].update(episode.position_entropy)
        self._moments["velocity_entropy"].update(episode.velocity_entropy)

        for name, value in episode.metric_errors.items():
            self._max_errors[name] = max(self._max_errors.get(name, 0.0), value)
        self._seeds.append(seed)
        self._sources.append(
            str(episode.source.resolve()) if episode.source is not None else "<memory>"
        )

    def finalize(self, expected_count: Optional[int] = None) -> PopulationAggregate:
        if expected_count is not None and self.count != int(expected_count):
            raise ValueError(
                "N={} expected {} episodes, found {}".format(
                    self.num_agents, int(expected_count), self.count
                )
            )
        if self.count < 1 or self._horizon is None or self._dt is None:
            raise ValueError("N={} population is empty".format(self.num_agents))
        centroid_mean, centroid_std = self._moments["centroid"].finalize()
        heading_mean, heading_std = self._moments["heading"].finalize()
        centroid_metric_mean, centroid_metric_std = self._moments[
            "centroid_metric"
        ].finalize()
        heading_metric_mean, heading_metric_std = self._moments[
            "heading_metric"
        ].finalize()
        position_entropy_mean, position_entropy_std = self._moments[
            "position_entropy"
        ].finalize()
        velocity_entropy_mean, velocity_entropy_std = self._moments[
            "velocity_entropy"
        ].finalize()
        return PopulationAggregate(
            num_agents=self.num_agents,
            episode_count=self.count,
            horizon=self._horizon,
            dt=self._dt,
            seeds=tuple(self._seeds),
            source_paths=tuple(self._sources),
            centroid_mean=centroid_mean,
            centroid_std=centroid_std,
            heading_mean=heading_mean,
            heading_std=heading_std,
            centroid_metric_mean=centroid_metric_mean,
            centroid_metric_std=centroid_metric_std,
            heading_metric_mean=heading_metric_mean,
            heading_metric_std=heading_metric_std,
            position_entropy_mean=position_entropy_mean,
            position_entropy_std=position_entropy_std,
            velocity_entropy_mean=velocity_entropy_mean,
            velocity_entropy_std=velocity_entropy_std,
            max_metric_errors=dict(self._max_errors),
        )


def load_episode_data(path: Path, *, require_metrics: bool = True) -> EpisodeData:
    """Load via the canonical artifact interface, imported lazily for CLI use."""

    from eval.artifacts import load_episode

    source = Path(path).resolve()
    try:
        view = load_episode(source)
        return coerce_episode(
            view, source=source, require_metrics=require_metrics
        )
    except Exception as exc:
        raise ValueError("failed to load episode {}: {}".format(source, exc)) from exc


def discover_episode_paths(inputs: Sequence[Path]) -> List[Path]:
    """Resolve explicit episode files or directories without layout assumptions."""

    paths: List[Path] = []
    for raw in inputs:
        item = Path(raw).expanduser().resolve()
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            episode_root = item / "episodes"
            search_root = episode_root if episode_root.is_dir() else item
            paths.extend(sorted(search_root.rglob("*.npz")))
        else:
            raise FileNotFoundError("analysis input does not exist: {}".format(item))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("no episode .npz files were found")
    return unique


def aggregate_paths(
    paths: Sequence[Path],
    *,
    num_agents: Sequence[int] = (20, 40),
    dt_override: Optional[float] = None,
    expected_count: Optional[int] = None,
    require_deterministic: bool = True,
    reject_duplicate_seeds: bool = True,
) -> Dict[int, PopulationAggregate]:
    """Stream canonical full episodes into independent N-specific moments."""

    requested = tuple(int(value) for value in num_agents)
    if len(set(requested)) != len(requested):
        raise ValueError("num_agents contains duplicates")
    builders = {
        value: PopulationBuilder(
            value,
            dt_override=dt_override,
            require_deterministic=require_deterministic,
            reject_duplicate_seeds=reject_duplicate_seeds,
        )
        for value in requested
    }
    ignored = 0
    for index, path in enumerate(paths, start=1):
        episode = load_episode_data(path, require_metrics=True)
        if require_deterministic:
            mode = episode.meta.get(
                "action_mode", episode.meta.get("policy")
            )
            if mode is not None and str(mode).strip().lower() not in {
                "deterministic",
                "det",
                "greedy",
            }:
                ignored += 1
                continue
        builder = builders.get(episode.num_agents)
        if builder is None:
            ignored += 1
            continue
        builder.add(episode)
        if builder.count == 1 or builder.count % 10 == 0 or index == len(paths):
            print(
                "[analysis] source {}/{}: N={} accepted={} seed={}".format(
                    index,
                    len(paths),
                    episode.num_agents,
                    builder.count,
                    episode.meta.get("seed", "?"),
                ),
                flush=True,
            )
    aggregates = {
        value: builders[value].finalize(expected_count=expected_count)
        for value in requested
    }
    if ignored:
        print(
            "[analysis] ignored {} episode(s) outside requested N values {}".format(
                ignored, requested
            ),
            flush=True,
        )
    return aggregates


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    atomic_save_npz(path, arrays)


def _relative_source_paths(
    source_paths: Sequence[str], source_root: Path
) -> Tuple[str, ...]:
    root = Path(source_root).resolve()
    portable = []
    for value in source_paths:
        if value == "<memory>":
            portable.append(value)
            continue
        relative = os.path.relpath(str(Path(value).resolve()), str(root))
        portable.append(Path(relative).as_posix())
    return tuple(portable)


def _common_source_root(
    aggregates: Sequence[PopulationAggregate],
) -> Path:
    parents = [
        str(Path(value).resolve().parent)
        for aggregate in aggregates
        for value in aggregate.source_paths
        if value != "<memory>"
    ]
    return Path(os.path.commonpath(parents)) if parents else Path.cwd().resolve()


def save_aggregate(
    aggregate: PopulationAggregate,
    path: Path,
    *,
    source_root: Optional[Path] = None,
    source_path_base: str = "common_source_root",
) -> Path:
    """Persist full mean/std arrays once; all views are slices of this artifact."""

    resolved_source_root = (
        Path(source_root).resolve()
        if source_root is not None
        else _common_source_root((aggregate,))
    )
    portable_sources = _relative_source_paths(
        aggregate.source_paths, resolved_source_root
    )
    seeds = np.asarray(
        [-1 if value is None else value for value in aggregate.seeds], dtype=np.int64
    )
    _save_npz_atomic(
        path,
        schema_version=np.asarray(ANALYSIS_SCHEMA_VERSION),
        num_agents=np.asarray(aggregate.num_agents, dtype=np.int32),
        episode_count=np.asarray(aggregate.episode_count, dtype=np.int32),
        population_std_ddof=np.asarray(0, dtype=np.int8),
        horizon=np.asarray(aggregate.horizon, dtype=np.int32),
        dt_seconds=np.asarray(aggregate.dt, dtype=np.float64),
        seeds=seeds,
        source_path_base=np.asarray(source_path_base),
        source_paths=np.asarray(portable_sources),
        analysis_output_root=np.asarray("."),
        rank_semantics=np.asarray(
            "rank independently within every episode and action time before aggregation"
        ),
        entropy_time_alignment=np.asarray(
            "state boundaries t=0,dt,...,T*dt; heatmap cells use pre-step states"
        ),
        action_time_seconds=np.arange(aggregate.horizon, dtype=np.float64)
        * aggregate.dt,
        state_time_seconds=np.arange(aggregate.horizon + 1, dtype=np.float64)
        * aggregate.dt,
        centroid_rank_mean_cutoff_radius_m=aggregate.centroid_mean,
        centroid_rank_episode_std_cutoff_radius_m=aggregate.centroid_std,
        heading_rank_mean_cutoff_radius_m=aggregate.heading_mean,
        heading_rank_episode_std_cutoff_radius_m=aggregate.heading_std,
        centroid_rank_mean_distance_m=aggregate.centroid_metric_mean,
        centroid_rank_episode_std_distance_m=aggregate.centroid_metric_std,
        heading_rank_mean_deviation_rad=aggregate.heading_metric_mean,
        heading_rank_episode_std_deviation_rad=aggregate.heading_metric_std,
        position_entropy_population_mean_m=aggregate.position_entropy_mean,
        position_entropy_episode_std_m=aggregate.position_entropy_std,
        velocity_entropy_population_mean_m_per_s=aggregate.velocity_entropy_mean,
        velocity_entropy_episode_std_m_per_s=aggregate.velocity_entropy_std,
    )
    return Path(path)


def _time_suffix(max_time_seconds: Optional[float]) -> str:
    if max_time_seconds is None:
        return ""
    text = "{:.12g}".format(float(max_time_seconds)).replace(".", "p")
    return "_tmax{}s".format(text)


def figure_filename(
    num_agents: int,
    ranking_type: str,
    *,
    max_time_seconds: Optional[float] = None,
    with_entropy_panels: bool = False,
    statistic: str = "mean",
) -> str:
    descriptors = {
        "centroid_distance": "centroid_rank",
        "mean_heading_difference": "mean_heading_rank",
    }
    if ranking_type not in descriptors:
        raise ValueError("unknown ranking type {!r}".format(ranking_type))
    if statistic not in {"mean", "std"}:
        raise ValueError("statistic must be 'mean' or 'std'")
    suffix = _time_suffix(max_time_seconds)
    if with_entropy_panels:
        suffix += "_with_entropies"
    return "N{}_{}_{}_cutoff_radius_heatmap{}.png".format(
        int(num_agents), descriptors[ranking_type], statistic, suffix
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure_atomic(figure: Any, output: Path, dpi: int) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp.png")
    try:
        figure.savefig(temporary, dpi=int(dpi), facecolor="white")
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _ranking_labels(ranking_type: str) -> Tuple[str, str, str]:
    if ranking_type == "centroid_distance":
        return (
            "centroid-distance rank (left = nearest)",
            "distance to swarm centroid",
            "centroid-distance ranked",
        )
    if ranking_type == "mean_heading_difference":
        return (
            "mean-heading-deviation rank (left = closest)",
            "|heading - circular swarm mean heading|",
            "mean-heading-deviation ranked",
        )
    raise ValueError("unknown ranking type {!r}".format(ranking_type))


def render_heatmap(
    aggregate: PopulationAggregate,
    ranking_type: str,
    output: Path,
    *,
    shared_vmax: float,
    dpi: int = 180,
    max_time_seconds: Optional[float] = None,
    statistic: str = "mean",
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    values = aggregate.cutoff_values(ranking_type, statistic)
    display_steps = time_window_steps(
        aggregate.horizon, aggregate.dt, max_time_seconds
    )
    values = values[:display_steps]
    max_time = display_steps * aggregate.dt
    xlabel, ordering, short_name = _ranking_labels(ranking_type)
    ranks = np.arange(1, aggregate.num_agents + 1)
    rank_edges = np.arange(aggregate.num_agents + 1, dtype=np.float64) + 0.5
    time_edges = np.arange(display_steps + 1, dtype=np.float64) * aggregate.dt

    figure, axis = plt.subplots(figsize=(9.2, 6.2), facecolor="white")
    mesh = axis.pcolormesh(
        rank_edges,
        time_edges,
        values,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=float(shared_vmax)),
        shading="flat",
        rasterized=True,
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.025)
    colorbar.set_label(
        "population mean cutoff radius [m]"
        if statistic == "mean"
        else "episode standard deviation of cutoff radius [m]"
    )
    tick_stride = 1 if aggregate.num_agents <= 20 else 2
    axis.set_xticks(ranks[::tick_stride])
    axis.set_xlim(0.5, aggregate.num_agents + 0.5)
    axis.set_ylim(0.0, max_time)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("physical time [s]")
    axis.set_title(
        "N={} deterministic RL — population {} cutoff radius{}\n"
        "{}; rank each episode/time before aggregation".format(
            aggregate.num_agents,
            statistic,
            " (0–{:g} s view)".format(max_time)
            if max_time_seconds is not None
            else "",
            short_name,
        ),
        fontsize=12,
    )
    axis.text(
        0.012,
        0.985,
        "ordering metric: {}\n{} independent episodes | overall {}={:.2f} m".format(
            ordering,
            aggregate.episode_count,
            statistic,
            float(np.mean(values)),
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.52, "edgecolor": "none"},
    )
    figure.tight_layout()
    try:
        _save_figure_atomic(figure, output, dpi)
    finally:
        plt.close(figure)
    return {
        "num_agents": aggregate.num_agents,
        "ranking_type": ranking_type,
        "statistic": statistic,
        "path": str(Path(output).resolve()),
        "sha256": _sha256(output),
        "file_size_bytes": Path(output).stat().st_size,
        "height_time_cells": display_steps,
        "width_rank_cells": aggregate.num_agents,
        "display_max_time_seconds": max_time,
    }


def entropy_axis_limits(
    aggregates: Sequence[PopulationAggregate],
    max_time_seconds: float,
) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    for name in ("position_entropy_mean", "velocity_entropy_mean"):
        series = []
        for aggregate in aggregates:
            points = time_window_steps(
                aggregate.horizon, aggregate.dt, max_time_seconds
            ) + 1
            mean = np.asarray(getattr(aggregate, name))[:points]
            std = np.asarray(getattr(aggregate, name.replace("mean", "std")))[:points]
            series.extend((np.maximum(0.0, mean - std), mean + std))
        minimum = min(float(np.min(values)) for values in series)
        maximum = max(float(np.max(values)) for values in series)
        span = maximum - minimum
        padding = max(0.06 * span, 0.01 * max(abs(maximum), 1.0), 1e-6)
        limits[name] = (max(0.0, minimum - padding), maximum + padding)
    return limits


def render_heatmap_with_entropies(
    aggregate: PopulationAggregate,
    ranking_type: str,
    output: Path,
    *,
    shared_vmax: float,
    entropy_limits: Mapping[str, Tuple[float, float]],
    max_time_seconds: float = 30.0,
    dpi: int = 180,
    statistic: str = "mean",
) -> Dict[str, Any]:
    """Render two left-facing entropy panels beside the unchanged heatmap."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    display_steps = time_window_steps(
        aggregate.horizon, aggregate.dt, max_time_seconds
    )
    max_time = display_steps * aggregate.dt
    values = aggregate.cutoff_values(ranking_type, statistic)[:display_steps]
    entropy_points = display_steps + 1
    entropy_time = np.arange(entropy_points, dtype=np.float64) * aggregate.dt
    xlabel, ordering, short_name = _ranking_labels(ranking_type)

    rank_edges = np.arange(aggregate.num_agents + 1, dtype=np.float64) + 0.5
    time_edges = np.arange(display_steps + 1, dtype=np.float64) * aggregate.dt
    figure = plt.figure(figsize=(14.8, 6.2), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.4, 1.4, 7.2, 0.30),
        left=0.055,
        right=0.965,
        bottom=0.14,
        top=0.80,
        wspace=0.23,
    )
    velocity_axis = figure.add_subplot(grid[0, 0])
    position_axis = figure.add_subplot(grid[0, 1], sharey=velocity_axis)
    heatmap_axis = figure.add_subplot(grid[0, 2], sharey=velocity_axis)
    colorbar_axis = figure.add_subplot(grid[0, 3])

    panel_specs = (
        (
            velocity_axis,
            aggregate.velocity_entropy_mean[:entropy_points],
            aggregate.velocity_entropy_std[:entropy_points],
            "(a) Velocity entropy",
            "$\\sigma_v$ [m/s]  (larger $\\leftarrow$)",
            "velocity_entropy_mean",
            "#2C7FB8",
        ),
        (
            position_axis,
            aggregate.position_entropy_mean[:entropy_points],
            aggregate.position_entropy_std[:entropy_points],
            "(b) Position entropy",
            "$\\sigma_p$ [m]  (larger $\\leftarrow$)",
            "position_entropy_mean",
            "#D95F0E",
        ),
    )
    for axis, mean, std, title, label, limit_key, color in panel_specs:
        mean = np.asarray(mean)
        std = np.asarray(std)
        axis.fill_betweenx(
            entropy_time,
            np.maximum(0.0, mean - std),
            mean + std,
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        axis.plot(mean, entropy_time, color=color, linewidth=1.65)
        low, high = entropy_limits[limit_key]
        axis.set_xlim(high, low)  # positive/larger values point to the left
        axis.set_ylim(0.0, max_time)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.grid(color="#D9D9D9", linewidth=0.55, alpha=0.75)
        axis.set_title(title, fontsize=10, pad=7)
        axis.set_xlabel(label, fontsize=8.5)
        axis.tick_params(axis="both", labelsize=8)
        axis.text(
            0.5,
            0.985,
            "mean ± episode std",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#333333",
        )
    velocity_axis.set_ylabel("physical time [s]")
    position_axis.tick_params(axis="y", labelleft=False)

    mesh = heatmap_axis.pcolormesh(
        rank_edges,
        time_edges,
        values,
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=float(shared_vmax)),
        shading="flat",
        rasterized=True,
    )
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label(
        "population mean cutoff radius [m]"
        if statistic == "mean"
        else "episode standard deviation of cutoff radius [m]"
    )
    ranks = np.arange(1, aggregate.num_agents + 1)
    tick_stride = 1 if aggregate.num_agents <= 20 else 2
    heatmap_axis.set_xticks(ranks[::tick_stride])
    heatmap_axis.set_xlim(0.5, aggregate.num_agents + 0.5)
    heatmap_axis.set_ylim(0.0, max_time)
    heatmap_axis.set_xlabel(xlabel)
    heatmap_axis.set_ylabel("physical time [s]")
    heatmap_axis.set_title("(c) Cutoff-radius heatmap", fontsize=10, pad=7)
    heatmap_axis.text(
        0.012,
        0.985,
        "ordering metric: {}\n{} episodes | overall {}={:.2f} m".format(
            ordering, aggregate.episode_count, statistic, float(np.mean(values))
        ),
        transform=heatmap_axis.transAxes,
        va="top",
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.52, "edgecolor": "none"},
    )
    figure.suptitle(
        "N={} deterministic RL — population cutoff {} (0–{:g} s)\n"
        "velocity/position entropy means + {} cutoff radius".format(
            aggregate.num_agents, statistic, max_time, short_name
        ),
        fontsize=12,
        y=0.965,
    )
    try:
        _save_figure_atomic(figure, output, dpi)
    finally:
        plt.close(figure)
    return {
        "num_agents": aggregate.num_agents,
        "ranking_type": ranking_type,
        "statistic": statistic,
        "path": str(Path(output).resolve()),
        "sha256": _sha256(output),
        "file_size_bytes": Path(output).stat().st_size,
        "panel_count_excluding_colorbar": 3,
        "panel_order_left_to_right": [
            "velocity_entropy",
            "position_entropy",
            "cutoff_radius_heatmap",
        ],
        "entropy_positive_direction": "left",
        "entropy_time_points": entropy_points,
        "height_time_cells": display_steps,
        "width_rank_cells": aggregate.num_agents,
        "display_max_time_seconds": max_time,
    }


def _shared_vmax(
    aggregates: Sequence[PopulationAggregate],
    max_time_seconds: Optional[float],
    statistic: str,
) -> float:
    maxima = []
    for aggregate in aggregates:
        steps = time_window_steps(
            aggregate.horizon, aggregate.dt, max_time_seconds
        )
        for ranking in RANKING_TYPES:
            maxima.append(
                float(np.max(aggregate.cutoff_values(ranking, statistic)[:steps]))
            )
    maximum = max(maxima)
    return max(1e-12, maximum)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def generate_standard_figure_sets(
    aggregates: Mapping[int, PopulationAggregate],
    output_dir: Path,
    *,
    detail_seconds: float = 30.0,
    views: Sequence[str] = ("full", "detail", "entropy"),
    statistic: str = "mean",
    dpi: int = 180,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate four N/ranking figures per requested view.

    With the default N=20/N=40 mapping and all three views this writes twelve
    figures: full, 30-second detail, and 30-second detail with two entropy panels.
    """

    if statistic not in {"mean", "std"}:
        raise ValueError("statistic must be 'mean' or 'std'")
    unknown = set(views).difference({"full", "detail", "entropy"})
    if unknown:
        raise ValueError("unknown figure views: {}".format(sorted(unknown)))
    output_dir = Path(output_dir).resolve()
    figure_dir = output_dir / "figures"
    derived_dir = output_dir / "derived"
    ordered = [aggregates[key] for key in sorted(aggregates)]
    resolved_source_root = (
        Path(source_root).resolve()
        if source_root is not None
        else _common_source_root(ordered)
    )
    source_path_base = (
        "source_bundle" if source_root is not None else "common_source_root"
    )
    for aggregate in ordered:
        save_aggregate(
            aggregate,
            derived_dir
            / "N{}_population_ranked_cutoff_radius.npz".format(
                aggregate.num_agents
            ),
            source_root=resolved_source_root,
            source_path_base=source_path_base,
        )

    figures: List[Dict[str, Any]] = []
    view_specs = []
    if "full" in views:
        view_specs.append(("full", None, False))
    if "detail" in views:
        view_specs.append(("detail", float(detail_seconds), False))
    if "entropy" in views:
        view_specs.append(("entropy", float(detail_seconds), True))
    for view_name, max_time, with_entropy in view_specs:
        vmax = _shared_vmax(ordered, max_time, statistic)
        limits = (
            entropy_axis_limits(ordered, float(max_time)) if with_entropy else None
        )
        for aggregate in ordered:
            for ranking_type in RANKING_TYPES:
                filename = figure_filename(
                    aggregate.num_agents,
                    ranking_type,
                    max_time_seconds=max_time,
                    with_entropy_panels=with_entropy,
                    statistic=statistic,
                )
                output = figure_dir / filename
                if with_entropy:
                    record = render_heatmap_with_entropies(
                        aggregate,
                        ranking_type,
                        output,
                        shared_vmax=vmax,
                        entropy_limits=limits,
                        max_time_seconds=float(max_time),
                        dpi=dpi,
                        statistic=statistic,
                    )
                else:
                    record = render_heatmap(
                        aggregate,
                        ranking_type,
                        output,
                        shared_vmax=vmax,
                        dpi=dpi,
                        max_time_seconds=max_time,
                        statistic=statistic,
                    )
                record["view"] = view_name
                record["path"] = output.relative_to(output_dir).as_posix()
                figures.append(record)

    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistic_rendered": statistic,
        "detail_seconds": float(detail_seconds),
        "views": list(views),
        "analysis_output_root": ".",
        "source_path_base": source_path_base,
        "path_semantics": {
            "source_paths": (
                "POSIX paths relative to the source bundle"
                if source_path_base == "source_bundle"
                else "POSIX paths relative to the caller's common source root"
            ),
            "output_paths": "POSIX paths relative to analysis_output_root",
        },
        "figure_count": len(figures),
        "figures": figures,
        "populations": {
            str(aggregate.num_agents): {
                "episode_count": aggregate.episode_count,
                "horizon": aggregate.horizon,
                "dt_seconds": aggregate.dt,
                "seeds": list(aggregate.seeds),
                "source_paths": list(
                    _relative_source_paths(
                        aggregate.source_paths, resolved_source_root
                    )
                ),
                "max_metric_validation_errors": dict(
                    aggregate.max_metric_errors
                ),
                "pointer_mask_mismatch_count": 0,
                "population_std_ddof": 0,
                "derived_path": (
                    derived_dir
                    / "N{}_population_ranked_cutoff_radius.npz".format(
                        aggregate.num_agents
                    )
                ).relative_to(output_dir).as_posix(),
            }
            for aggregate in ordered
        },
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream deterministic full-episode artifacts and render ranked "
            "cutoff-radius population heatmaps."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="episode .npz or directory; repeat for multiple roots",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-agents", type=int, nargs="+", default=(20, 40), metavar="N"
    )
    parser.add_argument("--detail-seconds", type=float, default=30.0)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("full", "detail", "entropy"),
        default=("full", "detail", "entropy"),
    )
    parser.add_argument("--statistic", choices=("mean", "std"), default="mean")
    parser.add_argument("--expected-count-per-n", type=int)
    parser.add_argument("--dt", type=float, help="override artifact dt metadata")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="permit artifacts not marked deterministic (not recommended)",
    )
    parser.add_argument(
        "--allow-duplicate-seeds",
        action="store_true",
        help="permit duplicate seed metadata within an N population",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paths = discover_episode_paths(args.input)
    if not args.allow_nondeterministic:
        from .bundle import deterministic_paths

        paths = deterministic_paths(paths)
    aggregates = aggregate_paths(
        paths,
        num_agents=args.num_agents,
        dt_override=args.dt,
        expected_count=args.expected_count_per_n,
        require_deterministic=not args.allow_nondeterministic,
        reject_duplicate_seeds=not args.allow_duplicate_seeds,
    )
    manifest = generate_standard_figure_sets(
        aggregates,
        args.output_dir,
        detail_seconds=args.detail_seconds,
        views=args.views,
        statistic=args.statistic,
        dpi=args.dpi,
    )
    print(
        "[analysis] wrote {} figures and {} derived population files under {}".format(
            manifest["figure_count"], len(aggregates), args.output_dir.resolve()
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
