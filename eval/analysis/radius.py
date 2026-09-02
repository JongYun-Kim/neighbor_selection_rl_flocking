"""Single-episode cutoff-circle previews and animations."""

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from eval.artifacts import atomic_write_json

from .core import (
    EpisodeData,
    centered_positions,
    rank_episode_cutoff_radii,
    require_deterministic_episode,
    resolve_dt,
    state_polarization,
    validate_pointer_mask,
)
from .population import load_episode_data


RADIUS_SCHEMA_VERSION = "dynamic-k-single-episode-radius-v1"


def derive_radius_arrays(episode: EpisodeData) -> Dict[str, np.ndarray]:
    ranked = rank_episode_cutoff_radii(
        episode.states, episode.pointer_actions
    )
    centered = centered_positions(episode.states)
    external = episode.binary_actions.copy()
    diagonal = np.arange(episode.num_agents)
    external[:, diagonal, diagonal] = False
    ranked.update(
        {
            "centered": centered,
            "selected_k": external.sum(axis=2, dtype=np.int16),
            "state_phi": state_polarization(episode.states),
        }
    )
    return ranked


def frame_indices(start: int, end: int, stride: int) -> List[int]:
    """Return inclusive-final action frames for a half-open ``[start, end)``."""

    start = int(start)
    end = int(end)
    stride = int(stride)
    if start < 0 or end <= start or stride < 1:
        raise ValueError("invalid frame bounds or stride")
    frames = list(range(start, end, stride))
    if frames[-1] != end - 1:
        frames.append(end - 1)
    return frames


def _spatial_limit(centered: np.ndarray, radii: np.ndarray) -> float:
    action_centers = np.asarray(centered[:-1], dtype=np.float64)
    x_extent = np.max(np.abs(action_centers[..., 0]) + radii)
    y_extent = np.max(np.abs(action_centers[..., 1]) + radii)
    return max(1.0, float(max(x_extent, y_extent)) * 1.04)


def _figure_bundle(
    episode: EpisodeData, arrays: Mapping[str, np.ndarray], dt: float
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16.0, 9.0), facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.10, 1.38, 1.38),
        height_ratios=(1.0, 1.0),
        hspace=0.34,
        wspace=0.30,
    )
    centroid_axis = figure.add_subplot(grid[0, 0])
    heading_axis = figure.add_subplot(grid[1, 0])
    heading_metric_axis = heading_axis.twinx()
    spatial_axis = figure.add_subplot(grid[:, 1:])
    figure.subplots_adjust(top=0.91, bottom=0.09, left=0.055, right=0.965)

    radii = arrays["cutoff_radii"]
    centered = arrays["centered"]
    position_span = max(
        float(np.ptp(centered[..., 0])), float(np.ptp(centered[..., 1])), 1.0
    )
    radius_max = max(1.0, float(np.max(radii)))
    spatial_limit = _spatial_limit(centered, radii)
    style = {
        "agent_colors": plt.get_cmap("turbo")(
            np.linspace(0.03, 0.97, episode.num_agents)
        ),
        "radius_max": radius_max,
        "bar_max": max(
            radius_max, float(np.max(arrays["centroid_ranked_distances"]))
        ),
        "spatial_limit": spatial_limit,
        "arrow_length": max(1.0, min(12.0, 0.035 * position_span)),
        "trail_steps": max(1, int(round(5.0 / dt))),
    }
    axes = {
        "centroid": centroid_axis,
        "heading": heading_axis,
        "heading_metric": heading_metric_axis,
        "spatial": spatial_axis,
    }
    return figure, axes, style


def draw_radius_frame(
    figure: Any,
    axes: Mapping[str, Any],
    style: Mapping[str, Any],
    episode: EpisodeData,
    arrays: Mapping[str, np.ndarray],
    dt: float,
    time_index: int,
) -> None:
    """Draw one action-time frame with all circles and both contextual ranks."""

    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Circle

    time_index = int(time_index)
    if time_index < 0 or time_index >= episode.horizon:
        raise ValueError("time_index is outside the action horizon")
    for name in ("centroid", "heading", "heading_metric", "spatial"):
        axes[name].clear()
    centroid_axis = axes["centroid"]
    heading_axis = axes["heading"]
    heading_metric_axis = axes["heading_metric"]
    spatial_axis = axes["spatial"]

    num_agents = episode.num_agents
    rank = np.arange(num_agents)
    current_time = time_index * dt
    position = arrays["centered"][time_index]
    absolute_position = episode.states[time_index, :, :2]
    swarm_centroid = np.mean(absolute_position, axis=0)
    headings = episode.states[time_index, :, 4]
    radii = arrays["cutoff_radii"][time_index]
    pointer = episode.pointer_actions[time_index]
    selected_k = arrays["selected_k"][time_index]
    colors = style["agent_colors"]

    for agent in np.argsort(radii)[::-1]:
        radius = float(radii[agent])
        if radius > 1e-12:
            spatial_axis.add_patch(
                Circle(
                    position[agent],
                    radius,
                    fill=False,
                    edgecolor=colors[agent],
                    linewidth=0.75 if num_agents <= 20 else 0.55,
                    alpha=0.34 if num_agents <= 20 else 0.25,
                    zorder=1,
                )
            )
    tail_start = max(0, time_index - int(style["trail_steps"]))
    for agent in range(num_agents):
        trail = arrays["centered"][tail_start : time_index + 1, agent]
        spatial_axis.plot(
            trail[:, 0],
            trail[:, 1],
            color=colors[agent],
            linewidth=0.85 if num_agents <= 20 else 0.60,
            alpha=0.45,
            zorder=2,
        )
        target = int(pointer[agent])
        if target != agent:
            spatial_axis.plot(
                [position[agent, 0], position[target, 0]],
                [position[agent, 1], position[target, 1]],
                color=colors[agent],
                linewidth=0.85 if num_agents <= 20 else 0.55,
                alpha=0.65,
                zorder=3,
            )
    arrow_length = float(style["arrow_length"])
    spatial_axis.quiver(
        position[:, 0],
        position[:, 1],
        arrow_length * np.cos(headings),
        arrow_length * np.sin(headings),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#202020",
        width=0.0022,
        alpha=0.72,
        zorder=5,
    )
    node_size = 48 if num_agents <= 20 else 32
    spatial_axis.scatter(
        position[:, 0],
        position[:, 1],
        c=colors,
        s=node_size,
        edgecolors="black",
        linewidths=0.55,
        zorder=6,
    )
    self_agents = np.flatnonzero(pointer == np.arange(num_agents))
    if self_agents.size:
        spatial_axis.scatter(
            position[self_agents, 0],
            position[self_agents, 1],
            marker="x",
            s=node_size * 1.15,
            color="black",
            linewidths=1.0,
            zorder=7,
        )
    label_offset = 0.02 * float(style["spatial_limit"])
    for agent, coordinate in enumerate(position):
        label = spatial_axis.text(
            coordinate[0] + label_offset,
            coordinate[1] + label_offset,
            str(agent),
            fontsize=7.0 if num_agents <= 20 else 5.2,
            color="#151515",
            zorder=8,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.8, foreground="white")]
        )
    limit = float(style["spatial_limit"])
    spatial_axis.set_xlim(-limit, limit)
    spatial_axis.set_ylim(-limit, limit)
    spatial_axis.set_aspect("equal", adjustable="box")
    spatial_axis.set_xlabel("x - swarm centroid [m]")
    spatial_axis.set_ylabel("y - swarm centroid [m]")
    spatial_axis.set_title(
        "All-agent pointer cutoff circles\n"
        "circle/line color = ego agent; × = self pointer",
        fontsize=11,
    )
    spatial_axis.grid(alpha=0.18, linewidth=0.55)
    spatial_axis.text(
        0.012,
        0.985,
        "step={}/{} | t={:.2f} s | phi={:.4f}\n"
        "radius mean/median/range={:.1f}/{:.1f}/{:.1f}–{:.1f} m | self={}/{}\n"
        "absolute centroid=({:.1f}, {:.1f}) m".format(
            time_index,
            episode.horizon - 1,
            current_time,
            arrays["state_phi"][time_index],
            float(np.mean(radii)),
            float(np.median(radii)),
            float(np.min(radii)),
            float(np.max(radii)),
            self_agents.size,
            num_agents,
            swarm_centroid[0],
            swarm_centroid[1],
        ),
        transform=spatial_axis.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
        zorder=10,
    )

    annotation_offset = 0.012 * float(style["bar_max"])
    centroid_order = arrays["centroid_order"][time_index]
    centroid_radii = radii[centroid_order]
    centroid_k = selected_k[centroid_order]
    centroid_distance = np.linalg.norm(position[centroid_order], axis=1)
    centroid_axis.bar(
        rank,
        centroid_radii,
        color=colors[centroid_order],
        edgecolor="#303030",
        linewidth=0.30,
        width=0.82,
        label="cutoff radius",
    )
    centroid_axis.plot(
        rank,
        centroid_distance,
        color="#111111",
        marker="D",
        markersize=2.8 if num_agents <= 20 else 2.0,
        linewidth=0.75,
        label="distance to centroid",
        zorder=4,
    )
    r0 = episode.meta.get("r0")
    if r0 is not None and np.isfinite(float(r0)):
        centroid_axis.axhline(
            float(r0),
            color="#2CA02C",
            linestyle="--",
            linewidth=1.0,
            label="r0={:g} m".format(float(r0)),
        )
    for position_index in range(num_agents):
        centroid_axis.text(
            position_index,
            centroid_radii[position_index] + annotation_offset,
            "{:.0f}/{}".format(
                centroid_radii[position_index], int(centroid_k[position_index])
            ),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.3 if num_agents <= 20 else 3.9,
            clip_on=True,
        )
    centroid_axis.set_xlim(-0.7, num_agents - 0.3)
    centroid_axis.set_ylim(0.0, float(style["bar_max"]) * 1.16)
    centroid_axis.set_xticks(rank)
    centroid_axis.set_xticklabels(
        [str(int(agent)) for agent in centroid_order],
        rotation=90,
        fontsize=6.0 if num_agents <= 20 else 4.2,
    )
    centroid_axis.set_xlabel("agent ID: centroid-nearest → farthest")
    centroid_axis.set_ylabel("distance [m]")
    centroid_axis.set_title(
        "Current radius ordered by centroid distance\nlabel = radius [m] / selected k",
        fontsize=9.5,
    )
    centroid_axis.grid(axis="y", alpha=0.20, linewidth=0.5)
    centroid_axis.legend(loc="upper left", fontsize=6.3, framealpha=0.85)

    heading_order = arrays["heading_order"][time_index]
    heading_radii = radii[heading_order]
    heading_k = selected_k[heading_order]
    heading_degrees = np.degrees(
        arrays["heading_ranked_deviation"][time_index]
    )
    heading_axis.bar(
        rank,
        heading_radii,
        color=colors[heading_order],
        edgecolor="#303030",
        linewidth=0.30,
        width=0.82,
        label="cutoff radius",
    )
    if r0 is not None and np.isfinite(float(r0)):
        heading_axis.axhline(
            float(r0),
            color="#2CA02C",
            linestyle="--",
            linewidth=1.0,
            label="r0={:g} m".format(float(r0)),
        )
    heading_metric_axis.plot(
        rank,
        heading_degrees,
        color="#111111",
        marker="D",
        markersize=2.8 if num_agents <= 20 else 2.0,
        linewidth=0.75,
        label="heading deviation [deg]",
        zorder=4,
    )
    for position_index in range(num_agents):
        heading_axis.text(
            position_index,
            heading_radii[position_index] + annotation_offset,
            "{:.0f}/{}".format(
                heading_radii[position_index], int(heading_k[position_index])
            ),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.3 if num_agents <= 20 else 3.9,
            clip_on=True,
        )
    heading_axis.set_xlim(-0.7, num_agents - 0.3)
    heading_axis.set_ylim(0.0, float(style["radius_max"]) * 1.16)
    heading_axis.set_xticks(rank)
    heading_axis.set_xticklabels(
        [str(int(agent)) for agent in heading_order],
        rotation=90,
        fontsize=6.0 if num_agents <= 20 else 4.2,
    )
    heading_axis.set_xlabel("agent ID: mean-heading-nearest → farthest")
    heading_axis.set_ylabel("cutoff radius [m]")
    heading_axis.set_title(
        "Current radius ordered by heading deviation\nlabel = radius [m] / selected k",
        fontsize=9.5,
    )
    heading_axis.grid(axis="y", alpha=0.20, linewidth=0.5)
    heading_metric_axis.set_ylim(0.0, 189.0)
    heading_metric_axis.yaxis.set_label_position("right")
    heading_metric_axis.yaxis.tick_right()
    heading_metric_axis.tick_params(axis="y", labelsize=7)
    heading_metric_axis.set_ylabel("heading deviation [deg]", fontsize=7)
    handles, labels = heading_axis.get_legend_handles_labels()
    metric_handles, metric_labels = heading_metric_axis.get_legend_handles_labels()
    heading_axis.legend(
        handles + metric_handles,
        labels + metric_labels,
        loc="upper left",
        fontsize=6.3,
        framealpha=0.85,
    )

    figure.suptitle(
        "Deterministic dynamic-k neighbor radius — N={}, seed={}".format(
            num_agents, episode.meta.get("seed", "?")
        ),
        fontsize=14,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_stem(episode: EpisodeData) -> str:
    seed = episode.meta.get("seed", "unknown")
    return "N{}_seed{}_deterministic_radius".format(episode.num_agents, seed)


def render_previews(
    episode: EpisodeData,
    arrays: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    dt: float,
    steps: Optional[Sequence[int]] = None,
    dpi: int = 170,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = sorted({0, episode.horizon // 2, episode.horizon - 1})
    normalized = []
    for step in steps:
        value = int(step)
        if value < 0 or value >= episode.horizon:
            raise ValueError("preview step {} is outside [0, {})".format(value, episode.horizon))
        if value not in normalized:
            normalized.append(value)
    figure, axes, style = _figure_bundle(episode, arrays, dt)
    import matplotlib.pyplot as plt

    records = []
    try:
        for step in normalized:
            draw_radius_frame(figure, axes, style, episode, arrays, dt, step)
            output = output_dir / "{}_step{:05d}.png".format(
                _output_stem(episode), step
            )
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
            records.append(
                {
                    "step": step,
                    "physical_time_seconds": step * dt,
                    "path": str(output.resolve()),
                    "sha256": _sha256(output),
                    "file_size_bytes": output.stat().st_size,
                }
            )
    finally:
        plt.close(figure)
    return records


def render_animation(
    episode: EpisodeData,
    arrays: Mapping[str, np.ndarray],
    output: Path,
    *,
    dt: float,
    start_step: int = 0,
    end_step: Optional[int] = None,
    stride: int = 5,
    fps: int = 15,
    dpi: int = 130,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    end = episode.horizon if end_step is None else int(end_step)
    if end > episode.horizon:
        raise ValueError("end_step exceeds the action horizon")
    frames = frame_indices(start_step, end, stride)
    output = Path(output)
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError("animation output must end in .mp4 or .gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes, style = _figure_bundle(episode, arrays, dt)

    def update(step: int) -> Tuple[Any, ...]:
        draw_radius_frame(figure, axes, style, episode, arrays, dt, step)
        return tuple()

    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000.0 / int(fps), blit=False
    )
    if output.suffix.lower() == ".gif":
        writer = animation.PillowWriter(fps=int(fps))
    else:
        if not animation.writers.is_available("ffmpeg"):
            plt.close(figure)
            raise RuntimeError("matplotlib ffmpeg writer is unavailable; use .gif")
        writer = animation.FFMpegWriter(
            fps=int(fps), metadata={"title": "dynamic-k cutoff radii"}
        )
    temporary = output.with_name("." + output.stem + ".tmp" + output.suffix)
    try:
        movie.save(temporary, writer=writer, dpi=int(dpi))
        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(figure)
    return {
        "path": str(output.resolve()),
        "sha256": _sha256(output),
        "file_size_bytes": output.stat().st_size,
        "frame_count": len(frames),
        "start_step": frames[0],
        "end_step_inclusive": frames[-1],
        "stride": int(stride),
        "fps": int(fps),
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render deterministic dynamic-k cutoff circles through one episode."
    )
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preview-step",
        action="append",
        type=int,
        help="action step to save; repeat (default: first/middle/final)",
    )
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument(
        "--animation",
        type=Path,
        help="optional .mp4 or .gif output (relative paths use output-dir)",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--dt", type=float, help="override artifact dt metadata")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    episode = load_episode_data(args.episode, require_metrics=False)
    if not args.allow_nondeterministic:
        require_deterministic_episode(episode)
    mask_report = validate_pointer_mask(episode)
    dt = resolve_dt(episode, args.dt)
    arrays = derive_radius_arrays(episode)
    output_dir = args.output_dir.resolve()
    previews = []
    if not args.no_previews:
        previews = render_previews(
            episode,
            arrays,
            output_dir / "previews",
            dt=dt,
            steps=args.preview_step,
            dpi=args.dpi,
        )
    animation_record = None
    if args.animation is not None:
        animation_path = args.animation
        if not animation_path.is_absolute():
            animation_path = output_dir / animation_path
        animation_record = render_animation(
            episode,
            arrays,
            animation_path,
            dt=dt,
            start_step=args.start_step,
            end_step=args.end_step,
            stride=args.stride,
            fps=args.fps,
            dpi=args.dpi,
        )
    manifest = {
        "schema_version": RADIUS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_episode": str(args.episode.resolve()),
        "num_agents": episode.num_agents,
        "horizon": episode.horizon,
        "dt_seconds": dt,
        "seed": episode.meta.get("seed"),
        "mask_validation": mask_report,
        "metric_validation_errors": dict(episode.metric_errors),
        "previews": previews,
        "animation": animation_record,
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    print(
        "[analysis] wrote {} preview(s){} under {}".format(
            len(previews),
            " and one animation" if animation_record is not None else "",
            output_dir,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
