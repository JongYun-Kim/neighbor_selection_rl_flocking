"""Unified-CLI adapter for ranked deterministic population heatmaps."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from eval.artifacts import atomic_write_json

from .bundle import (
    deterministic_paths,
    discover_bundle,
    expected_episode_count,
    manifest_agent_counts,
    manifest_number,
    resolve_analysis_output,
)
from .population import aggregate_paths, generate_standard_figure_sets


def _canonical_n_paths(paths, requested):
    selected = [
        path
        for path in paths
        if any(
            path.parent.parent.name == "N{}".format(value)
            for value in requested
        )
    ]
    return selected if selected else list(paths)


def _default_view_directory(
    root: Path,
    t_max_seconds: Optional[float],
    with_entropies: bool,
) -> Path:
    if t_max_seconds is None:
        name = "full"
    else:
        time_text = "{:.12g}".format(float(t_max_seconds)).replace(".", "p")
        name = "tmax{}s".format(time_text)
    if with_entropies:
        name += "_with_entropies"
    return root / "analysis" / "ranked_heatmaps" / name


def run_from_bundle(
    run,
    output=None,
    t_max_seconds: Optional[float] = None,
    with_entropies: bool = False,
) -> Dict[str, Any]:
    """Create one four-figure N20/N40 set from a population bundle.

    The rank axis is recomputed within every episode and action time before the
    online population moments are updated.  Mean and episode-standard-deviation
    arrays are always saved; the rendered heatmaps show the requested mean view.
    """

    root, source_manifest, paths = discover_bundle(run)
    available = set(manifest_agent_counts(source_manifest))
    requested = (20, 40)
    if available and not set(requested).issubset(available):
        raise ValueError(
            "heatmaps require N=20 and N=40; bundle declares {}".format(
                sorted(available)
            )
        )
    paths = deterministic_paths(paths)
    paths = _canonical_n_paths(paths, requested)
    dt = manifest_number(source_manifest, ("dt", "dt_seconds"))
    if dt is None:
        raise ValueError(
            "bundle manifest has no unambiguous dt/dt_seconds; physical-time "
            "heatmaps require it to be persisted by evaluation"
        )
    aggregates = aggregate_paths(
        paths,
        num_agents=requested,
        dt_override=dt,
        expected_count=expected_episode_count(source_manifest),
        require_deterministic=True,
        reject_duplicate_seeds=True,
    )

    detail = None if t_max_seconds is None else float(t_max_seconds)
    if with_entropies and detail is None:
        durations = {
            round(item.horizon * item.dt, 12) for item in aggregates.values()
        }
        if len(durations) != 1:
            raise ValueError(
                "full entropy view requires the same physical duration for N20/N40"
            )
        detail = float(next(iter(durations)))
    if with_entropies:
        views = ("entropy",)
    elif detail is None:
        views = ("full",)
    else:
        views = ("detail",)

    default_output = _default_view_directory(
        root, t_max_seconds, with_entropies
    )
    output_dir = resolve_analysis_output(
        root,
        source_manifest,
        output,
        *default_output.relative_to(root).parts,
    )
    result = generate_standard_figure_sets(
        aggregates,
        output_dir,
        detail_seconds=30.0 if detail is None else detail,
        views=views,
        statistic="mean",
        source_root=root,
    )
    source_bundle = Path(os.path.relpath(root, output_dir)).as_posix()
    result.update(
        {
            "source_bundle": source_bundle,
            "source_run_id": source_manifest.get("run_id"),
            "source_fingerprint": source_manifest.get("fingerprint"),
            "output_dir": ".",
            "t_max_seconds": detail,
            "with_entropies": bool(with_entropies),
        }
    )
    result["path_semantics"].update({
        "source_bundle": "POSIX path relative to analysis_output_root",
        "output_dir": "analysis_output_root itself",
    })
    atomic_write_json(output_dir / "manifest.json", result)
    return {
        "source_bundle": str(root),
        "source_run_id": source_manifest.get("run_id"),
        "source_fingerprint": source_manifest.get("fingerprint"),
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "figure_count": result["figure_count"],
        "figures": result["figures"],
        "populations": {
            key: {
                "episode_count": value["episode_count"],
                "horizon": value["horizon"],
                "dt_seconds": value["dt_seconds"],
                "derived_path": value["derived_path"],
            }
            for key, value in result["populations"].items()
        },
        "t_max_seconds": detail,
        "with_entropies": bool(with_entropies),
    }


__all__ = ["run_from_bundle"]
