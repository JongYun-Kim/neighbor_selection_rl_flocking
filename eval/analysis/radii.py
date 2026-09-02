"""Unified-CLI adapter for one deterministic episode's cutoff circles."""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict

from eval.artifacts import atomic_write_json

from .bundle import (
    deterministic_paths, discover_bundle, manifest_number,
    resolve_analysis_output,
)
from .core import require_deterministic_episode, resolve_dt, validate_pointer_mask
from .population import load_episode_data
from .radius import (
    RADIUS_SCHEMA_VERSION,
    derive_radius_arrays,
    render_animation,
    render_previews,
)


def _metadata_matches(episode, n_agents: int, seed: int) -> bool:
    value = episode.meta.get("seed")
    return episode.num_agents == int(n_agents) and value is not None and int(value) == int(seed)


def _candidate_paths(paths, n_agents: int, seed: int):
    n_part = "N{}".format(int(n_agents))
    expected_stems = {
        "seed_{:05d}".format(int(seed)),
        "episode_{:05d}".format(int(seed)),
    }
    n_paths = [path for path in paths if path.parent.parent.name == n_part]
    selected = [path for path in n_paths if path.stem in expected_stems]
    # Metadata remains the final authority, but even its fallback must never
    # wander into a different population size.
    return selected if selected else n_paths


def run_from_bundle(
    run,
    n,
    seed,
    output=None,
    animation: bool = False,
) -> Dict[str, Any]:
    """Render first/middle/final previews and optionally an MP4 for one seed."""

    n_agents = int(n)
    seed = int(seed)
    root, source_manifest, paths = discover_bundle(run)
    paths = _candidate_paths(deterministic_paths(paths), n_agents, seed)
    matches = []
    for path in paths:
        episode = load_episode_data(path, require_metrics=False)
        if _metadata_matches(episode, n_agents, seed):
            try:
                require_deterministic_episode(episode)
            except ValueError:
                # Candidate discovery may deliberately fall back to metadata
                # for non-standard layouts.  A stochastic sibling with the
                # same N/seed is not an analysis failure; it is not a match.
                continue
            matches.append(episode)
    if not matches:
        raise FileNotFoundError(
            "no deterministic N={} seed={} episode in {}".format(
                n_agents, seed, root
            )
        )
    if len(matches) != 1:
        raise ValueError(
            "found {} deterministic N={} seed={} episodes; bundle is ambiguous".format(
                len(matches), n_agents, seed
            )
        )
    episode = matches[0]
    if episode.meta.get("r0") is None:
        bundle_r0 = manifest_number(
            source_manifest, ("r0",), num_agents=n_agents
        )
        if bundle_r0 is not None:
            episode.meta["r0"] = bundle_r0
    dt_override = manifest_number(
        source_manifest, ("dt", "dt_seconds"), num_agents=n_agents
    )
    dt = resolve_dt(episode, dt_override)
    mask_report = validate_pointer_mask(episode)
    arrays = derive_radius_arrays(episode)
    output_dir = resolve_analysis_output(
        root,
        source_manifest,
        output,
        "analysis",
        "radii",
        "N{}_seed_{:05d}".format(n_agents, seed),
    )
    previews = render_previews(
        episode, arrays, output_dir / "previews", dt=dt
    )
    animation_record = None
    if animation:
        animation_record = render_animation(
            episode,
            arrays,
            output_dir / "N{}_seed_{:05d}_cutoff_radii.mp4".format(
                n_agents, seed
            ),
            dt=dt,
        )
    portable_previews = []
    for record in previews:
        portable = dict(record)
        portable["path"] = Path(os.path.relpath(
            Path(record["path"]), output_dir
        )).as_posix()
        portable_previews.append(portable)
    portable_animation = None
    if animation_record is not None:
        portable_animation = dict(animation_record)
        portable_animation["path"] = Path(os.path.relpath(
            Path(animation_record["path"]), output_dir
        )).as_posix()
    source_bundle = Path(os.path.relpath(root, output_dir)).as_posix()
    source_episode = Path(os.path.relpath(episode.source, root)).as_posix()
    persisted = {
        "schema_version": RADIUS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_output_root": ".",
        "source_bundle": source_bundle,
        "source_run_id": source_manifest.get("run_id"),
        "source_fingerprint": source_manifest.get("fingerprint"),
        "source_episode": source_episode,
        "output_dir": ".",
        "path_semantics": {
            "source_bundle": "POSIX path relative to analysis_output_root",
            "source_episode": "POSIX path relative to source_bundle",
            "output_paths": "POSIX paths relative to analysis_output_root",
        },
        "num_agents": n_agents,
        "seed": seed,
        "horizon": episode.horizon,
        "dt_seconds": dt,
        "r0_m": episode.meta.get("r0"),
        "mask_validation": mask_report,
        "metric_validation_errors": dict(episode.metric_errors),
        "previews": portable_previews,
        "animation": portable_animation,
    }
    atomic_write_json(output_dir / "manifest.json", persisted)
    # Runtime callers get convenient absolute locations.  The on-disk
    # manifest above is the relocatable provenance record.
    return {
        **persisted,
        "source_bundle": str(root),
        "source_episode": str(episode.source),
        "output_dir": str(output_dir),
        "manifest": str(output_dir / "manifest.json"),
        "previews": previews,
        "animation": animation_record,
    }


__all__ = ["run_from_bundle"]
