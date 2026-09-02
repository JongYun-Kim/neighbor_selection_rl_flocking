"""Analysis tools for canonical dynamic-k full-episode artifacts.

The package intentionally has no hard-coded checkpoint or ``test_field`` paths.
Use ``python -m eval.analysis population --help`` for N=20/N=40 population
figures and ``python -m eval.analysis radius --help`` for one-episode previews.
"""

from .core import (
    EpisodeData,
    centered_positions,
    centroid_distance_ranking,
    coerce_episode,
    compute_cutoff_radii,
    mean_heading_deviation_ranking,
    mean_heading_ranking,
    pointer_actions_to_mask,
    rank_episode_cutoff_radii,
    resolve_dt,
    state_entropy_series,
    state_polarization,
    time_window_steps,
    validate_pointer_mask,
    wrap_angle,
)
from .population import (
    OnlineMoments,
    PopulationAggregate,
    PopulationBuilder,
    aggregate_paths,
    figure_filename,
    generate_standard_figure_sets,
)

__all__ = [
    "EpisodeData",
    "OnlineMoments",
    "PopulationAggregate",
    "PopulationBuilder",
    "aggregate_paths",
    "centered_positions",
    "centroid_distance_ranking",
    "coerce_episode",
    "compute_cutoff_radii",
    "figure_filename",
    "generate_standard_figure_sets",
    "mean_heading_deviation_ranking",
    "mean_heading_ranking",
    "pointer_actions_to_mask",
    "rank_episode_cutoff_radii",
    "resolve_dt",
    "state_entropy_series",
    "state_polarization",
    "time_window_steps",
    "validate_pointer_mask",
    "wrap_angle",
]
