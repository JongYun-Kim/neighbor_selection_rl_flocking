"""Canonical C2 convergence protocol and episode judgment.

``MAIN_C2_V1`` is the criterion of record used by the main evaluation lane.
The strict inequalities and rolling-window indexing in this module intentionally
match the historical :mod:`eval.eval_c2` implementation exactly.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class C2ProtocolSpec:
    """Immutable parameters of the main C2 convergence criterion."""

    phi_goal: float
    alignment_window: int
    stability_window: int
    spatial_band_epsilon: float


MAIN_C2_V1 = C2ProtocolSpec(
    phi_goal=0.98,
    alignment_window=50,
    stability_window=300,
    spatial_band_epsilon=0.05,
)


def first_fire_c2(phi, s, comp, spec=MAIN_C2_V1):
    """Return the first C2 firing index, or ``-1`` when C2 never fires.

    Index zero is the initial state and subsequent indices are post-step states.
    Consequently, with the main 300-sample stability window the earliest
    possible firing index is 299.  Boundary comparisons are deliberately
    strict: ``phi > goal`` and relative spatial band ``< epsilon``.
    """
    pphi = pd.Series(phi)
    ps = pd.Series(s)
    pcomp = pd.Series(comp)
    align = (
        pphi.rolling(spec.alignment_window).min() > spec.phi_goal
    ).values
    cohesion = (
        pcomp.rolling(spec.stability_window).max() == 1
    ).values
    spatial_band = (
        (ps.rolling(spec.stability_window).max()
         - ps.rolling(spec.stability_window).min())
        / ps.rolling(spec.stability_window).mean()
    ).values
    with np.errstate(invalid="ignore"):
        converged = (
            align
            & cohesion
            & (spatial_band < spec.spatial_band_epsilon)
        )
    hits = np.flatnonzero(converged)
    return int(hits[0]) if hits.size else -1


def judge_episode_c2(phi, spatial_entropy, n_components, reward,
                     spec=MAIN_C2_V1):
    """Judge one episode and return the canonical summary fields.

    ``reward[0]`` corresponds to the initial state and is excluded.  On a
    success, J includes rewards from step 1 through the firing step, inclusive;
    on a failure J remains NaN.  ``np.nansum`` is retained for bit-compatible
    treatment of the initial/archived NaNs.
    """
    t_fire = first_fire_c2(phi, spatial_entropy, n_components, spec=spec)
    j_value = (
        float(-np.nansum(reward[1:t_fire + 1]))
        if t_fire >= 0 else np.nan
    )
    return {
        "t_fire": t_fire,
        "success": int(t_fire >= 0),
        "J": j_value,
    }


__all__ = [
    "C2ProtocolSpec",
    "MAIN_C2_V1",
    "first_fire_c2",
    "judge_episode_c2",
]
