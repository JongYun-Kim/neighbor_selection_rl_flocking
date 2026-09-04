"""Compact statistics and four core plots for sensitivity episode rows."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from eval.artifacts import atomic_open
from eval.stats import wilson


REFERENCE_POLICY = "pure_acs"
SETTING_COLUMNS = ("mode", "factor", "factor_value", "setting_id")
FACTOR_VALUE_COLUMNS = {
    "num_agents": "num_agents",
    "minimum_turn_radius": "minimum_turn_radius",
    "interaction_radius": "interaction_radius",
    "acs_gain_multiplier": "acs_gain_multiplier",
    "initial_position_bound": "initial_position_bound",
}
DETAIL_COLUMNS = ("memberships",) + tuple(FACTOR_VALUE_COLUMNS.values())
REQUIRED_COLUMNS = SETTING_COLUMNS + (
    "policy", "seed", "horizon", "success", "t_fire", "t_fire_seconds",
    "J", "path",
)
SUMMARY_FILENAMES = {
    "settings": "settings.csv", "episodes": "episodes.csv",
    "aggregate": "aggregate.csv",
    "paired_vs_acs": "paired_vs_acs.csv", "dominance": "dominance.csv",
}
PLOT_FILENAMES = (
    "success_rate.png", "restricted_time.png",
    "paired_time_delta.png", "dominance.png",
)
POLICY_RANK = {
    "learned_deterministic": 0, "learned_stochastic": 1,
    REFERENCE_POLICY: 2,
}


@dataclass(frozen=True)
class SensitivityAnalysis:
    episodes: pd.DataFrame
    aggregate: pd.DataFrame
    paired: pd.DataFrame
    dominance: pd.DataFrame
    overall_dominance: pd.DataFrame
    factors: tuple


def _csv_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "summaries" / "episodes.csv"
    elif path.name == "manifest.json":
        path = path.parent / "summaries" / "episodes.csv"
    if not path.is_file():
        raise FileNotFoundError("sensitivity episodes CSV not found: {}".format(path))
    return path


def load_episode_rows(source):
    """Load a DataFrame, iterable of rows, CSV, manifest, or run directory."""
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    if isinstance(source, (str, Path)):
        return pd.read_csv(_csv_path(source))
    if isinstance(source, Mapping):
        raise TypeError("expected an iterable of episode rows, not one mapping")
    try:
        return pd.DataFrame(list(source))
    except TypeError as error:
        raise TypeError("unsupported sensitivity episode source") from error


def _source_factors(source):
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        manifest = path / "manifest.json"
    elif path.name == "manifest.json":
        manifest = path
    elif path.name == "episodes.csv" and path.parent.name == "summaries":
        manifest = path.parent.parent / "manifest.json"
    else:
        return None
    if not manifest.is_file():
        return None
    try:
        return tuple(json.loads(manifest.read_text(encoding="utf-8"))[
            "spec"]["config"]["factors"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid sensitivity manifest factors") from error


def _memberships(value):
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, str):
        try:
            items = json.loads(value) if value.strip().startswith("[") else re.split(
                r"[;,]", value
            )
        except json.JSONDecodeError as error:
            raise ValueError("memberships is not valid JSON") from error
    else:
        raise ValueError("memberships must be a list or comma-separated string")
    result = tuple(str(item).strip() for item in items if str(item).strip())
    if (not result or len(result) != len(set(result))
            or set(result).difference(FACTOR_VALUE_COLUMNS)):
        raise ValueError("memberships contains invalid sensitivity factors")
    return result


def _resolve_factors(frame, requested=None):
    represented = set(frame.loc[frame.factor != "baseline", "factor"])
    if requested is None:
        if represented:
            requested = tuple(
                factor for factor in FACTOR_VALUE_COLUMNS if factor in represented
            )
        else:
            memberships = set()
            for value in frame.loc[frame.factor == "baseline", "memberships"]:
                memberships.update(_memberships(value))
            requested = tuple(
                factor for factor in FACTOR_VALUE_COLUMNS if factor in memberships
            )
    result = tuple(requested)
    if (not result or len(result) != len(set(result))
            or set(result).difference(FACTOR_VALUE_COLUMNS)):
        raise ValueError("resolved sensitivity factors are invalid")
    if represented.difference(result):
        raise ValueError("sensitivity rows disagree with manifest factors")
    return result


def _sort(frame):
    if frame.empty:
        return frame.reset_index(drop=True)
    result = frame.copy()
    result["_rank"] = result.policy.map(POLICY_RANK).fillna(10)
    columns = [name for name in SETTING_COLUMNS if name in result]
    columns += ["_rank", "policy"] + (["seed"] if "seed" in result else [])
    return result.sort_values(columns, kind="stable").drop(columns="_rank").reset_index(drop=True)


def _normalize(source, policy_interval_seconds):
    frame = load_episode_rows(source)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame]
    if missing or frame.empty:
        raise ValueError(
            "sensitivity episodes are empty or missing columns {}".format(missing)
        )
    frame = frame.copy(deep=True)
    for name in ("setting_id", "factor", "mode", "policy", "path"):
        if frame[name].isna().any() or (frame[name].astype(str).str.strip() == "").any():
            raise ValueError("{} must contain non-empty values".format(name))
        frame[name] = frame[name].astype(str)
    if not set(frame["mode"]).issubset({"standard", "refined"}):
        raise ValueError("mode must be 'standard' or 'refined'")

    for name in ("factor_value", "t_fire", "t_fire_seconds", "J"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    for name, minimum in (("seed", None), ("horizon", 1), ("t_fire", None)):
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise ValueError("{} must contain finite integers".format(name))
        if minimum is not None and np.any(values < minimum):
            raise ValueError("{} must be >= {}".format(name, minimum))
        frame[name] = values.astype(np.int64)
    success_map = {True: 1, False: 0, 1: 1, 0: 0, "1": 1, "0": 0,
                   "true": 1, "false": 0, "True": 1, "False": 0}
    frame["success"] = frame.success.map(success_map)
    if frame.success.isna().any() or not np.isfinite(frame.factor_value).all():
        raise ValueError("success must be boolean/0/1 and factor_value must be finite")
    frame["success"] = frame.success.astype(np.int8)

    if "policy_interval_seconds" not in frame:
        frame["policy_interval_seconds"] = float(policy_interval_seconds)
    frame["policy_interval_seconds"] = pd.to_numeric(
        frame.policy_interval_seconds, errors="raise"
    )
    interval = frame.policy_interval_seconds.to_numpy(dtype=float)
    success = frame.success.to_numpy(dtype=bool)
    fire = frame.t_fire.to_numpy(dtype=np.int64)
    seconds = frame.t_fire_seconds.to_numpy(dtype=float)
    horizon = frame.horizon.to_numpy(dtype=np.int64)
    if (not np.isfinite(interval).all() or np.any(interval <= 0)
            or np.any(success & ((fire < 0) | (fire > horizon)))
            or not np.isfinite(seconds[success]).all()
            or not np.allclose(seconds[success], fire[success] * interval[success])):
        raise ValueError("successful t_fire_seconds must lie on a positive policy interval")
    frame["restricted_time_seconds"] = np.where(
        success, seconds, (horizon + 1) * interval
    )

    if frame.duplicated(["mode", "setting_id", "policy", "seed"]).any():
        raise ValueError("duplicate setting_id/policy/seed episode rows")
    if (frame.groupby(["mode", "setting_id"])[
            ["factor", "factor_value", "horizon"]
        ].nunique(dropna=False) > 1).any().any():
        raise ValueError("one setting_id maps to inconsistent metadata")

    baseline = frame.factor == "baseline"
    if baseline.any():
        absent = [name for name in DETAIL_COLUMNS if name not in frame]
        if absent:
            raise ValueError("shared baseline rows are missing columns {}".format(absent))
        for index in frame.index[baseline]:
            frame.at[index, "memberships"] = ",".join(
                _memberships(frame.at[index, "memberships"])
            )
        for name in FACTOR_VALUE_COLUMNS.values():
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(frame.loc[baseline, name]).all():
                raise ValueError("baseline {} must be finite".format(name))
    return _sort(frame)


def _aggregate(frame):
    rows, keys = [], list(SETTING_COLUMNS) + ["policy"]
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        n, successes = len(group), int(group.success.sum())
        failure_low, failure_high = wilson(n - successes, n)
        successful_time = group.loc[group.success == 1, "t_fire_seconds"]
        row = dict(zip(keys, key))
        row.update({name: group.iloc[0][name] for name in DETAIL_COLUMNS if name in group})
        row.update({
            "episode_count": n, "success_count": successes,
            "failure_count": n - successes, "success_rate": successes / n,
            "success_wilson_95_low": 1.0 - failure_high,
            "success_wilson_95_high": 1.0 - failure_low,
            "restricted_time_mean_seconds": float(group.restricted_time_seconds.mean()),
            "restricted_time_median_seconds": float(group.restricted_time_seconds.median()),
            "successful_time_median_seconds": (
                float(successful_time.median()) if len(successful_time) else np.nan
            ),
        })
        rows.append(row)
    return _sort(pd.DataFrame(rows))


def _paired(frame):
    columns = list(SETTING_COLUMNS) + [
        "policy", "seed", "success", "reference_success",
        "restricted_time_seconds", "reference_restricted_time_seconds",
        "delta_success", "delta_restricted_time_seconds", "outcome",
    ]
    rows = []
    for _, setting in frame.groupby(["mode", "setting_id"], sort=False):
        reference = setting.loc[setting.policy == REFERENCE_POLICY].set_index("seed")
        for policy, learned in setting.loc[
            setting.policy != REFERENCE_POLICY
        ].groupby("policy", sort=False):
            learned = learned.set_index("seed")
            if reference.empty or set(learned.index) != set(reference.index):
                continue
            for seed in sorted(reference.index):
                left, right = learned.loc[seed], reference.loc[seed]
                delta = float(left.restricted_time_seconds - right.restricted_time_seconds)
                rows.append({
                    **{name: left[name] for name in SETTING_COLUMNS},
                    "policy": policy, "seed": int(seed), "success": int(left.success),
                    "reference_success": int(right.success),
                    "restricted_time_seconds": float(left.restricted_time_seconds),
                    "reference_restricted_time_seconds": float(right.restricted_time_seconds),
                    "delta_success": int(left.success - right.success),
                    "delta_restricted_time_seconds": delta,
                    "outcome": "win" if delta < 0 else "loss" if delta > 0 else "tie",
                })
    return _sort(pd.DataFrame(rows, columns=columns))


def _dominance(aggregate, paired):
    details = [name for name in DETAIL_COLUMNS if name in aggregate]
    columns = list(SETTING_COLUMNS) + details + [
        "policy", "pair_count", "success_rate_delta",
        "restricted_time_delta_mean_seconds", "restricted_time_delta_median_seconds",
        "win_count", "tie_count", "loss_count", "point_estimate_weak_dominance",
    ]
    references = aggregate.loc[aggregate.policy == REFERENCE_POLICY].set_index(
        ["mode", "setting_id"]
    )
    rows = []
    for _, learned in aggregate.loc[aggregate.policy != REFERENCE_POLICY].iterrows():
        key = (learned["mode"], learned["setting_id"])
        group = paired.loc[
            (paired["mode"] == key[0]) & (paired.setting_id == key[1])
            & (paired.policy == learned.policy)
        ]
        if key not in references.index or len(group) != learned.episode_count:
            continue
        success_delta = float(learned.success_rate - references.loc[key].success_rate)
        deltas = group.delta_restricted_time_seconds.to_numpy(dtype=float)
        mean_delta = float(deltas.mean())
        rows.append({
            **{name: learned[name] for name in SETTING_COLUMNS + tuple(details)},
            "policy": learned.policy, "pair_count": len(group),
            "success_rate_delta": success_delta,
            "restricted_time_delta_mean_seconds": mean_delta,
            "restricted_time_delta_median_seconds": float(np.median(deltas)),
            "win_count": int((group.outcome == "win").sum()),
            "tie_count": int((group.outcome == "tie").sum()),
            "loss_count": int((group.outcome == "loss").sum()),
            "point_estimate_weak_dominance": bool(
                success_delta >= 0 and mean_delta <= 0
                and (success_delta > 0 or mean_delta < 0)
            ),
        })
    return _sort(pd.DataFrame(rows, columns=columns))


def _overall(dominance, active_factors):
    columns = [
        "mode", "policy", "factor_count", "setting_count",
        "dominant_setting_count", "dominant_setting_rate", "all_settings_dominant",
        "mean_success_rate_delta", "mean_restricted_time_delta_seconds",
    ]
    rows = []
    allowed = set(active_factors)
    for (mode, policy), group in dominance.groupby(["mode", "policy"], sort=False):
        count = int(group.point_estimate_weak_dominance.sum())
        factors = set(group.loc[group.factor != "baseline", "factor"])
        if "memberships" in group:
            for value in group.loc[group.factor == "baseline", "memberships"].dropna():
                factors.update(set(_memberships(value)).intersection(allowed))
        rows.append({
            "mode": mode, "policy": policy, "factor_count": len(factors),
            "setting_count": len(group), "dominant_setting_count": count,
            "dominant_setting_rate": count / len(group),
            "all_settings_dominant": bool(count == len(group)),
            "mean_success_rate_delta": float(group.success_rate_delta.mean()),
            "mean_restricted_time_delta_seconds": float(
                group.restricted_time_delta_mean_seconds.mean()
            ),
        })
    return _sort(pd.DataFrame(rows, columns=columns))


def analyze(source, policy_interval_seconds=0.1):
    """Return aggregate, identical-seed paired, and dominance summaries."""
    source_factors = _source_factors(source)
    episodes = _normalize(source, policy_interval_seconds)
    active_factors = _resolve_factors(episodes, source_factors)
    aggregate = _aggregate(episodes)
    paired = _paired(episodes)
    dominance = _dominance(aggregate, paired)
    return SensitivityAnalysis(
        episodes, aggregate, paired, dominance,
        _overall(dominance, active_factors), active_factors,
    )


def _settings(frame):
    preferred = list(SETTING_COLUMNS) + list(DETAIL_COLUMNS) + [
        "speed", "max_turn_rate", "acs_lambda", "acs_sigma",
        "substeps_per_policy_interval", "dynamics_dt", "horizon",
        "policy_interval_seconds",
    ]
    columns = [name for name in preferred if name in frame]
    return (frame[columns].drop_duplicates(["mode", "setting_id"])
            .sort_values(list(SETTING_COLUMNS), kind="stable")
            .reset_index(drop=True))


def _plot_view(frame, factors=None):
    """Expand the shared OAT baseline in memory, never in episode artifacts."""
    factors = _resolve_factors(frame, factors)
    allowed = set(factors)
    rows = []
    for _, row in frame.iterrows():
        if row.factor != "baseline":
            rows.append(row.to_dict())
            continue
        for factor in _memberships(row.memberships):
            if factor not in allowed:
                continue
            item = row.to_dict()
            item.update(factor=factor, factor_value=float(row[FACTOR_VALUE_COLUMNS[factor]]))
            rows.append(item)
    return pd.DataFrame(rows, columns=frame.columns)


def _require_pairs(result):
    learned = result.aggregate.loc[result.aggregate.policy != REFERENCE_POLICY]
    reference = result.aggregate.loc[result.aggregate.policy == REFERENCE_POLICY]
    keys = list(SETTING_COLUMNS) + ["policy"]
    if (learned.empty or reference.empty
            or set(learned[keys].itertuples(index=False, name=None))
            != set(result.dominance[keys].itertuples(index=False, name=None))):
        raise ValueError(
            "sensitivity plots require complete identical-seed Pure ACS pairs"
        )


def _plot(result, output):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _require_pairs(result)
    aggregate = _plot_view(result.aggregate, result.factors)
    dominance = _plot_view(result.dominance, result.factors)
    panels = list(aggregate[["mode", "factor"]].drop_duplicates().itertuples(
        index=False, name=None
    ))
    colors = {
        "learned_deterministic": "#0072B2", "learned_stochastic": "#E69F00",
        REFERENCE_POLICY: "#555555",
    }

    def line_plot(table, column, ylabel, filename, interval_columns=None, zero=False):
        figure, axes = plt.subplots(
            len(panels), 1, squeeze=False,
            figsize=(7.2, max(4.2, 3.5 * len(panels))),
        )
        for axis, (mode, factor) in zip(axes[:, 0], panels):
            selected = table.loc[
                (table["mode"] == mode) & (table["factor"] == factor)
            ]
            for policy, group in selected.groupby("policy", sort=False):
                group = group.sort_values("factor_value")
                values = group[column].to_numpy(dtype=float)
                options = {}
                if interval_columns:
                    low, high = (group[name].to_numpy(dtype=float) for name in interval_columns)
                    options["yerr"] = np.vstack([
                        np.maximum(values - low, 0), np.maximum(high - values, 0)
                    ])
                    options["capsize"] = 3
                axis.errorbar(
                    group.factor_value, values, marker="o", label=policy,
                    color=colors.get(policy), **options
                )
            if zero:
                axis.axhline(0, color="black", linewidth=0.8)
            axis.set(title="{}: {}".format(mode, factor), xlabel=factor, ylabel=ylabel)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)

    line_plot(
        aggregate, "success_rate", "C2 success rate", PLOT_FILENAMES[0],
        ("success_wilson_95_low", "success_wilson_95_high"),
    )
    line_plot(
        aggregate, "restricted_time_mean_seconds", "Restricted C2 time (s)",
        PLOT_FILENAMES[1],
    )
    line_plot(
        dominance, "restricted_time_delta_mean_seconds",
        "Learned - Pure ACS restricted time (s)", PLOT_FILENAMES[2], zero=True,
    )

    overall = result.overall_dominance
    labels = ["{}/{}".format(row.mode, row.policy) for row in overall.itertuples()]
    figure, axis = plt.subplots(figsize=(max(6.4, 1.3 * len(labels)), 4.4))
    bars = axis.bar(
        np.arange(len(labels)), overall.dominant_setting_rate,
        color=[colors.get(policy, "#777777") for policy in overall.policy],
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axis.set(ylim=(0, 1.05), ylabel="Weak-dominance setting rate",
             title="Overall point-estimate dominance over Pure ACS")
    for bar, row in zip(bars, overall.itertuples()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  "{}/{}".format(row.dominant_setting_count, row.setting_count),
                  ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output / PLOT_FILENAMES[3], dpi=160, bbox_inches="tight")
    plt.close(figure)


def expected_summary_frames(rows, policy_interval_seconds=0.1):
    result = analyze(rows, policy_interval_seconds)
    return {
        "settings": _settings(result.episodes),
        "episodes": result.episodes, "aggregate": result.aggregate,
        "paired_vs_acs": result.paired, "dominance": result.dominance,
    }


def _write_csv(path, frame):
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def write_summary_tables(rows, bundle, policy_interval_seconds=0.1):
    """Write resumable run summaries under ``bundle/summaries``."""
    directory = Path(bundle).expanduser().resolve() / "summaries"
    paths = {}
    for name, frame in expected_summary_frames(rows, policy_interval_seconds).items():
        path = directory / SUMMARY_FILENAMES[name]
        _write_csv(path, frame)
        paths[name] = str(path)
    return paths


def verify_summary_tables(rows, bundle, policy_interval_seconds=0.1):
    """Verify all runner summaries byte-for-byte without changing them."""
    directory = Path(bundle).expanduser().resolve() / "summaries"
    paths = {}
    for name, frame in expected_summary_frames(rows, policy_interval_seconds).items():
        path, expected = directory / SUMMARY_FILENAMES[name], io.StringIO()
        frame.to_csv(expected, index=False)
        if not path.is_file() or path.read_bytes() != expected.getvalue().encode():
            raise ValueError("sensitivity summary differs from episode rows: {}".format(path))
        paths[name] = str(path)
    return paths


def _analysis_output(source, output):
    if output is not None:
        return Path(output).expanduser().resolve()
    if not isinstance(source, (str, Path)):
        raise ValueError("output_dir is required for in-memory rows")
    path = Path(source).expanduser().resolve()
    root = path if path.is_dir() else path.parent
    if path.name == "manifest.json":
        root = path.parent
    elif root.name == "summaries":
        root = root.parent
    return root / "analysis"


def write_analysis(source, output_dir=None, policy_interval_seconds=0.1):
    """Write core analysis tables and exactly four PNG plot types."""
    result = analyze(source, policy_interval_seconds)
    output = _analysis_output(source, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "aggregate": result.aggregate, "paired_vs_acs": result.paired,
        "dominance": result.dominance,
        "overall_dominance": result.overall_dominance,
    }
    summaries = {}
    for name, frame in tables.items():
        path = output / "{}.csv".format(name)
        _write_csv(path, frame)
        summaries[name] = str(path)
    _plot(result, output)
    return {
        "output_dir": str(output), "episode_count": len(result.episodes),
        "setting_count": int(result.episodes.setting_id.nunique()),
        "summaries": summaries,
        "plots": {Path(name).stem: str(output / name) for name in PLOT_FILENAMES},
    }


__all__ = [
    "PLOT_FILENAMES", "SensitivityAnalysis", "analyze",
    "expected_summary_frames", "load_episode_rows", "verify_summary_tables",
    "write_analysis", "write_summary_tables",
]
