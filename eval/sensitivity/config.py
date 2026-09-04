"""Strict, immutable configuration for Dynamic-k NN sensitivity suites."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Tuple

import yaml


CONFIG_SCHEMA_VERSION = "dynamic-k-nn-sensitivity-v1"
POLICY_NAMES = (
    "learned_deterministic",
    "learned_stochastic",
    "pure_acs",
)
OAT_FACTORS = (
    "num_agents",
    "minimum_turn_radius",
    "interaction_radius",
    "acs_gain_multiplier",
    "initial_position_bound",
)
CONFIG_DIRECTORY = Path(__file__).resolve().parent / "configs"
OAT_CONFIG_PATH = CONFIG_DIRECTORY / "oat.yaml"
REFINED_NEAR_ZERO_CONFIG_PATH = CONFIG_DIRECTORY / "refined_near_zero.yaml"


class SensitivityConfigError(ValueError):
    """Raised when a sensitivity YAML or a requested filter is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise SensitivityConfigError("duplicate YAML key: {!r}".format(key))
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _expect_mapping(value, label, keys):
    if not isinstance(value, dict):
        raise SensitivityConfigError("{} must be a mapping".format(label))
    expected = set(keys)
    actual = set(value)
    missing, unknown = expected - actual, actual - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing {}".format(sorted(missing)))
        if unknown:
            parts.append("unknown {}".format(sorted(unknown)))
        raise SensitivityConfigError("{}: {}".format(label, "; ".join(parts)))
    return value


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SensitivityConfigError("{} must be a number".format(label))
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SensitivityConfigError("{} must be finite and positive".format(label))
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SensitivityConfigError("{} must be an integer".format(label))
    value = int(value)
    if value < minimum:
        raise SensitivityConfigError("{} must be >= {}".format(label, minimum))
    return value


def _sequence(value, label):
    if not isinstance(value, list) or not value:
        raise SensitivityConfigError("{} must be a non-empty list".format(label))
    return value


def _same_number(left, right):
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _number_token(value):
    if isinstance(value, int):
        return str(value)
    token = format(float(value), ".15g").lower()
    return token.replace("-", "m").replace("+", "").replace(".", "p")


@dataclass(frozen=True)
class SensitivityBaseline:
    num_agents: int
    speed: float
    minimum_turn_radius: float
    interaction_radius: float
    acs_lambda: float
    acs_sigma: float
    initial_position_bound: float

    @property
    def max_turn_rate(self):
        return self.speed / self.minimum_turn_radius

    def to_dict(self):
        return {
            "num_agents": self.num_agents,
            "speed": self.speed,
            "minimum_turn_radius": self.minimum_turn_radius,
            "max_turn_rate": self.max_turn_rate,
            "interaction_radius": self.interaction_radius,
            "r0": self.interaction_radius,
            "c2_proximity_radius": self.interaction_radius,
            "acs_lambda": self.acs_lambda,
            "acs_sigma": self.acs_sigma,
            "initial_position_bound": self.initial_position_bound,
        }


@dataclass(frozen=True)
class SensitivitySetting:
    setting_id: str
    factor: str
    factor_value: float
    mode: str
    memberships: Tuple[str, ...]
    num_agents: int
    speed: float
    minimum_turn_radius: float
    max_turn_rate: float
    interaction_radius: float
    r0: float
    c2_proximity_radius: float
    acs_gain_multiplier: float
    acs_lambda: float
    acs_sigma: float
    initial_position_bound: float
    substeps_per_policy_interval: int
    dynamics_dt: float

    def to_dict(self):
        return {
            "setting_id": self.setting_id,
            "factor": self.factor,
            "factor_value": self.factor_value,
            "mode": self.mode,
            "memberships": list(self.memberships),
            "num_agents": self.num_agents,
            "speed": self.speed,
            "minimum_turn_radius": self.minimum_turn_radius,
            "max_turn_rate": self.max_turn_rate,
            "interaction_radius": self.interaction_radius,
            "r0": self.r0,
            "c2_proximity_radius": self.c2_proximity_radius,
            "acs_gain_multiplier": self.acs_gain_multiplier,
            "acs_lambda": self.acs_lambda,
            "acs_sigma": self.acs_sigma,
            "initial_position_bound": self.initial_position_bound,
            "substeps_per_policy_interval": self.substeps_per_policy_interval,
            "dynamics_dt": self.dynamics_dt,
        }


@dataclass(frozen=True)
class ResolvedSensitivityConfig:
    source_path: Path
    suite: str
    baseline: SensitivityBaseline
    macro_steps: int
    policy_interval_seconds: float
    max_heading_change_per_substep: Optional[float]
    factors: Tuple[str, ...]
    policies: Tuple[str, ...]
    seeds: Tuple[int, ...]
    settings: Tuple[SensitivitySetting, ...]

    @property
    def horizon(self):
        """Alias used by the evaluation artifact schema."""
        return self.macro_steps

    @property
    def setting_ids(self):
        return tuple(setting.setting_id for setting in self.settings)

    def setting_map(self):
        return {setting.setting_id: setting for setting in self.settings}

    def fingerprint_input(self):
        """Return path-independent, JSON-compatible resolved run semantics."""
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "suite": self.suite,
            "baseline": self.baseline.to_dict(),
            "macro_steps": self.macro_steps,
            "policy_interval_seconds": self.policy_interval_seconds,
            "max_heading_change_per_substep": self.max_heading_change_per_substep,
            "factors": list(self.factors),
            "policies": list(self.policies),
            "seeds": list(self.seeds),
            "settings": [setting.to_dict() for setting in self.settings],
        }


def _load_yaml(path):
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise SensitivityConfigError("cannot read {}: {}".format(path, error)) from error
    if not isinstance(data, dict):
        raise SensitivityConfigError("configuration root must be a mapping")
    return path, data


def _parse_baseline(value):
    keys = (
        "num_agents", "speed", "minimum_turn_radius", "interaction_radius",
        "acs_lambda", "acs_sigma", "initial_position_bound",
    )
    value = _expect_mapping(value, "baseline", keys)
    return SensitivityBaseline(
        num_agents=_integer(value["num_agents"], "baseline.num_agents", 2),
        speed=_positive_number(value["speed"], "baseline.speed"),
        minimum_turn_radius=_positive_number(
            value["minimum_turn_radius"], "baseline.minimum_turn_radius"),
        interaction_radius=_positive_number(
            value["interaction_radius"], "baseline.interaction_radius"),
        acs_lambda=_positive_number(value["acs_lambda"], "baseline.acs_lambda"),
        acs_sigma=_positive_number(value["acs_sigma"], "baseline.acs_sigma"),
        initial_position_bound=_positive_number(
            value["initial_position_bound"], "baseline.initial_position_bound"),
    )


def _parse_evaluation(value):
    value = _expect_mapping(
        value, "evaluation",
        ("policies", "seeds", "macro_steps", "policy_interval_seconds"),
    )
    policies = tuple(_sequence(value["policies"], "evaluation.policies"))
    if policies != POLICY_NAMES:
        raise SensitivityConfigError(
            "evaluation.policies must be exactly {}".format(list(POLICY_NAMES)))
    seed_spec = _expect_mapping(value["seeds"], "evaluation.seeds", ("start", "stop"))
    start = _integer(seed_spec["start"], "evaluation.seeds.start")
    stop = _integer(seed_spec["stop"], "evaluation.seeds.stop")
    if stop < start:
        raise SensitivityConfigError("evaluation.seeds.stop precedes start")
    macro_steps = _integer(value["macro_steps"], "evaluation.macro_steps", 1)
    interval = _positive_number(
        value["policy_interval_seconds"], "evaluation.policy_interval_seconds")
    if not _same_number(interval, 0.1):
        raise SensitivityConfigError("policy_interval_seconds must be 0.1 (10 Hz)")
    return policies, tuple(range(start, stop + 1)), macro_steps, interval


def _setting(baseline, factor, factor_value, mode, memberships, interval,
             max_heading_change=None, setting_id=None):
    try:
        resolved_factor_value = float(factor_value)
    except (OverflowError, TypeError, ValueError) as error:
        raise SensitivityConfigError("resolved factor value must be finite") from error
    resolved_factor_value = _positive_number(
        resolved_factor_value, "resolved factor value")
    num_agents = baseline.num_agents
    radius = baseline.minimum_turn_radius
    interaction = baseline.interaction_radius
    gain = 1.0
    bound = baseline.initial_position_bound
    if factor == "num_agents":
        num_agents = int(factor_value)
    elif factor == "minimum_turn_radius":
        radius = float(factor_value)
    elif factor == "interaction_radius":
        interaction = float(factor_value)
    elif factor == "acs_gain_multiplier":
        gain = float(factor_value)
    elif factor == "initial_position_bound":
        bound = float(factor_value)
    elif factor != "baseline":
        raise AssertionError(factor)

    max_turn_rate = _positive_number(
        baseline.speed / radius, "resolved max_turn_rate")
    acs_lambda = _positive_number(
        baseline.acs_lambda * gain, "resolved acs_lambda")
    acs_sigma = _positive_number(
        baseline.acs_sigma * gain, "resolved acs_sigma")
    if mode == "standard":
        substeps = 1
    elif mode == "refined":
        heading_ratio = max_turn_rate * interval / max_heading_change
        if not math.isfinite(heading_ratio) or heading_ratio < 0.0:
            raise SensitivityConfigError(
                "resolved refined substep ratio must be finite")
        substeps = max(1, int(math.ceil(heading_ratio)))
    else:
        raise AssertionError(mode)
    try:
        dynamics_dt = interval / substeps
    except OverflowError as error:
        raise SensitivityConfigError("resolved dynamics_dt must be finite") from error
    dynamics_dt = _positive_number(dynamics_dt, "resolved dynamics_dt")
    if setting_id is None:
        prefix = "refined_" if mode == "refined" else ""
        setting_id = "{}{}_{}".format(prefix, factor, _number_token(factor_value))
    return SensitivitySetting(
        setting_id=setting_id,
        factor=factor,
        factor_value=resolved_factor_value,
        mode=mode,
        memberships=tuple(memberships),
        num_agents=num_agents,
        speed=baseline.speed,
        minimum_turn_radius=radius,
        max_turn_rate=max_turn_rate,
        interaction_radius=interaction,
        r0=interaction,
        c2_proximity_radius=interaction,
        acs_gain_multiplier=gain,
        acs_lambda=acs_lambda,
        acs_sigma=acs_sigma,
        initial_position_bound=bound,
        substeps_per_policy_interval=substeps,
        dynamics_dt=dynamics_dt,
    )


def _parse_oat(data, baseline, interval):
    factors = _expect_mapping(data, "factors", OAT_FACTORS)
    baseline_values = {
        "num_agents": baseline.num_agents,
        "minimum_turn_radius": baseline.minimum_turn_radius,
        "interaction_radius": baseline.interaction_radius,
        "acs_gain_multiplier": 1.0,
        "initial_position_bound": baseline.initial_position_bound,
    }
    parsed = {}
    for factor in OAT_FACTORS:
        values = _sequence(factors[factor], "factors.{}".format(factor))
        if factor == "num_agents":
            resolved = tuple(
                _integer(item, "factors.num_agents", 2) for item in values)
        else:
            resolved = tuple(
                _positive_number(item, "factors.{}".format(factor))
                for item in values
            )
        if len(set(resolved)) != len(resolved):
            raise SensitivityConfigError("factors.{} contains duplicates".format(factor))
        if sum(_same_number(item, baseline_values[factor]) for item in resolved) != 1:
            raise SensitivityConfigError(
                "factors.{} must contain its baseline exactly once".format(factor))
        parsed[factor] = resolved

    settings = [
        _setting(
            baseline, "baseline", 1.0, "standard", OAT_FACTORS,
            interval, setting_id="baseline",
        )
    ]
    for factor in OAT_FACTORS:
        for value in parsed[factor]:
            if _same_number(value, baseline_values[factor]):
                continue
            settings.append(
                _setting(baseline, factor, value, "standard", (factor,), interval)
            )
    return OAT_FACTORS, tuple(settings), None


def _parse_refined(data, baseline, interval):
    refinement = _expect_mapping(
        data, "refinement",
        ("minimum_turn_radius", "max_heading_change_per_substep"),
    )
    values = tuple(
        _positive_number(item, "refinement.minimum_turn_radius")
        for item in _sequence(
            refinement["minimum_turn_radius"],
            "refinement.minimum_turn_radius",
        )
    )
    if len(set(values)) != len(values):
        raise SensitivityConfigError("refinement.minimum_turn_radius contains duplicates")
    max_heading_change = _positive_number(
        refinement["max_heading_change_per_substep"],
        "refinement.max_heading_change_per_substep",
    )
    factor = "minimum_turn_radius"
    settings = tuple(
        _setting(
            baseline, factor, value, "refined", (factor,), interval,
            max_heading_change=max_heading_change,
        )
        for value in values
    )
    return (factor,), settings, max_heading_change


def _selected(requested, available, label):
    if requested is None:
        return tuple(available)
    requested = tuple(requested)
    if not requested or len(set(requested)) != len(requested):
        raise SensitivityConfigError("{} filter must be non-empty and unique".format(label))
    unknown = set(requested).difference(available)
    if unknown:
        raise SensitivityConfigError("unknown {}: {}".format(label, sorted(unknown)))
    requested_set = set(requested)
    return tuple(item for item in available if item in requested_set)


def load_sensitivity_config(path, *, factors=None, settings=None,
                            policies=None, seeds=None):
    """Load, validate, expand, and filter one sensitivity YAML.

    Factor filtering keeps the OAT shared baseline.  A setting filter is an
    exact intersection and may omit it.  All output ordering follows the YAML
    contract rather than caller filter order, keeping fingerprint input stable.
    """
    source_path, data = _load_yaml(path)
    common = ("schema_version", "suite", "baseline", "evaluation")
    schema = data.get("schema_version")
    if schema != CONFIG_SCHEMA_VERSION:
        raise SensitivityConfigError(
            "schema_version must be {!r}".format(CONFIG_SCHEMA_VERSION))
    suite = data.get("suite")
    if suite == "oat":
        _expect_mapping(data, "configuration", common + ("factors",))
    elif suite == "refined_near_zero":
        _expect_mapping(data, "configuration", common + ("refinement",))
    else:
        raise SensitivityConfigError("suite must be oat or refined_near_zero")

    baseline = _parse_baseline(data["baseline"])
    configured_policies, configured_seeds, macro_steps, interval = _parse_evaluation(
        data["evaluation"])
    if suite == "oat":
        available_factors, all_settings, max_heading_change = _parse_oat(
            data["factors"], baseline, interval)
    else:
        available_factors, all_settings, max_heading_change = _parse_refined(
            data["refinement"], baseline, interval)

    if len({item.setting_id for item in all_settings}) != len(all_settings):
        raise SensitivityConfigError("resolved setting IDs are not unique")
    selected_factors = _selected(factors, available_factors, "factors")
    candidate_settings = tuple(
        item for item in all_settings
        if item.factor in selected_factors
        or (item.factor == "baseline" and suite == "oat")
    )
    available_setting_ids = tuple(item.setting_id for item in candidate_settings)
    selected_setting_ids = _selected(settings, available_setting_ids, "settings")
    setting_id_set = set(selected_setting_ids)
    selected_settings = tuple(
        item for item in candidate_settings if item.setting_id in setting_id_set)
    effective_factors = selected_factors
    if suite == "oat" and settings is not None:
        represented = {
            item.factor for item in selected_settings if item.factor != "baseline"
        }
        if represented:
            effective_factors = tuple(
                factor for factor in selected_factors if factor in represented
            )
    if not selected_settings:
        raise SensitivityConfigError("setting filters select no settings")

    selected_policies = _selected(policies, configured_policies, "policies")
    selected_seeds = _selected(seeds, configured_seeds, "seeds")
    return ResolvedSensitivityConfig(
        source_path=source_path,
        suite=suite,
        baseline=baseline,
        macro_steps=macro_steps,
        policy_interval_seconds=interval,
        max_heading_change_per_substep=max_heading_change,
        factors=effective_factors,
        policies=selected_policies,
        seeds=selected_seeds,
        settings=selected_settings,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "POLICY_NAMES",
    "OAT_FACTORS",
    "OAT_CONFIG_PATH",
    "REFINED_NEAR_ZERO_CONFIG_PATH",
    "SensitivityConfigError",
    "SensitivityBaseline",
    "SensitivitySetting",
    "ResolvedSensitivityConfig",
    "load_sensitivity_config",
]
