"""Sparse standard/refined rollouts for configurable sensitivity studies."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from eval.artifacts import (
    atomic_save_npz,
    atomic_write_json,
    canonical_json,
    json_compatible,
    run_fingerprint,
)
from eval.common import _n_components, _pairwise_dist, build_config
from eval.policies import DynamicKNNInferencePolicy
from eval.provenance import git_snapshot, runtime_snapshot
from eval.protocol import MAIN_C2_V1, judge_episode_c2
from eval.run_support import (
    archive_checkpoint,
    checkpoint_identity,
    derive_action_seed,
    pure_acs_pointer,
    quarantine,
    state_hash,
    utc_now,
)
from eval.sensitivity.config import (
    POLICY_NAMES,
    ResolvedSensitivityConfig,
    SensitivitySetting,
)


PROTOCOL_ID = "main_c2_v1"
SUITE_ID = "sensitivity_v1"
SPARSE_SCHEMA_VERSION = "main-c2-sensitivity-sparse-1.0"
STATE_SERIES = ("phi", "s_ent", "v_ent", "n_comp_r0", "reward")
STEP_SERIES = (
    "mean_selected_neighbors",
    "mean_abs_control",
    "max_abs_control",
    "control_saturation_fraction",
)
EVIDENCE_FIELDS = STATE_SERIES + STEP_SERIES + ("initial_agent_states",)


def build_sensitivity_env_config(setting: SensitivitySetting, macro_horizon: int,
                                 obs_position_scale: str):
    """Resolve one setting onto the maintained fixed-horizon eval config."""
    physics_horizon = int(macro_horizon) * int(setting.substeps_per_policy_interval)
    cfg = build_config(
        n_agents=int(setting.num_agents),
        max_steps=physics_horizon,
        initial_position_bound=float(setting.initial_position_bound),
    )
    cfg.env.action_type = "dynamic_k_nn"
    cfg.env.evaluation_diagnostics = True
    cfg.env.expose_aux_target = False
    cfg.env.expose_global_stats = False
    cfg.env.obs_position_scale = str(obs_position_scale)
    cfg.env.termination_mode = "legacy"
    cfg.env.reward_mode = "legacy"
    cfg.env.initial_position_bound_pool = None
    cfg.env.dt = float(setting.dynamics_dt)
    cfg.env.entropy_p_goal = 0.7 * float(setting.interaction_radius)
    cfg.control.speed = float(setting.speed)
    cfg.control.max_turn_rate = float(setting.max_turn_rate)
    cfg.control.initial_position_bound = float(setting.initial_position_bound)
    cfg.control.r0 = float(setting.r0)
    cfg.control.lam = float(setting.acs_lambda)
    cfg.control.sig = float(setting.acs_sigma)
    return cfg


def _state_metrics(env, proximity_radius: float):
    active = np.flatnonzero(env.state["padding_mask"])
    state = env.state["agent_states"][active]
    position = state[:, :2].astype(np.float64)
    velocity = state[:, 2:4].astype(np.float64)
    speeds = np.linalg.norm(velocity, axis=1)
    headings = velocity / np.maximum(speeds, 1e-12)[:, None]
    distances = _pairwise_dist(position)
    np.fill_diagonal(distances, np.inf)
    return {
        "phi": float(np.linalg.norm(headings.mean(axis=0))),
        "s_ent": float(np.sqrt(position.var(axis=0).sum())),
        "v_ent": float(np.sqrt(velocity.var(axis=0).sum())),
        "n_comp_r0": int(_n_components(distances < float(proximity_radius))),
    }


def _finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _evidence_hashes(payload: Mapping[str, object]):
    """Hash sparse arrays independently of NPZ serialization."""
    result = {}
    for name in EVIDENCE_FIELDS:
        digest = hashlib.sha256()
        value = np.ascontiguousarray(payload[name])
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(canonical_json(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.view(np.uint8))
        result[name] = digest.hexdigest()
    return result


def _summary_from_payload(meta: Mapping[str, object], payload: Mapping[str, object]):
    judgment = judge_episode_c2(
        payload["phi"], payload["s_ent"], payload["n_comp_r0"], payload["reward"])
    horizon = int(meta["horizon"])
    interval = float(meta["policy_interval_seconds"])
    tail = MAIN_C2_V1.stability_window
    t_fire = int(judgment["t_fire"])
    return {
        "run_id": meta["run_id"],
        "suite": meta["suite"],
        "protocol_id": meta["protocol_id"],
        "checkpoint_id": meta["checkpoint_id"],
        "checkpoint_hash": meta["checkpoint_hash"],
        "setting_id": meta["setting_id"],
        "factor": meta["factor"],
        "factor_value": float(meta["factor_value"]),
        "memberships": canonical_json(list(meta["memberships"])),
        "mode": meta["mode"],
        "policy": meta["policy"],
        "seed": int(meta["seed"]),
        "action_seed": int(meta["action_seed"]),
        "horizon": horizon,
        "policy_interval_seconds": interval,
        "substeps_per_policy_interval": int(meta["substeps_per_policy_interval"]),
        "dynamics_dt": float(meta["dynamics_dt"]),
        "physics_steps": int(meta["physics_steps"]),
        "num_agents": int(meta["num_agents"]),
        "speed": float(meta["speed"]),
        "minimum_turn_radius": float(meta["minimum_turn_radius"]),
        "max_turn_rate": float(meta["max_turn_rate"]),
        "interaction_radius": float(meta["interaction_radius"]),
        "acs_gain_multiplier": float(meta["acs_gain_multiplier"]),
        "acs_lambda": float(meta["acs_lambda"]),
        "acs_sigma": float(meta["acs_sigma"]),
        "initial_position_bound": float(meta["initial_position_bound"]),
        **judgment,
        "t_fire_seconds": float(t_fire * interval) if t_fire >= 0 else float("nan"),
        "restricted_time_seconds": float(
            t_fire * interval if t_fire >= 0 else (horizon + 1) * interval),
        "phi_ss": _finite_median(np.asarray(payload["phi"])[-tail:]),
        "sigma_p_ss": _finite_median(np.asarray(payload["s_ent"])[-tail:]),
        "n_comp_end": int(np.asarray(payload["n_comp_r0"])[-1]),
        "mean_selected_neighbors": float(np.mean(payload["mean_selected_neighbors"])),
        "mean_abs_control": float(np.mean(payload["mean_abs_control"])),
        "max_abs_control": float(np.max(payload["max_abs_control"])),
        "control_saturation_fraction": float(
            np.mean(payload["control_saturation_fraction"])),
    }


class SparseAccumulator:
    """Keep only macro-boundary C2 series and compact control diagnostics."""

    def __init__(self, env, setting: SensitivitySetting, horizon: int):
        self.env = env
        self.setting = setting
        self.horizon = int(horizon)
        self.initial_state = env.state["agent_states"].copy()
        self.series = {
            "phi": np.empty(self.horizon + 1, dtype=np.float32),
            "s_ent": np.empty(self.horizon + 1, dtype=np.float32),
            "v_ent": np.empty(self.horizon + 1, dtype=np.float32),
            "n_comp_r0": np.empty(self.horizon + 1, dtype=np.int16),
            "reward": np.full(self.horizon + 1, np.nan, dtype=np.float64),
        }
        self.steps = {
            name: np.empty(self.horizon, dtype=np.float32) for name in STEP_SERIES
        }
        self._record_state(0)

    def _record_state(self, index: int):
        metrics = _state_metrics(self.env, self.setting.c2_proximity_radius)
        for name, value in metrics.items():
            self.series[name][index] = value

    def append_macro(self, index: int, rewards, masks, controls):
        rewards = np.asarray(rewards, dtype=np.float64)
        masks = np.asarray(masks, dtype=bool)
        controls = np.asarray(controls, dtype=np.float64)
        n_agents = int(self.setting.num_agents)
        selected = masks.sum(axis=2) - 1
        absolute = np.abs(controls)
        self.series["reward"][index + 1] = float(rewards.sum())
        self.steps["mean_selected_neighbors"][index] = float(selected.mean())
        self.steps["mean_abs_control"][index] = float(absolute.mean())
        self.steps["max_abs_control"][index] = float(absolute.max())
        self.steps["control_saturation_fraction"][index] = float(np.mean(np.isclose(
            absolute, float(self.setting.max_turn_rate), rtol=1e-6, atol=1e-8)))
        if np.any(selected < 0) or np.any(selected > n_agents - 1):
            raise RuntimeError("realized selection degree is out of range")
        self._record_state(index + 1)

    def payload(self, meta: Mapping[str, object]):
        payload = {
            **self.series,
            **self.steps,
            "initial_agent_states": self.initial_state,
        }
        summary = _summary_from_payload(meta, payload)
        embedded = dict(meta)
        embedded.update({
            "initial_state_sha256": state_hash(self.initial_state),
            "evidence_sha256": _evidence_hashes(payload),
            "summary": json_compatible(summary),
        })
        payload.update({
            "schema_version": np.asarray(SPARSE_SCHEMA_VERSION),
            "meta": np.asarray(canonical_json(embedded)),
            "t_fire": np.asarray(summary["t_fire"], dtype=np.int32),
            "success": np.asarray(summary["success"], dtype=np.int8),
            "J": np.asarray(summary["J"], dtype=np.float64),
        })
        return payload, summary


def episode_path(bundle: Path, setting_id: str, policy: str, seed: int) -> Path:
    return (bundle / "episodes" / SUITE_ID / setting_id / policy
            / f"seed_{int(seed):05d}.npz")


def _scalar_text(value) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _load_sparse(path: Path):
    with np.load(str(path), allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if _scalar_text(payload.get("schema_version")) != SPARSE_SCHEMA_VERSION:
        raise ValueError(f"unsupported sensitivity episode schema: {path}")
    try:
        meta = json.loads(_scalar_text(payload["meta"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sensitivity episode metadata: {path}") from error
    if not isinstance(meta, dict):
        raise ValueError(f"invalid sensitivity episode metadata: {path}")
    return payload, meta


def _same_value(actual, expected) -> bool:
    if actual is None:
        try:
            return bool(np.isnan(float(expected)))
        except (TypeError, ValueError):
            return expected is None
    if isinstance(expected, (float, np.floating)):
        try:
            return bool(np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-10,
                                   equal_nan=True))
        except (TypeError, ValueError):
            return False
    return actual == expected


def validate_sensitivity_episode(path: Path, expected: Mapping[str, object]):
    """Validate the sparse evidence available without claiming physical replay."""
    path = Path(path)
    payload, meta = _load_sparse(path)
    for key, value in expected.items():
        if key not in meta or not _same_value(meta[key], value):
            raise ValueError(f"sensitivity metadata {key} mismatch: {path}")
    horizon = int(meta["horizon"])
    for name in STATE_SERIES:
        if name not in payload or np.asarray(payload[name]).shape != (horizon + 1,):
            raise ValueError(f"invalid {name} shape: {path}")
    for name in STEP_SERIES:
        if name not in payload or np.asarray(payload[name]).shape != (horizon,):
            raise ValueError(f"invalid {name} shape: {path}")
    initial = np.asarray(payload.get("initial_agent_states"))
    if initial.shape != (int(meta["num_agents"]), 5) or not np.isfinite(initial).all():
        raise ValueError(f"invalid initial state: {path}")
    if state_hash(initial) != meta.get("initial_state_sha256"):
        raise ValueError(f"initial-state hash mismatch: {path}")

    phi = np.asarray(payload["phi"], dtype=np.float64)
    spatial = np.asarray(payload["s_ent"], dtype=np.float64)
    velocity = np.asarray(payload["v_ent"], dtype=np.float64)
    components = np.asarray(payload["n_comp_r0"])
    reward = np.asarray(payload["reward"], dtype=np.float64)
    if (not np.isfinite(phi).all() or np.any(phi < -1e-6)
            or np.any(phi > 1.0 + 1e-6)):
        raise ValueError(f"invalid alignment series: {path}")
    if (not np.isfinite(spatial).all() or np.any(spatial < 0)
            or not np.isfinite(velocity).all() or np.any(velocity < 0)):
        raise ValueError(f"invalid entropy series: {path}")
    if (not np.issubdtype(components.dtype, np.integer)
            or np.any(components < 1) or np.any(components > int(meta["num_agents"]))):
        raise ValueError(f"invalid proximity component series: {path}")
    if not np.isnan(reward[0]) or not np.isfinite(reward[1:]).all():
        raise ValueError(f"invalid macro reward series: {path}")
    selected = np.asarray(payload["mean_selected_neighbors"], dtype=np.float64)
    saturation = np.asarray(payload["control_saturation_fraction"], dtype=np.float64)
    controls = np.asarray(payload["mean_abs_control"], dtype=np.float64)
    maximum = np.asarray(payload["max_abs_control"], dtype=np.float64)
    if (not np.isfinite(selected).all() or np.any(selected < 0)
            or np.any(selected > int(meta["num_agents"]) - 1)):
        raise ValueError(f"invalid selected-neighbor series: {path}")
    if (not np.isfinite(saturation).all() or np.any(saturation < 0)
            or np.any(saturation > 1)):
        raise ValueError(f"invalid saturation series: {path}")
    tolerance = float(meta["max_turn_rate"]) * 1e-6 + 1e-8
    if (not np.isfinite(controls).all() or np.any(controls < 0)
            or not np.isfinite(maximum).all()
            or np.any(maximum + tolerance < controls)
            or np.any(maximum > float(meta["max_turn_rate"]) + tolerance)):
        raise ValueError(f"invalid control series: {path}")
    actual_hashes = _evidence_hashes(payload)
    stored_hashes = meta.get("evidence_sha256")
    if not isinstance(stored_hashes, dict):
        raise ValueError(f"sparse evidence hashes are missing: {path}")
    for name, value in actual_hashes.items():
        if stored_hashes.get(name) != value:
            raise ValueError(f"sparse evidence hash mismatch: {name}: {path}")

    summary = _summary_from_payload(meta, payload)
    for key in ("t_fire", "success", "J"):
        if key not in payload or not _same_value(np.asarray(payload[key]).item(), summary[key]):
            raise ValueError(f"stored top-level {key} mismatch: {path}")
    stored = meta.get("summary")
    if not isinstance(stored, dict):
        raise ValueError(f"sensitivity episode has no embedded summary: {path}")
    for key, value in summary.items():
        if key not in stored or not _same_value(stored[key], value):
            raise ValueError(f"stored summary {key} mismatch: {path}")
    return summary, state_hash(initial)


def _expected_meta(job, setting: SensitivitySetting, policy: str, seed: int,
                   config_sha256: str):
    return {
        "run_id": job["run_id"],
        "suite": SUITE_ID,
        "protocol_id": PROTOCOL_ID,
        "fingerprint": job["fingerprint"],
        "checkpoint_id": job["checkpoint_id"],
        "checkpoint_hash": job["checkpoint_hash"],
        "setting_id": setting.setting_id,
        "factor": setting.factor,
        "factor_value": float(setting.factor_value),
        "memberships": list(setting.memberships),
        "mode": setting.mode,
        "policy": policy,
        "seed": int(seed),
        "action_seed": derive_action_seed(seed, setting.num_agents),
        "horizon": int(job["horizon"]),
        "policy_interval_seconds": float(job["policy_interval_seconds"]),
        "substeps_per_policy_interval": int(setting.substeps_per_policy_interval),
        "dynamics_dt": float(setting.dynamics_dt),
        "physics_steps": int(job["horizon"]) * int(setting.substeps_per_policy_interval),
        "num_agents": int(setting.num_agents),
        "speed": float(setting.speed),
        "minimum_turn_radius": float(setting.minimum_turn_radius),
        "max_turn_rate": float(setting.max_turn_rate),
        "interaction_radius": float(setting.interaction_radius),
        "acs_gain_multiplier": float(setting.acs_gain_multiplier),
        "acs_lambda": float(setting.acs_lambda),
        "acs_sigma": float(setting.acs_sigma),
        "initial_position_bound": float(setting.initial_position_bound),
        "backend": job["device"],
        "batch_size": int(job["batch_size"]),
        "config_sha256": config_sha256,
    }


def _run_job(job: Dict[str, object]) -> List[Dict[str, object]]:
    setting = job["setting"]
    if not isinstance(setting, SensitivitySetting):
        setting = SensitivitySetting(**setting)
    bundle = Path(job["bundle"])
    policy = str(job["policy"])
    horizon = int(job["horizon"])
    cfg = build_sensitivity_env_config(
        setting, horizon, str(job["obs_position_scale"]))
    config_hash = run_fingerprint(cfg.dict())
    if config_hash != job["config_sha256"]:
        raise ValueError("effective sensitivity config differs from manifest")

    cached, seed_entries = [], []
    for seed_value in job["seeds"]:
        seed = int(seed_value)
        expected = _expected_meta(job, setting, policy, seed, config_hash)
        path = episode_path(bundle, setting.setting_id, policy, seed)
        cached_summary = None
        if path.is_file():
            try:
                cached_summary, _ = validate_sensitivity_episode(path, expected)
                cached_summary["path"] = path.relative_to(bundle).as_posix()
                cached.append(cached_summary)
            except Exception:
                if not job["repair_invalid"]:
                    raise
                quarantine(path, bundle)
                cached_summary = None
        seed_entries.append((seed, path, expected, cached_summary))
    if all(item[3] is not None for item in seed_entries):
        return sorted(cached, key=lambda row: int(row["seed"]))

    import torch
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input

    torch.set_num_threads(1)
    inference = None
    completed = list(cached)
    batch_size = int(job["batch_size"])
    for start in range(0, len(seed_entries), batch_size):
        batch_entries = seed_entries[start:start + batch_size]
        if all(item[3] is not None for item in batch_entries):
            continue
        # Preserve configured seed-batch boundaries across resume/repair.  A
        # partial batch is replayed at its original shape, but valid artifacts
        # are left untouched.
        batch = [item[:3] for item in batch_entries]
        envs, observations, accumulators = [], [], []
        for seed, _, _ in batch:
            env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
            env.seed(seed)
            observation = env.reset()
            envs.append(env)
            observations.append(observation)
            accumulators.append(SparseAccumulator(env, setting, horizon))
        if policy != "pure_acs" and inference is None:
            inference = DynamicKNNInferencePolicy(
                Path(job["checkpoint"]), envs[0], device=str(job["device"]))
        generators = None
        if policy == "learned_stochastic":
            generators = [
                inference.make_generator(int(item[2]["action_seed"])) for item in batch
            ]

        physics_index = 0
        for macro_index in range(horizon):
            if policy == "pure_acs":
                held_actions = None
            else:
                held_actions = inference.actions(
                    observations,
                    mode=policy.removeprefix("learned_"),
                    generators=generators,
                )
            rewards = [[] for _ in batch]
            masks = [[] for _ in batch]
            controls = [[] for _ in batch]
            for _ in range(int(setting.substeps_per_policy_interval)):
                actions = (
                    np.stack([pure_acs_pointer(env) for env in envs])
                    if policy == "pure_acs" else held_actions
                )
                next_observations = []
                for index, env in enumerate(envs):
                    next_obs, reward, done, info = env.step(actions[index])
                    final_step = physics_index == int(job["physics_horizon"]) - 1
                    if bool(done) != final_step:
                        raise RuntimeError(
                            f"unexpected done={done} at physics step {physics_index + 1}")
                    rewards[index].append(float(reward))
                    masks[index].append(np.asarray(info["binary_action"], dtype=bool))
                    controls[index].append(
                        np.asarray(info["control_inputs"], dtype=np.float64))
                    next_observations.append(next_obs)
                observations = next_observations
                physics_index += 1
            for index, accumulator in enumerate(accumulators):
                accumulator.append_macro(
                    macro_index, rewards[index], masks[index], controls[index])

        for accumulator, (_, path, expected, cached_summary) in zip(
                accumulators, batch_entries):
            if cached_summary is not None:
                continue
            meta = dict(expected)
            meta.update({
                "criterion_of_record": False,
                "official_c2_complete_horizon": horizon >= 6000,
                "created_at_utc": utc_now(),
            })
            payload, summary = accumulator.payload(meta)
            atomic_save_npz(path, payload)
            summary["path"] = path.relative_to(bundle).as_posix()
            completed.append(summary)
    return sorted(completed, key=lambda row: int(row["seed"]))


def build_manifest(run_id: str, checkpoint: os.PathLike,
                   config: ResolvedSensitivityConfig, device: str,
                   batch_size: int):
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint_info = checkpoint_identity(checkpoint)
    config_hashes = {
        setting.setting_id: run_fingerprint(build_sensitivity_env_config(
            setting, config.horizon,
            str(checkpoint_info["obs_position_scale"])).dict())
        for setting in config.settings
    }
    git = git_snapshot(Path(__file__).resolve().parents[2])
    runtime = runtime_snapshot()
    config_input = config.fingerprint_input()
    spec = {
        "suite": SUITE_ID,
        "protocol_id": PROTOCOL_ID,
        "schema_version": SPARSE_SCHEMA_VERSION,
        "config": config_input,
        "backend": device,
        "batch_size": int(batch_size),
        "checkpoint_package_sha256": checkpoint_info["package_sha256"],
        "checkpoint_state_sha256": checkpoint_info["state_file_sha256"],
        "checkpoint_config_sha256": checkpoint_info["config_file_sha256"],
        "effective_config_sha256_by_setting": config_hashes,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_dirty_diff_sha256": git["dirty_diff_sha256"],
        "runtime": runtime,
    }
    fingerprint = run_fingerprint(spec)
    policy_count, seed_count = len(config.policies), len(config.seeds)
    return {
        "schema_version": "sensitivity-bundle-1.0",
        "run_id": run_id,
        "suite": SUITE_ID,
        "protocol": {
            "id": PROTOCOL_ID,
            "phi_goal_strict_gt": MAIN_C2_V1.phi_goal,
            "alignment_window": MAIN_C2_V1.alignment_window,
            "stability_window": MAIN_C2_V1.stability_window,
            "spatial_band_strict_lt": MAIN_C2_V1.spatial_band_epsilon,
            "sample_clock": "10 Hz policy boundaries",
            "criterion_of_record": False,
        },
        "fingerprint": fingerprint,
        "config_source": str(config.source_path),
        "spec": spec,
        "checkpoint": checkpoint_info,
        "workload": {
            "settings": len(config.settings),
            "episodes": len(config.settings) * policy_count * seed_count,
            "policy_steps": len(config.settings) * policy_count * seed_count
            * config.horizon,
            "physics_steps": sum(
                setting.substeps_per_policy_interval for setting in config.settings
            ) * policy_count * seed_count * config.horizon,
        },
        "runtime": runtime,
        "git": git,
        "status": "initialized",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }


def run_sensitivity(checkpoint: os.PathLike, bundle: os.PathLike, run_id: str,
                    config: ResolvedSensitivityConfig, *, device: str = "cpu",
                    batch_size: int = 1, workers: int = 8,
                    repair_invalid: bool = False, dry_run: bool = False):
    """Run or resume a resolved sensitivity configuration."""
    device = str(device).strip().lower()
    if not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", device):
        raise ValueError("device must be cpu or cuda[:index]")
    if int(batch_size) <= 0 or int(workers) <= 0:
        raise ValueError("batch size and workers must be positive")
    if not set(config.policies).issubset(POLICY_NAMES):
        raise ValueError("unsupported sensitivity policy")
    bundle = Path(bundle).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest = build_manifest(run_id, checkpoint, config, str(device), int(batch_size))
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_id") != run_id:
            raise ValueError("existing sensitivity bundle belongs to another run ID")
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                "existing sensitivity bundle fingerprint differs; use a new run ID")
        manifest = previous
    if dry_run:
        return manifest

    archive = archive_checkpoint(
        bundle, checkpoint, manifest["checkpoint"], repair_invalid=repair_invalid)
    manifest["checkpoint"]["archive"] = archive
    evaluation_checkpoint = bundle / archive / "checkpoint"
    manifest["status"] = "running"
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)

    jobs = []
    for setting in config.settings:
        for policy in config.policies:
            jobs.append({
                "bundle": str(bundle),
                "run_id": run_id,
                "fingerprint": manifest["fingerprint"],
                "checkpoint": str(evaluation_checkpoint),
                "checkpoint_id": manifest["checkpoint"]["id"],
                "checkpoint_hash": manifest["checkpoint"]["package_sha256"],
                "obs_position_scale": manifest["checkpoint"]["obs_position_scale"],
                "config_sha256": manifest["spec"][
                    "effective_config_sha256_by_setting"][setting.setting_id],
                "setting": setting,
                "policy": policy,
                "seeds": list(config.seeds),
                "horizon": config.horizon,
                "policy_interval_seconds": config.policy_interval_seconds,
                "physics_horizon": config.horizon
                * setting.substeps_per_policy_interval,
                "device": str(device),
                "batch_size": int(batch_size),
                "repair_invalid": bool(repair_invalid),
            })
    if device == "cpu" and workers > 1:
        context = mp.get_context("spawn")
        with context.Pool(min(int(workers), len(jobs))) as pool:
            groups = pool.map(_run_job, jobs)
    else:
        groups = [_run_job(job) for job in jobs]
    rows = [row for group in groups for row in group]

    from eval.sensitivity.analysis import write_summary_tables
    write_summary_tables(rows, bundle)
    manifest["status"] = "completed"
    manifest["episode_count"] = len(rows)
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest


def _setting_from_manifest(value):
    data = dict(value)
    data["memberships"] = tuple(data["memberships"])
    return SensitivitySetting(**data)


def validate_sensitivity_bundle(bundle: os.PathLike, *,
                                write_report: bool = False,
                                rebuild_summaries: bool = False):
    """Validate sparse artifacts and summaries; no full-dynamics replay is claimed."""
    bundle = Path(bundle).expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("suite") != SUITE_ID or manifest.get("status") != "completed":
        raise ValueError("not a completed sensitivity bundle")
    if run_fingerprint(manifest["spec"]) != manifest.get("fingerprint"):
        raise ValueError("sensitivity manifest fingerprint mismatch")
    if (manifest["spec"].get("checkpoint_package_sha256")
            != manifest["checkpoint"].get("package_sha256")):
        raise ValueError("sensitivity manifest checkpoint identity mismatch")
    archived = checkpoint_identity(bundle / manifest["checkpoint"]["archive"])
    if archived["package_sha256"] != manifest["checkpoint"]["package_sha256"]:
        raise ValueError("archived checkpoint package hash mismatch")

    spec = manifest["spec"]
    resolved = spec["config"]
    rows, initial_hashes = [], {}
    for raw_setting in resolved["settings"]:
        setting = _setting_from_manifest(raw_setting)
        for policy in resolved["policies"]:
            for seed_value in resolved["seeds"]:
                seed = int(seed_value)
                job = {
                    "run_id": manifest["run_id"],
                    "fingerprint": manifest["fingerprint"],
                    "checkpoint_id": manifest["checkpoint"]["id"],
                    "checkpoint_hash": manifest["checkpoint"]["package_sha256"],
                    "horizon": int(resolved["macro_steps"]),
                    "policy_interval_seconds": float(
                        resolved["policy_interval_seconds"]),
                    "device": spec["backend"],
                    "batch_size": int(spec["batch_size"]),
                }
                expected = _expected_meta(
                    job, setting, policy, seed,
                    spec["effective_config_sha256_by_setting"][setting.setting_id])
                path = episode_path(bundle, setting.setting_id, policy, seed)
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                summary, initial_hash = validate_sensitivity_episode(path, expected)
                key = (setting.setting_id, seed)
                if key in initial_hashes and initial_hashes[key] != initial_hash:
                    raise ValueError(f"initial state mismatch across policies for {key}")
                initial_hashes[key] = initial_hash
                summary["path"] = path.relative_to(bundle).as_posix()
                rows.append(summary)

    if int(manifest.get("episode_count", -1)) != len(rows):
        raise ValueError("sensitivity manifest episode count mismatch")

    from eval.sensitivity.analysis import verify_summary_tables, write_summary_tables
    summary_status = "rebuilt" if rebuild_summaries else "verified"
    if rebuild_summaries:
        write_summary_tables(rows, bundle)
    else:
        verify_summary_tables(rows, bundle)
    result = {
        "schema_version": "sensitivity-validation-1.0",
        "run_id": manifest["run_id"],
        "fingerprint": manifest["fingerprint"],
        "episodes": len(rows),
        "initial_state_groups": len(initial_hashes),
        "evidence_level": "sparse-series",
        "summaries": summary_status,
        "report_written": bool(write_report or rebuild_summaries),
        "status": "pass",
        "validated_at_utc": utc_now(),
    }
    if write_report or rebuild_summaries:
        atomic_write_json(bundle / "validation.json", result)
    return result


__all__ = [
    "PROTOCOL_ID",
    "SPARSE_SCHEMA_VERSION",
    "SUITE_ID",
    "SparseAccumulator",
    "build_manifest",
    "build_sensitivity_env_config",
    "episode_path",
    "run_sensitivity",
    "validate_sensitivity_bundle",
    "validate_sensitivity_episode",
]
