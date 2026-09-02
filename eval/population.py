"""Full-trace Dynamic-k population evaluation under the main C2 protocol."""

from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from eval.artifacts import (
    FULL_SCHEMA_VERSION, SCALAR_SERIES, atomic_open, atomic_save_npz,
    atomic_write_json, canonical_json, json_compatible, load_episode,
    make_staging_directory, run_fingerprint, sha256_file, sha256_tree,
    validate_full_episode,
)
from eval.common import SERIES_NAMES, _n_components, _pairwise_dist, build_config
from eval.policies import DynamicKNNInferencePolicy, resolve_dynamic_checkpoint
from eval.provenance import git_snapshot, runtime_snapshot
from eval.protocol import MAIN_C2_V1, judge_episode_c2
from eval.stats import cvar10, mcnemar, paired_dj, wilson


PROTOCOL_ID = "main_c2_v1"
SUITE_ID = "population_v1"
DEFAULT_POLICIES = ("deterministic", "stochastic", "pure_acs")
DEFAULT_NUM_AGENTS = (10, 20, 40)


def parse_int_range(value: str) -> List[int]:
    """Parse ``A-B`` or a comma-separated integer list."""
    value = str(value).strip()
    if "," in value:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif "-" in value:
        left, right = (int(item) for item in value.split("-", 1))
        if right < left:
            raise ValueError("range end precedes range start")
        result = list(range(left, right + 1))
    else:
        result = [int(value)]
    if not result or len(result) != len(set(result)):
        raise ValueError("seed list must be non-empty and unique")
    return result


def derive_action_seed(seed: int, num_agents: int) -> int:
    value = f"distance-pointer-action:{int(seed)}:{int(num_agents)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") & ((1 << 63) - 1)


def pure_acs_pointer(env) -> np.ndarray:
    active = np.flatnonzero(env.state["padding_mask"])
    distances = env.rel_state["rel_agent_dists"]
    pointer = np.zeros(env.num_agents_max, dtype=np.int64)
    for ego in active:
        candidates = active[active != ego]
        if not candidates.size:
            pointer[ego] = ego
        else:
            pointer[ego] = candidates[int(np.argmax(distances[ego, candidates]))]
    return pointer


def _state_hash(state: np.ndarray) -> str:
    value = np.ascontiguousarray(state)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_identity(checkpoint_path: Path) -> Dict[str, object]:
    source = resolve_dynamic_checkpoint(checkpoint_path)
    checkpoint = Path(source["checkpoint"])
    config_file = Path(source["config_file"])
    if source["kind"] == "rllib":
        state_file = Path(source["policy_state"])
        checkpoint_id = checkpoint.name
    else:
        state_file = Path(source["state_dict"])
        checkpoint_id = Path(source["archive_root"]).name
    tree_hash = sha256_tree(checkpoint)
    state_hash = sha256_file(state_file)
    config_hash = sha256_file(config_file)
    model_config_hash = run_fingerprint(source["model_config"])
    package_hash = run_fingerprint({
        "checkpoint_tree_sha256": tree_hash,
        "state_file_sha256": state_hash,
        "config_file_sha256": config_hash,
        "model_config_sha256": model_config_hash,
        "obs_position_scale": source["obs_position_scale"],
        "source_kind": source["kind"],
    })
    return {
        "id": checkpoint_id,
        "source": str(Path(checkpoint_path).expanduser().resolve()),
        "source_kind": source["kind"],
        "tree_sha256": tree_hash,
        "state_file": str(state_file),
        "state_file_sha256": state_hash,
        "config_file": str(config_file),
        "config_file_sha256": config_hash,
        "model_config_sha256": model_config_hash,
        "package_sha256": package_hash,
        "obs_position_scale": source["obs_position_scale"],
    }


def _archive_checkpoint(bundle: Path, checkpoint_path: Path,
                        checkpoint_info: Dict[str, object],
                        repair_invalid: bool = False) -> str:
    """Atomically archive and verify the complete inference load package."""
    source = resolve_dynamic_checkpoint(checkpoint_path)
    archive_root = bundle / "checkpoint"
    if archive_root.exists():
        try:
            archived = _checkpoint_identity(archive_root)
            if archived["package_sha256"] != checkpoint_info["package_sha256"]:
                raise ValueError("archived finalist package hash differs from source")
            return archive_root.relative_to(bundle).as_posix()
        except Exception:
            if not repair_invalid:
                raise
            _quarantine(archive_root, bundle)

    bundle.mkdir(parents=True, exist_ok=True)
    staging = make_staging_directory(bundle, ".tmp-checkpoint.")
    try:
        archived_checkpoint = staging / "checkpoint"
        source_checkpoint = Path(source["checkpoint"])
        if source_checkpoint.is_dir():
            shutil.copytree(str(source_checkpoint), str(archived_checkpoint))
        else:
            shutil.copy2(str(source_checkpoint), str(archived_checkpoint))
        if source["kind"] == "rllib":
            shutil.copy2(str(source["config_file"]), str(staging / "params.json"))
        else:
            source_root = Path(source["archive_root"])
            for name in ("metadata.json", "model_state_dict.pt", "checksums.json"):
                item = source_root / name
                if item.is_file():
                    shutil.copy2(str(item), str(staging / name))
        archived = _checkpoint_identity(staging)
        if archived["package_sha256"] != checkpoint_info["package_sha256"]:
            raise RuntimeError("finalist checkpoint package failed checksum validation")
        os.replace(staging, archive_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return archive_root.relative_to(bundle).as_posix()


def _build_config(n_agents: int, horizon: int, bound: float,
                  obs_position_scale: str):
    cfg = build_config(n_agents=n_agents, max_steps=horizon,
                       initial_position_bound=bound)
    cfg.env.action_type = "dynamic_k_nn"
    cfg.env.evaluation_diagnostics = True
    cfg.env.expose_aux_target = False
    cfg.env.expose_global_stats = False
    cfg.env.obs_position_scale = obs_position_scale
    cfg.env.termination_mode = "legacy"  # fixed horizon; judge offline
    cfg.env.reward_mode = "legacy"       # is_training=False -> raw control cost
    cfg.env.initial_position_bound_pool = None
    return cfg


class FullAccumulator:
    def __init__(self, env, horizon: int):
        self.env = env
        self.horizon = int(horizon)
        self.n_agents = int(env.num_agents_max)
        self.states = np.empty((horizon + 1, self.n_agents, 5), dtype=np.float64)
        self.states[0] = env.state["agent_states"].copy()
        self.pointer = np.empty((horizon, self.n_agents), dtype=np.int16)
        self.binary = np.empty((horizon, self.n_agents, self.n_agents), dtype=np.bool_)
        self.controls = np.empty((horizon, self.n_agents), dtype=np.float64)
        self.deg_agents = np.empty((horizon, self.n_agents), dtype=np.int16)
        self.rec = {
            name: np.full(horizon + 1, np.nan, dtype=np.float32)
            for name in SERIES_NAMES
        }
        self.previous_edges = None
        self._log_state(0, None, None, np.nan)

    def _log_state(self, t: int, action, info, reward: float):
        state = self.env.state["agent_states"]
        active = np.flatnonzero(self.env.state["padding_mask"])
        pos = state[active, :2].astype(np.float64)
        vel = state[active, 2:4].astype(np.float64)
        self.rec["s_ent"][t] = np.sqrt(pos.var(axis=0).sum())
        self.rec["v_ent"][t] = np.sqrt(vel.var(axis=0).sum())
        if info is not None:
            self.rec["s_ent_env"][t] = info.get("spatial_entropy", np.nan)
            self.rec["v_ent_env"][t] = info.get("velocity_entropy", np.nan)
            self.rec["reward"][t] = reward
        speed = np.linalg.norm(vel, axis=1)
        unit = vel / np.maximum(speed, 1e-12)[:, None]
        self.rec["phi"][t] = np.linalg.norm(unit.mean(axis=0))
        distances = _pairwise_dist(pos)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        self.rec["nnd_mean"][t] = nearest.mean()
        self.rec["nnd_max"][t] = nearest.max()
        self.rec["min_pair"][t] = nearest.min()
        finite_dist = distances.copy()
        finite_dist[np.isinf(finite_dist)] = 0.0
        self.rec["diam"][t] = finite_dist.max()
        self.rec["radius"][t] = np.linalg.norm(
            pos - pos.mean(axis=0), axis=1
        ).max()
        self.rec["n_comp_r0"][t] = _n_components(
            distances < float(self.env.config.control.r0)
        )
        if action is not None:
            selected = np.asarray(action)[np.ix_(active, active)].astype(bool)
            offdiag = selected & ~np.eye(len(active), dtype=bool)
            self.rec["deg_mean"][t] = offdiag.sum() / len(active)
            self.rec["n_comp_sel"][t] = _n_components(offdiag)
            if self.previous_edges is not None:
                intersection = (offdiag & self.previous_edges).sum()
                union = (offdiag | self.previous_edges).sum()
                self.rec["churn"][t] = 1.0 - (
                    intersection / union if union else 1.0
                )
            self.previous_edges = offdiag

    def append(self, step_index: int, pointer: np.ndarray, reward: float,
               info: Dict[str, object]):
        mask = info.get("binary_action")
        controls = info.get("control_inputs")
        if mask is None or controls is None:
            raise RuntimeError("evaluation_diagnostics did not return action/control")
        self.pointer[step_index] = pointer
        self.binary[step_index] = np.asarray(mask, dtype=np.bool_)
        self.controls[step_index] = np.asarray(controls, dtype=np.float64)
        self.states[step_index + 1] = self.env.state["agent_states"].copy()
        self.deg_agents[step_index] = (
            self.binary[step_index] & ~np.eye(self.n_agents, dtype=bool)
        ).sum(axis=1)
        self._log_state(step_index + 1, mask, info, reward)

    def payload(self, meta: Dict[str, object]):
        judgment = judge_episode_c2(
            self.rec["phi"], self.rec["s_ent"], self.rec["n_comp_r0"],
            self.rec["reward"],
        )
        summary = {
            "run_id": meta["run_id"], "suite": meta["suite"],
            "protocol_id": meta["protocol_id"], "checkpoint_id": meta["checkpoint_id"],
            "checkpoint_hash": meta["checkpoint_hash"],
            "num_agents": self.n_agents, "bound": meta["bound"],
            "seed": meta["seed"], "action_seed": meta["action_seed"],
            "policy": meta["policy"], "horizon": self.horizon,
            **judgment,
            "phi_ss": float(np.nanmedian(self.rec["phi"][-300:])),
            "sigma_p_ss": float(np.nanmedian(self.rec["s_ent"][-300:])),
            "min_pair": float(np.nanmin(self.rec["min_pair"])),
            "deg_ss": float(np.nanmedian(self.rec["deg_mean"][-300:])),
            "churn_ss": float(np.nanmedian(self.rec["churn"][-300:])),
            "n_comp_end": float(self.rec["n_comp_r0"][-1]),
        }
        meta = dict(meta)
        meta.update({
            "summary": json_compatible(summary),
            "initial_state_sha256": _state_hash(self.states[0]),
        })
        payload = {
            "schema_version": np.asarray(FULL_SCHEMA_VERSION),
            "meta": np.asarray(canonical_json(meta)),
            "agent_states": self.states,
            "pointer_actions": self.pointer,
            "binary_actions": self.binary,
            "control_inputs": self.controls,
            "deg_agents": self.deg_agents,
            "pos_snaps": self.states[::10, :, :2].astype(np.float32),
            "snap_ts": np.arange(0, self.horizon + 1, 10, dtype=np.int32),
            "t_fire": np.asarray(judgment["t_fire"], dtype=np.int32),
            "success": np.asarray(judgment["success"], dtype=np.int8),
            "J": np.asarray(judgment["J"], dtype=np.float64),
        }
        if payload["snap_ts"][-1] != self.horizon:
            payload["pos_snaps"] = np.concatenate([
                payload["pos_snaps"], self.states[-1:, :, :2].astype(np.float32)
            ])
            payload["snap_ts"] = np.concatenate([
                payload["snap_ts"], np.asarray([self.horizon], dtype=np.int32)
            ])
        payload.update(self.rec)
        return payload, summary


def _episode_path(bundle: Path, checkpoint_id: str, n_agents: int,
                  policy: str, seed: int) -> Path:
    return (bundle / "episodes" / SUITE_ID / checkpoint_id / f"N{n_agents}"
            / policy / f"seed_{seed:05d}.npz")


def _validate_cached(path: Path, expected_meta: Dict[str, object]):
    view = load_episode(path)
    return validate_population_episode(view, expected_meta, deep=True)


def _quarantine(path: Path, bundle: Path):
    relative = path.relative_to(bundle)
    target = bundle / "quarantine" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(target.stem + "-" + datetime.now().strftime("%H%M%S%f") + target.suffix)
    shutil.move(str(path), str(target))
    return target


def _run_group(job: Dict[str, object]) -> List[Dict[str, object]]:
    # Imports remain process-local so CPU groups can run independently.
    import torch
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input

    torch.set_num_threads(1)
    bundle = Path(job["bundle"])
    n_agents = int(job["n_agents"])
    horizon = int(job["horizon"])
    bound = float(job["bound"])
    policy_name = str(job["policy"])
    seeds = [int(seed) for seed in job["seeds"]]
    device = str(job["device"])
    batch_size = int(job["batch_size"])
    checkpoint = Path(job["checkpoint"])
    checkpoint_id = str(job["checkpoint_id"])
    cfg = _build_config(n_agents, horizon, bound, str(job["obs_position_scale"]))
    actual_config_hash = run_fingerprint(cfg.dict())
    if actual_config_hash != job["config_sha256"]:
        raise ValueError("effective evaluation config differs from bundle manifest")
    physical = {
        "dt": float(cfg.env.dt), "speed": float(cfg.control.speed),
        "r0": float(cfg.control.r0), "rho": float(cfg.control.rho),
    }
    for name, value in physical.items():
        if value != float(job[name]):
            raise ValueError(f"effective {name} differs from bundle manifest")

    inference = None
    summaries = []
    pending = []
    for seed in seeds:
        action_seed = derive_action_seed(seed, n_agents)
        expected = {
            "run_id": job["run_id"], "suite": SUITE_ID,
            "protocol_id": PROTOCOL_ID, "fingerprint": job["fingerprint"],
            "checkpoint_id": checkpoint_id, "checkpoint_hash": job["checkpoint_hash"],
            "num_agents": n_agents, "bound": bound, "seed": seed,
            "action_seed": action_seed, "policy": policy_name,
            "horizon": horizon, "backend": device, "batch_size": batch_size,
            "config_sha256": actual_config_hash, **physical,
        }
        path = _episode_path(bundle, checkpoint_id, n_agents, policy_name, seed)
        if path.is_file():
            try:
                summary = _validate_cached(path, expected)
                summary["path"] = path.relative_to(bundle).as_posix()
                summaries.append(summary)
                continue
            except Exception:
                if not job["repair_invalid"]:
                    raise
                _quarantine(path, bundle)
        pending.append((seed, action_seed, path, expected))

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        envs, observations, accumulators, generators = [], [], [], []
        for seed, action_seed, _, _ in batch:
            env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=seed))
            env.seed(seed)
            observation = env.reset()
            envs.append(env)
            observations.append(observation)
            accumulators.append(FullAccumulator(env, horizon))
            if policy_name == "stochastic":
                # Lazily instantiate below after the first env defines spaces.
                generators.append(None)
        if policy_name != "pure_acs" and inference is None:
            inference = DynamicKNNInferencePolicy(checkpoint, envs[0], device=device)
        if policy_name == "stochastic":
            generators = [inference.make_generator(item[1]) for item in batch]

        for step_index in range(horizon):
            if policy_name == "pure_acs":
                actions = np.stack([pure_acs_pointer(env) for env in envs])
            else:
                actions = inference.actions(
                    observations, mode=policy_name,
                    generators=generators if policy_name == "stochastic" else None,
                )
            next_observations = []
            for index, env in enumerate(envs):
                next_obs, reward, done, info = env.step(actions[index])
                expected_done = step_index == horizon - 1
                if bool(done) != expected_done:
                    raise RuntimeError(
                        f"unexpected done={done} at step {step_index + 1}/{horizon}"
                    )
                accumulators[index].append(step_index, actions[index], reward, info)
                next_observations.append(next_obs)
            observations = next_observations

        for index, (seed, action_seed, path, expected) in enumerate(batch):
            meta = dict(expected)
            meta.update({
                "k1": float(cfg.control.k1), "k2": float(cfg.control.k2),
                "lam": float(cfg.control.lam), "sig": float(cfg.control.sig),
                "criterion_of_record": False,
                "official_c2_complete_horizon": horizon >= 6000,
                "created_at_utc": _utc_now(),
            })
            payload, summary = accumulators[index].payload(meta)
            atomic_save_npz(path, payload)
            summary["path"] = path.relative_to(bundle).as_posix()
            summaries.append(summary)
    return sorted(summaries, key=lambda row: int(row["seed"]))


def _write_csv_atomic(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(list(rows)).to_csv(stream, index=False)


def summarize(rows: Sequence[Dict[str, object]], bundle: Path):
    import pandas as pd
    frame = pd.DataFrame(rows).sort_values(["num_agents", "policy", "seed"])
    _write_csv_atomic(bundle / "summaries" / "episodes.csv", frame.to_dict("records"))
    aggregate = []
    for (n_agents, policy), group in frame.groupby(["num_agents", "policy"]):
        success = group[group.success == 1]
        failures = int((group.success == 0).sum())
        lo, hi = wilson(failures, len(group))
        aggregate.append({
            "num_agents": int(n_agents), "policy": policy, "n": len(group),
            "failures": failures, "success_rate": 1.0 - failures / len(group),
            "failure_wilson_lo": lo, "failure_wilson_hi": hi,
            "t_fire_median": float(success.t_fire.median()) if len(success) else np.nan,
            "J_median": float(success.J.median()) if len(success) else np.nan,
            "J_cvar10": cvar10(success.J.astype(float)) if len(success) else np.nan,
        })
    _write_csv_atomic(bundle / "summaries" / "aggregate.csv", aggregate)

    paired = []
    paired_stats = []
    for n_agents, n_frame in frame.groupby("num_agents"):
        reference = n_frame[n_frame.policy == "pure_acs"].set_index("seed")
        for policy in ("deterministic", "stochastic"):
            candidate = n_frame[n_frame.policy == policy].set_index("seed")
            common = candidate.index.intersection(reference.index)
            for seed in common:
                left, right = candidate.loc[seed], reference.loc[seed]
                paired.append({
                    "num_agents": int(n_agents), "policy": policy,
                    "reference": "pure_acs", "seed": int(seed),
                    "success": int(left.success), "reference_success": int(right.success),
                    "delta_t_fire": (float(left.t_fire - right.t_fire)
                                     if left.success and right.success else np.nan),
                    "delta_J": (float(left.J - right.J)
                                if left.success and right.success else np.nan),
                })
            if len(common):
                learned = candidate.loc[common].sort_index()
                baseline = reference.loc[common].sort_index()
                b_count, c_count, mcnemar_p = mcnemar(learned, baseline)
                dj = paired_dj(learned, baseline)
                both = (learned.success == 1) & (baseline.success == 1)
                delta_t = (learned.t_fire[both] - baseline.t_fire[both]).astype(float)
                paired_stats.append({
                    "num_agents": int(n_agents), "policy": policy,
                    "reference": "pure_acs", "n_pairs": len(common),
                    "learned_fail_reference_success": b_count,
                    "learned_success_reference_fail": c_count,
                    "mcnemar_exact_p": mcnemar_p,
                    "co_success_pairs": int(both.sum()),
                    "delta_t_fire_median": (
                        float(delta_t.median()) if len(delta_t) else np.nan),
                    "delta_J_median": dj["med"],
                    "delta_J_wilcoxon_p": dj["wilcox_p"],
                    "delta_J_worse_count": dj["worse"],
                    "delta_J_sign_p": dj["sign_p"],
                })
    _write_csv_atomic(bundle / "summaries" / "paired_vs_acs.csv", paired)
    _write_csv_atomic(
        bundle / "summaries" / "paired_stats_vs_acs.csv", paired_stats)
    return aggregate, paired, paired_stats


def build_manifest(bundle: Path, run_id: str, checkpoint: Path,
                   num_agents: Sequence[int], seeds: Sequence[int], horizon: int,
                   bound: float, policies: Sequence[str], device: str,
                   batch_size: int) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_info = _checkpoint_identity(checkpoint)
    git = git_snapshot(repo_root)
    runtime = runtime_snapshot()
    effective_configs = {
        str(int(n_agents)): _build_config(
            int(n_agents), horizon, bound,
            str(checkpoint_info["obs_position_scale"]),
        ).dict()
        for n_agents in num_agents
    }
    config_hashes = {
        key: run_fingerprint(value) for key, value in effective_configs.items()
    }
    representative = _build_config(
        int(num_agents[0]), horizon, bound,
        str(checkpoint_info["obs_position_scale"]),
    )
    spec = {
        "suite": SUITE_ID, "protocol_id": PROTOCOL_ID,
        "num_agents": [int(item) for item in num_agents], "bound": float(bound),
        "seeds": [int(item) for item in seeds], "horizon": int(horizon),
        "policies": list(policies), "backend": device, "batch_size": int(batch_size),
        "checkpoint_package_sha256": checkpoint_info["package_sha256"],
        "checkpoint_state_sha256": checkpoint_info["state_file_sha256"],
        "checkpoint_config_sha256": checkpoint_info["config_file_sha256"],
        "checkpoint_tree_sha256": checkpoint_info["tree_sha256"],
        "effective_config_sha256_by_num_agents": config_hashes,
        "effective_config_by_num_agents": effective_configs,
        "schema_version": FULL_SCHEMA_VERSION,
        "dt": float(representative.env.dt),
        "speed": float(representative.control.speed),
        "r0": float(representative.control.r0),
        "rho": float(representative.control.rho),
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "git_dirty_diff_sha256": git["dirty_diff_sha256"],
        "runtime": runtime,
    }
    fingerprint = run_fingerprint(spec)
    return {
        "schema_version": "evaluation-bundle-2.0", "run_id": run_id,
        "suite": SUITE_ID, "protocol": {
            "id": PROTOCOL_ID, "phi_goal_strict_gt": MAIN_C2_V1.phi_goal,
            "alignment_window": MAIN_C2_V1.alignment_window,
            "stability_window": MAIN_C2_V1.stability_window,
            "spatial_band_strict_lt": MAIN_C2_V1.spatial_band_epsilon,
            "proximity_edge": "distance < r0", "criterion_of_record": False,
        },
        "fingerprint": fingerprint, "spec": spec, "checkpoint": checkpoint_info,
        "runtime": runtime,
        "git": git, "status": "initialized",
        "created_at_utc": _utc_now(), "updated_at_utc": _utc_now(),
    }


def run_population(checkpoint: os.PathLike, bundle: os.PathLike, run_id: str,
                   num_agents: Sequence[int] = DEFAULT_NUM_AGENTS,
                   seeds: Sequence[int] = tuple(range(50)), horizon: int = 6000,
                   bound: float = 250.0, policies: Sequence[str] = DEFAULT_POLICIES,
                   device: str = "cpu", batch_size: int = 1, workers: int = 8,
                   repair_invalid: bool = False, dry_run: bool = False):
    bundle = Path(bundle).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or cuda[:index]")
    if device.startswith("cuda") and batch_size <= 0:
        raise ValueError("CUDA population evaluation requires --batch-size > 0")
    if device == "cpu":
        batch_size = 1  # criterion-compatible single-episode inference
    invalid_policies = set(policies).difference(DEFAULT_POLICIES)
    if invalid_policies:
        raise ValueError(f"unsupported population policies: {sorted(invalid_policies)}")
    manifest = build_manifest(bundle, run_id, checkpoint, num_agents, seeds,
                              horizon, bound, policies, device, batch_size)
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                "existing bundle fingerprint differs; use a new --run-id/output directory"
            )
        manifest = previous
    if dry_run:
        return manifest
    bundle.mkdir(parents=True, exist_ok=True)
    archive_relative = _archive_checkpoint(
        bundle, checkpoint, manifest["checkpoint"],
        repair_invalid=repair_invalid)
    manifest["checkpoint"]["archive"] = archive_relative
    evaluation_checkpoint = bundle / archive_relative / "checkpoint"
    manifest["status"] = "running"
    manifest["updated_at_utc"] = _utc_now()
    atomic_write_json(manifest_path, manifest)

    checkpoint_info = manifest["checkpoint"]
    jobs = []
    for n_agents in num_agents:
        for policy in policies:
            jobs.append({
                "bundle": str(bundle), "run_id": run_id,
                "fingerprint": manifest["fingerprint"],
                "checkpoint": str(evaluation_checkpoint), "checkpoint_id": checkpoint_info["id"],
                "checkpoint_hash": checkpoint_info["package_sha256"],
                "obs_position_scale": checkpoint_info["obs_position_scale"],
                "config_sha256": manifest["spec"][
                    "effective_config_sha256_by_num_agents"][str(int(n_agents))],
                "dt": manifest["spec"]["dt"],
                "speed": manifest["spec"]["speed"],
                "r0": manifest["spec"]["r0"],
                "rho": manifest["spec"]["rho"],
                "n_agents": int(n_agents), "bound": float(bound),
                "seeds": list(seeds), "horizon": int(horizon), "policy": policy,
                "device": device, "batch_size": int(batch_size),
                "repair_invalid": bool(repair_invalid),
            })

    if device == "cpu" and workers > 1:
        context = mp.get_context("spawn")
        with context.Pool(min(int(workers), len(jobs))) as pool:
            grouped = pool.map(_run_group, jobs)
    else:
        grouped = [_run_group(job) for job in jobs]
    rows = [row for group in grouped for row in group]
    summarize(rows, bundle)
    manifest["status"] = "completed"
    manifest["episode_count"] = len(rows)
    manifest["updated_at_utc"] = _utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest


def reconstruct_cutoff_mask(states: np.ndarray, pointers: np.ndarray) -> np.ndarray:
    """Reconstruct masks from pre-step state using the env's exact rule.

    The environment first stores relative positions in a float32 buffer, then
    takes their norm.  Mirroring that cast matters for rare near-ties.  A self
    pointer is the explicit ``k=0`` action and therefore selects self only.
    """
    states = np.asarray(states, dtype=np.float64)
    pointers = np.asarray(pointers)
    horizon, n_agents = pointers.shape
    if states.shape[:2] != (horizon + 1, n_agents):
        raise ValueError("state/pointer shapes are inconsistent")
    if np.any(pointers < 0) or np.any(pointers >= n_agents):
        raise ValueError("pointer action out of range")
    positions = states[:-1, :, :2]
    relative = np.asarray(
        positions[:, None, :, :] - positions[:, :, None, :], dtype=np.float32)
    distances = np.linalg.norm(relative, axis=3)
    time_index = np.arange(horizon)[:, None]
    ego_index = np.arange(n_agents)[None, :]
    cutoff = distances[time_index, ego_index, pointers]
    result = distances <= cutoff[:, :, None]
    self_rows = np.where(pointers == ego_index)
    if self_rows[0].size:
        result[self_rows[0], self_rows[1], :] = False
        result[self_rows[0], self_rows[1], self_rows[1]] = True
    return result


def _nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def _derived_full_series(states: np.ndarray, binary: np.ndarray,
                         controls: np.ndarray, *, dt: float, speed: float,
                         rho: float, r0: float) -> Dict[str, np.ndarray]:
    """Recompute every canonical scalar series from physical trace arrays."""
    states = np.asarray(states, dtype=np.float64)
    binary = np.asarray(binary, dtype=bool)
    controls = np.asarray(controls, dtype=np.float64)
    horizon, n_agents = controls.shape
    result = {
        name: np.full(horizon + 1, np.nan, dtype=np.float64)
        for name in SCALAR_SERIES
    }
    positions = states[:, :, :2]
    velocities = states[:, :, 2:4]
    result["s_ent"] = np.sqrt(positions.var(axis=1).sum(axis=1))
    result["v_ent"] = np.sqrt(velocities.var(axis=1).sum(axis=1))
    speeds = np.linalg.norm(velocities, axis=2)
    units = velocities / np.maximum(speeds, 1e-12)[:, :, None]
    result["phi"] = np.linalg.norm(units.mean(axis=1), axis=1)
    result["s_ent_env"][1:] = result["s_ent"][1:]
    result["v_ent_env"][1:] = result["v_ent"][1:]

    for t, position in enumerate(positions):
        distances = _pairwise_dist(position)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        result["nnd_mean"][t] = nearest.mean()
        result["nnd_max"][t] = nearest.max()
        result["min_pair"][t] = nearest.min()
        finite_distances = distances.copy()
        finite_distances[np.isinf(finite_distances)] = 0.0
        result["diam"][t] = finite_distances.max()
        result["radius"][t] = np.linalg.norm(
            position - position.mean(axis=0), axis=1).max()
        result["n_comp_r0"][t] = _n_components(distances < r0)

    offdiag = binary & ~np.eye(n_agents, dtype=bool)[None, :, :]
    deg_agents = offdiag.sum(axis=2).astype(np.int16)
    for step in range(horizon):
        state_index = step + 1
        result["deg_mean"][state_index] = offdiag[step].sum() / n_agents
        result["n_comp_sel"][state_index] = _n_components(offdiag[step])
        if step:
            intersection = (offdiag[step] & offdiag[step - 1]).sum()
            union = (offdiag[step] | offdiag[step - 1]).sum()
            result["churn"][state_index] = 1.0 - (
                intersection / union if union else 1.0)
    control32 = controls.astype(np.float32)
    result["reward"][1:] = -(
        dt * speed * np.mean(np.abs(control32), axis=1) + rho * dt)
    result["deg_agents"] = deg_agents
    return result


def _assert_series_close(name: str, actual: np.ndarray,
                         expected: np.ndarray, path: Path) -> None:
    if not np.allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=2e-6, atol=5e-6, equal_nan=True):
        difference = np.abs(
            np.asarray(actual, dtype=np.float64)
            - np.asarray(expected, dtype=np.float64))
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if finite.size else float("nan")
        raise ValueError(
            f"physical {name} mismatch (max abs {maximum:.3e}): {path}")


def _same_scalar(actual, expected, *, atol: float = 1e-9) -> bool:
    if actual is None:
        return isinstance(expected, (float, np.floating)) and np.isnan(expected)
    try:
        left, right = float(actual), float(expected)
    except (TypeError, ValueError):
        return actual == expected
    if np.isnan(left) and np.isnan(right):
        return True
    return bool(np.isclose(left, right, rtol=1e-9, atol=atol))


def _recomputed_summary(view) -> Dict[str, object]:
    judgment = judge_episode_c2(
        view["phi"], view["s_ent"], view["n_comp_r0"], view["reward"])
    meta = view.meta
    summary = {
        "run_id": meta["run_id"], "suite": meta["suite"],
        "protocol_id": meta["protocol_id"],
        "checkpoint_id": meta["checkpoint_id"],
        "checkpoint_hash": meta["checkpoint_hash"],
        "num_agents": int(meta["num_agents"]), "bound": float(meta["bound"]),
        "seed": int(meta["seed"]), "action_seed": int(meta["action_seed"]),
        "policy": meta["policy"], "horizon": int(meta["horizon"]),
        **judgment,
        "phi_ss": _nanmedian(np.asarray(view["phi"])[-300:]),
        "sigma_p_ss": _nanmedian(np.asarray(view["s_ent"])[-300:]),
        "min_pair": float(np.nanmin(view["min_pair"])),
        "deg_ss": _nanmedian(np.asarray(view["deg_mean"])[-300:]),
        "churn_ss": _nanmedian(np.asarray(view["churn"])[-300:]),
        "n_comp_end": float(np.asarray(view["n_comp_r0"])[-1]),
    }
    stored_top = {
        "t_fire": int(np.asarray(view["t_fire"]).item()),
        "success": int(np.asarray(view["success"]).item()),
        "J": float(np.asarray(view["J"]).item()),
    }
    for key in ("t_fire", "success", "J"):
        if not _same_scalar(stored_top[key], judgment[key]):
            raise ValueError(f"stored top-level {key} mismatch: {view.path}")
    stored_summary = view.meta.get("summary")
    if not isinstance(stored_summary, dict):
        raise ValueError(f"canonical episode has no embedded summary: {view.path}")
    for key, expected in summary.items():
        if key not in stored_summary or not _same_scalar(stored_summary[key], expected):
            raise ValueError(f"stored summary {key} mismatch: {view.path}")
    initial_hash = _state_hash(np.asarray(view["agent_states"])[0])
    if view.meta.get("initial_state_sha256") != initial_hash:
        raise ValueError(f"initial-state hash mismatch: {view.path}")
    return summary


def validate_population_episode(view, expected: Dict[str, object],
                                deep: bool = False) -> Dict[str, object]:
    """Validate one canonical population episode and return trusted summary."""
    validate_full_episode(view, expected=expected)
    if view.schema_version != FULL_SCHEMA_VERSION or view.legacy:
        raise ValueError(f"population cache is not canonical: {view.path}")
    summary = _recomputed_summary(view)
    if not deep:
        return summary

    states = np.asarray(view["agent_states"], dtype=np.float64)
    pointers = np.asarray(view["pointer_actions"])
    binary = np.asarray(view["binary_actions"], dtype=bool)
    controls = np.asarray(view["control_inputs"], dtype=np.float64)
    rebuilt = reconstruct_cutoff_mask(states, pointers)
    if not np.array_equal(rebuilt, binary):
        raise ValueError(f"pointer/mask mismatch: {view.path}")
    derived = _derived_full_series(
        states, binary, controls,
        dt=float(expected["dt"]), speed=float(expected["speed"]),
        rho=float(expected["rho"]), r0=float(expected["r0"]),
    )
    for name in SCALAR_SERIES:
        _assert_series_close(name, view[name], derived[name], view.path)
    if not np.array_equal(np.asarray(view["deg_agents"]), derived["deg_agents"]):
        raise ValueError(f"per-agent degree mismatch: {view.path}")

    dt, speed = float(expected["dt"]), float(expected["speed"])
    control32 = controls.astype(np.float32)
    expected_positions = states[:-1, :, :2] + states[:-1, :, 2:4] * dt
    expected_headings = states[:-1, :, 4] + control32 * dt
    expected_velocities = np.asarray(
        speed * np.stack(
            [np.cos(expected_headings), np.sin(expected_headings)], axis=2),
        dtype=np.float32,
    ).astype(np.float64)
    if not np.allclose(states[1:, :, :2], expected_positions,
                       rtol=1e-9, atol=2e-6):
        raise ValueError(f"position transition mismatch: {view.path}")
    if not np.allclose(states[1:, :, 4], expected_headings,
                       rtol=1e-9, atol=2e-6):
        raise ValueError(f"heading transition mismatch: {view.path}")
    if not np.allclose(states[1:, :, 2:4], expected_velocities,
                       rtol=1e-7, atol=2e-6):
        raise ValueError(f"velocity transition mismatch: {view.path}")
    return summary


SUMMARY_PATHS = (
    Path("summaries/episodes.csv"),
    Path("summaries/aggregate.csv"),
    Path("summaries/paired_vs_acs.csv"),
    Path("summaries/paired_stats_vs_acs.csv"),
)


def _expected_summary_bytes(rows: Sequence[Dict[str, object]]):
    """Render trusted summaries outside the bundle for read-only comparison."""
    with tempfile.TemporaryDirectory(prefix="dynamic-k-summary-check.") as temporary:
        root = Path(temporary)
        summarize(rows, root)
        return {
            relative: (root / relative).read_bytes()
            for relative in SUMMARY_PATHS
        }


def _summary_csv_equivalent(actual: bytes, expected: bytes) -> bool:
    """Compare CSV meaning while tolerating harmless numeric formatting."""
    if actual == expected:
        return True
    try:
        actual_reader = csv.DictReader(io.StringIO(actual.decode("utf-8")))
        expected_reader = csv.DictReader(io.StringIO(expected.decode("utf-8")))
        if actual_reader.fieldnames != expected_reader.fieldnames:
            return False
        actual_rows = list(actual_reader)
        expected_rows = list(expected_reader)
    except (UnicodeDecodeError, csv.Error):
        return False
    if len(actual_rows) != len(expected_rows):
        return False

    def same_cell(left, right):
        left = "" if left is None else str(left).strip()
        right = "" if right is None else str(right).strip()
        if left == right:
            return True
        try:
            left_number, right_number = float(left), float(right)
        except ValueError:
            return False
        if np.isnan(left_number) and np.isnan(right_number):
            return True
        return bool(np.isclose(
            left_number, right_number, rtol=1e-12, atol=1e-12))

    columns = actual_reader.fieldnames or []
    return all(
        same_cell(actual_row.get(column), expected_row.get(column))
        for actual_row, expected_row in zip(actual_rows, expected_rows)
        for column in columns
    )


def _verify_summary_files(rows: Sequence[Dict[str, object]], bundle: Path) -> None:
    expected = _expected_summary_bytes(rows)
    for relative, content in expected.items():
        actual = bundle / relative
        if not actual.is_file():
            raise FileNotFoundError(
                "population summary is missing: {}; rerun validate with "
                "--rebuild-summaries to repair it".format(actual)
            )
        if not _summary_csv_equivalent(actual.read_bytes(), content):
            raise ValueError(
                "population summary differs from validated episodes: {}; rerun "
                "validate with --rebuild-summaries to repair it".format(actual)
            )


def validate_bundle(bundle: os.PathLike, deep: bool = False,
                    write_report: bool = False,
                    rebuild_summaries: bool = False) -> Dict[str, object]:
    """Validate a population bundle without mutating it by default.

    Summary CSVs are always checked against values recomputed from the episode
    artifacts.  ``rebuild_summaries`` is the explicit repair path and also
    records ``validation.json``; ``write_report`` records only that report.
    """
    bundle = Path(bundle).expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"bundle status is {manifest.get('status')!r}, not 'completed'")
    spec = manifest["spec"]
    checkpoint_id = manifest["checkpoint"]["id"]
    archived = _checkpoint_identity(bundle / manifest["checkpoint"]["archive"])
    if archived["package_sha256"] != manifest["checkpoint"]["package_sha256"]:
        raise ValueError("archived checkpoint package hash mismatch")
    paths, initial_hashes = [], {}
    rows = []
    for n_agents in spec["num_agents"]:
        for policy in spec["policies"]:
            for seed in spec["seeds"]:
                path = _episode_path(bundle, checkpoint_id, int(n_agents), policy, int(seed))
                if not path.is_file():
                    raise FileNotFoundError(str(path))
                expected = {
                    "run_id": manifest["run_id"], "suite": SUITE_ID,
                    "protocol_id": PROTOCOL_ID, "fingerprint": manifest["fingerprint"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_hash": manifest["checkpoint"]["package_sha256"],
                    "num_agents": int(n_agents), "bound": float(spec["bound"]),
                    "seed": int(seed),
                    "action_seed": derive_action_seed(int(seed), int(n_agents)),
                    "policy": policy, "horizon": int(spec["horizon"]),
                    "backend": spec["backend"], "batch_size": int(spec["batch_size"]),
                    "config_sha256": spec[
                        "effective_config_sha256_by_num_agents"][str(int(n_agents))],
                    "dt": float(spec["dt"]), "speed": float(spec["speed"]),
                    "r0": float(spec["r0"]), "rho": float(spec["rho"]),
                }
                view = load_episode(path)
                summary = validate_population_episode(view, expected, deep=deep)
                state_hash = _state_hash(np.asarray(view["agent_states"])[0])
                key = (int(n_agents), int(seed))
                if key in initial_hashes and initial_hashes[key] != state_hash:
                    raise ValueError(f"initial state mismatch across policies for {key}")
                initial_hashes[key] = state_hash
                summary["path"] = path.relative_to(bundle).as_posix()
                rows.append(summary)
                paths.append(path)
    if rebuild_summaries:
        summarize(rows, bundle)
        summary_status = "rebuilt"
    else:
        _verify_summary_files(rows, bundle)
        summary_status = "verified"
    report_written = bool(write_report or rebuild_summaries)
    result = {
        "schema_version": "validation-2.0", "run_id": manifest["run_id"],
        "fingerprint": manifest["fingerprint"], "deep": bool(deep),
        "episodes": len(paths), "paired_initial_state_groups": len(initial_hashes),
        "summaries": summary_status, "report_written": report_written,
        "status": "pass", "validated_at_utc": _utc_now(),
    }
    if report_written:
        atomic_write_json(bundle / "validation.json", result)
    return result
