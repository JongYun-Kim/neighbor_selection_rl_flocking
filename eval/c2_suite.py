"""Staged checkpoint evaluation on the main C2 dev/confirmation lanes."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from eval.artifacts import (
    atomic_open, atomic_write_json, make_staging_directory, run_fingerprint,
    sha256_file, sha256_tree)
from eval.checkpoint_selection import checkpoint_state_paths
from eval.common import build_config
from eval.eval_c2 import judge_npz, run_one
from eval.policies import is_dknn_params
from eval.provenance import git_snapshot, runtime_snapshot
from eval.protocol import MAIN_C2_V1


LANES = {
    "dev": {"seeds": list(range(1000, 1032)), "criterion_of_record": False},
    "confirm": {"seeds": list(range(1500, 2000)), "criterion_of_record": True},
}
PROTOCOL_ID = "main_c2_v1"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _checkpoint_identity(checkpoint: Path) -> Dict[str, object]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(str(checkpoint))
    params_path = checkpoint.parent / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"checkpoint has no sibling params.json: {params_path}")
    tree_hash = sha256_tree(checkpoint)
    params_hash = sha256_file(params_path)
    state_files = checkpoint_state_paths(checkpoint)
    state_hash = sha256_file(state_files[0]) if state_files else None
    return {
        "tree_sha256": tree_hash,
        "params_sha256": params_hash,
        "state_sha256": state_hash,
        "package_sha256": run_fingerprint({
            "checkpoint_tree_sha256": tree_hash,
            "params_json_sha256": params_hash,
        }),
    }


def _effective_config(checkpoint: Path, steps: int, bound: float,
                      n_agents: int) -> Dict[str, object]:
    params_path = Path(checkpoint).parent / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    cfg = build_config(
        n_agents=n_agents, max_steps=steps, initial_position_bound=bound)
    env_params = params.get("env_config", {}).get("config", {}).get("env", {})
    if is_dknn_params(params):
        cfg.env.action_type = "dynamic_k_nn"
        cfg.env.evaluation_diagnostics = True
        cfg.env.expose_aux_target = False
        cfg.env.expose_global_stats = False
    else:
        cfg.env.expose_aux_target = True
        cfg.env.expose_global_stats = True
    cfg.env.obs_position_scale = env_params.get("obs_position_scale", "legacy")
    return cfg.dict()


def _archive_confirmation_checkpoint(bundle: Path,
                                     candidate: Dict[str, object],
                                     repair_invalid: bool) -> Path:
    """Atomically copy the one confirmed RLlib load package into the bundle."""
    destination = bundle / "finalist"
    expected_hash = candidate["package_sha256"]
    archived_checkpoint = destination / "checkpoint"
    if destination.exists():
        try:
            if _checkpoint_identity(archived_checkpoint)["package_sha256"] != expected_hash:
                raise ValueError("archived confirmation finalist package differs")
            return archived_checkpoint
        except Exception:
            if not repair_invalid:
                raise
            _quarantine(destination, bundle)

    bundle.mkdir(parents=True, exist_ok=True)
    staging = make_staging_directory(bundle, ".tmp-finalist.")
    try:
        shutil.copytree(str(candidate["checkpoint"]), str(staging / "checkpoint"))
        shutil.copy2(
            str(Path(candidate["checkpoint"]).parent / "params.json"),
            str(staging / "params.json"),
        )
        if _checkpoint_identity(staging / "checkpoint")["package_sha256"] != expected_hash:
            raise RuntimeError("confirmation finalist archive checksum failed")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return archived_checkpoint


def candidates_from_json(path: os.PathLike) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = []
    for item in payload.get("checkpoints", []):
        checkpoint = Path(item["resolved_path"]).expanduser().resolve()
        actual = _checkpoint_identity(checkpoint)
        recorded_package = item.get("checkpoint_package_sha256")
        if recorded_package is None:
            raise ValueError(
                "selection JSON lacks checkpoint package hashes; rerun checkpoints")
        comparisons = {
            "tree_sha256": item.get("checkpoint_tree_sha256"),
            "params_sha256": item.get("checkpoint_params_sha256"),
            "state_sha256": item.get("checkpoint_state_sha256"),
            "package_sha256": recorded_package,
        }
        for key, recorded in comparisons.items():
            if recorded is not None and recorded != actual[key]:
                raise ValueError(f"selected checkpoint {key} changed: {checkpoint}")
        result.append({
            "checkpoint": checkpoint,
            "iteration": int(item["iteration"]),
            **actual,
            "screen_roles": list(item.get("roles", [])),
            "screen_metrics": dict(item.get("metrics", {})),
        })
    if not result:
        raise ValueError(f"selection JSON contains no checkpoints: {path}")
    return result


def explicit_candidate(path: os.PathLike) -> Dict[str, object]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(str(checkpoint))
    return {
        "checkpoint": checkpoint, "iteration": _checkpoint_iteration(checkpoint),
        **_checkpoint_identity(checkpoint),
        "screen_roles": ["explicit"], "screen_metrics": {},
    }


def _label(candidate: Dict[str, object], prefix: str) -> str:
    iteration = int(candidate["iteration"])
    suffix = f"ck{iteration:06d}" if iteration >= 0 else Path(candidate["checkpoint"]).name
    return f"{prefix}_{suffix}"


def _validate_cached(path: Path, seed: int, checkpoint: Path, steps: int,
                     bound: float, n_agents: int):
    with np.load(path, allow_pickle=True) as archive:
        meta = json.loads(str(archive["meta"]))
        required = {"phi", "s_ent", "n_comp_r0", "reward"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"cached C2 artifact missing {sorted(missing)}: {path}")
        if any(np.asarray(archive[key]).shape != (steps + 1,) for key in required):
            raise ValueError(f"cached C2 series length mismatch: {path}")
    expected = {
        "seed": int(seed), "max_steps": int(steps), "n_agents": int(n_agents),
        "initial_position_bound": float(bound), "ckpt": str(checkpoint),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"cached C2 metadata mismatch {key}: {meta.get(key)!r} != {value!r}"
            )
    return judge_npz(path)


def _quarantine(path: Path, bundle: Path):
    target = bundle / "quarantine" / path.relative_to(bundle)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target = target.with_name(target.stem + "-" + datetime.now().strftime("%H%M%S%f") + target.suffix)
    shutil.move(str(path), str(target))


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_open(path, "w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(list(rows)).to_csv(stream, index=False)


def evaluate_c2_lane(candidates: Sequence[Dict[str, object]], lane: str,
                     bundle: os.PathLike, run_id: str, workers: int = 8,
                     seeds: Optional[Sequence[int]] = None, steps: int = 6000,
                     bound: float = 250.0, n_agents: int = 20,
                     repair_invalid: bool = False):
    if lane not in LANES:
        raise ValueError(f"unknown lane: {lane}")
    if lane == "confirm" and len(candidates) != 1:
        raise ValueError("confirmation accepts exactly one explicitly chosen checkpoint")
    if lane == "confirm" and "explicit" not in candidates[0].get("screen_roles", []):
        raise ValueError("confirmation requires an explicitly chosen checkpoint")
    official_seeds = LANES[lane]["seeds"]
    seeds = list(official_seeds if seeds is None else seeds)
    official = (
        lane == "confirm" and seeds == official_seeds and steps == 6000
        and float(bound) == 250.0 and int(n_agents) == 20
    )
    bundle = Path(bundle).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    git = git_snapshot(repo_root)
    runtime = runtime_snapshot()
    effective_configs = []
    for candidate in candidates:
        checkpoint = Path(candidate["checkpoint"])
        actual = _checkpoint_identity(checkpoint)
        if actual["package_sha256"] != candidate["package_sha256"]:
            raise ValueError(f"checkpoint package changed before evaluation: {checkpoint}")
        config = _effective_config(checkpoint, steps, bound, n_agents)
        effective_configs.append({
            "checkpoint_package_sha256": actual["package_sha256"],
            "config_sha256": run_fingerprint(config),
            "config": config,
        })
    spec = {
        "protocol_id": PROTOCOL_ID, "lane": lane, "run_id": run_id,
        "seeds": [int(seed) for seed in seeds], "steps": int(steps),
        "bound": float(bound), "n_agents": int(n_agents),
        "inference": {
            "action_selection": "deterministic_argmax",
            "backend": "cpu",
            "batch_size": 1,
        },
        "episode_semantics": {
            "fixed_horizon": True,
            "is_training": False,
            "judgment": "offline_after_full_horizon",
        },
        "checkpoint_packages": [item["package_sha256"] for item in candidates],
        "effective_configs": effective_configs,
        "git_commit": git["commit"], "git_dirty": git["dirty"],
        "git_dirty_diff_sha256": git["dirty_diff_sha256"],
        "runtime": runtime,
    }
    fingerprint = run_fingerprint(spec)
    manifest_path = bundle / "manifest.json"
    manifest = {
        "schema_version": "evaluation-bundle-2.0", "run_id": run_id,
        "suite": f"c2_{lane}", "protocol": {
            "id": PROTOCOL_ID, "criterion_of_record": official,
            "phi_goal_strict_gt": MAIN_C2_V1.phi_goal,
            "alignment_window": MAIN_C2_V1.alignment_window,
            "stability_window": MAIN_C2_V1.stability_window,
            "spatial_band_strict_lt": MAIN_C2_V1.spatial_band_epsilon,
            "proximity_edge": "distance < r0",
        },
        "fingerprint": fingerprint, "spec": spec,
        "checkpoints": [
            {**{key: value for key, value in item.items() if key != "checkpoint"},
             "checkpoint": str(item["checkpoint"])} for item in candidates
        ],
        "git": git, "runtime": runtime,
        "status": "running", "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("existing C2 bundle fingerprint differs")
        manifest = previous
        manifest["status"] = "running"
        manifest["updated_at_utc"] = _utc_now()
    bundle.mkdir(parents=True, exist_ok=True)
    execution_paths = {
        str(item["checkpoint"]): Path(item["checkpoint"]) for item in candidates
    }
    if lane == "confirm":
        archived = _archive_confirmation_checkpoint(
            bundle, candidates[0], repair_invalid=repair_invalid)
        execution_paths[str(candidates[0]["checkpoint"])] = archived
        manifest["checkpoints"][0]["archive"] = "finalist"
    atomic_write_json(manifest_path, manifest)

    all_rows = []
    for candidate in candidates:
        source_checkpoint = Path(candidate["checkpoint"])
        checkpoint = execution_paths[str(candidate["checkpoint"])]
        label = _label(candidate, run_id)
        outdir = bundle / "episodes" / f"c2_{lane}" / label
        outdir.mkdir(parents=True, exist_ok=True)
        jobs = []
        cached = []
        for seed in seeds:
            path = outdir / f"{label}_s{seed}.npz"
            if path.is_file():
                try:
                    cached.append((path, _validate_cached(
                        path, seed, checkpoint, steps, bound, n_agents)))
                    continue
                except Exception:
                    if not repair_invalid:
                        raise
                    _quarantine(path, bundle)
            jobs.append((str(checkpoint), label, int(seed), int(steps), float(bound),
                         int(n_agents), str(outdir)))
        if jobs:
            with mp.get_context("spawn").Pool(min(int(workers), len(jobs))) as pool:
                generated_paths = [Path(path) for path in pool.map(run_one, jobs)]
        else:
            generated_paths = []
        paths = [path for path, _ in cached] + generated_paths
        rows = []
        for path in sorted(paths, key=lambda value: int(value.stem.rsplit("s", 1)[1])):
            row = judge_npz(path)
            row.update({
                "checkpoint_id": source_checkpoint.name,
                "checkpoint_iteration": int(candidate["iteration"]),
                "checkpoint_tree_sha256": candidate["tree_sha256"],
                "checkpoint_package_sha256": candidate["package_sha256"],
                "label": label, "lane": lane,
                "path": path.relative_to(bundle).as_posix(),
            })
            rows.append(row)
        _write_csv(bundle / "summaries" / f"{label}_summary.csv", rows)
        all_rows.extend(rows)

    import pandas as pd
    frame = pd.DataFrame(all_rows)
    ranking = []
    for (checkpoint_id, iteration), group in frame.groupby(
            ["checkpoint_id", "checkpoint_iteration"]):
        success = group[group.success == 1]
        ranking.append({
            "checkpoint_id": checkpoint_id, "checkpoint_iteration": int(iteration),
            "n": len(group), "successes": int(group.success.sum()),
            "failures": int((group.success == 0).sum()),
            "t_fire_median": float(success.t_fire.median()) if len(success) else None,
            "J_median": float(success.J.median()) if len(success) else None,
        })
    ranking.sort(key=lambda row: (
        row["failures"],
        math.inf if row["t_fire_median"] is None else row["t_fire_median"],
        math.inf if row["J_median"] is None else row["J_median"],
        -row["checkpoint_iteration"],
    ))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    _write_csv(bundle / "summaries" / "episodes.csv", all_rows)
    _write_csv(bundle / "summaries" / "ranking.csv", ranking)
    atomic_write_json(bundle / "checkpoint_selection.json", {
        "schema_version": "c2-dev-ranking-1.0", "lane": lane,
        "ranking_rule": ["failures asc", "t_fire_median asc", "J_median asc",
                         "checkpoint_iteration desc"],
        "ranking": ranking,
        "automatic_finalist": None,
    })
    manifest["status"] = "completed"
    manifest["episode_count"] = len(all_rows)
    manifest["updated_at_utc"] = _utc_now()
    atomic_write_json(manifest_path, manifest)
    return manifest, ranking
