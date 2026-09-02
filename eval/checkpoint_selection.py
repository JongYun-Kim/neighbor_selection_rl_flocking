"""Select evaluated RLlib checkpoints without copying checkpoint data.

The selector intersects rows from one trial's ``progress.csv`` with checkpoint
entries that actually exist in that trial directory.  Rankable checkpoints are
ordered by the in-training C2 metrics, in this exact priority order:

1. C2 success, descending
2. convergence time, ascending
3. successful-episode J, ascending
4. training iteration, descending

The emitted JSON is a provenance manifest only.  It contains resolved source
paths and content hashes, but never copies or modifies a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from eval.artifacts import (
    atomic_write_json, run_fingerprint, sha256_file, sha256_tree)


ITERATION_COLUMN = "training_iteration"
C2_SUCCESS_COLUMN = "evaluation/custom_metrics/c2_success_mean"
T_CONV_COLUMN = "evaluation/custom_metrics/t_conv_mean"
J_SUCCESS_COLUMN = "evaluation/custom_metrics/J_success_mean"
METRIC_COLUMNS: Tuple[str, ...] = (
    C2_SUCCESS_COLUMN,
    T_CONV_COLUMN,
    J_SUCCESS_COLUMN,
)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)$")
LEGACY_STATE_PATTERN = re.compile(r"^checkpoint-\d+$")
SCHEMA_VERSION = "c2-checkpoint-selection-1.1"


@dataclass(frozen=True)
class CheckpointCandidate:
    """One actual checkpoint matched to exactly one progress row."""

    path: Path
    iteration: int
    metrics: Mapping[str, Optional[float]]

    @property
    def is_rankable(self) -> bool:
        return all(self.metrics[column] is not None for column in METRIC_COLUMNS)

    def ranking_key(self) -> Tuple[float, float, float, int]:
        if not self.is_rankable:
            raise ValueError(
                "Checkpoint iteration {} lacks a finite ranking metric".format(
                    self.iteration
                )
            )
        return (
            -float(self.metrics[C2_SUCCESS_COLUMN]),
            float(self.metrics[T_CONV_COLUMN]),
            float(self.metrics[J_SUCCESS_COLUMN]),
            -int(self.iteration),
        )


def _resolved_directory(path: Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError("{} is not a directory: {}".format(label, path))
    return path


def discover_trial_directory(
    *, run_dir: Optional[Path] = None, trial_dir: Optional[Path] = None
) -> Path:
    """Resolve exactly one trial and fail rather than guessing on ambiguity."""

    if (run_dir is None) == (trial_dir is None):
        raise ValueError("Exactly one of run_dir and trial_dir must be provided")

    source = _resolved_directory(
        trial_dir if trial_dir is not None else run_dir,
        "trial_dir" if trial_dir is not None else "run_dir",
    )
    progress_files = set()
    direct = source / "progress.csv"
    if direct.is_file():
        progress_files.add(direct.resolve())
    progress_files.update(path.resolve() for path in source.rglob("progress.csv"))
    candidates = sorted({path.parent for path in progress_files}, key=str)
    if not candidates:
        raise FileNotFoundError("No progress.csv was found under {}".format(source))
    if len(candidates) != 1:
        raise ValueError(
            "Ambiguous trial discovery under {}: {}".format(
                source, ", ".join(str(path) for path in candidates)
            )
        )
    return candidates[0]


def _parse_iteration(value: object, row_number: int) -> int:
    text = "" if value is None else str(value).strip()
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(
            "Invalid {} at progress.csv row {}: {!r}".format(
                ITERATION_COLUMN, row_number, value
            )
        ) from error
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(
            "Invalid {} at progress.csv row {}: {!r}".format(
                ITERATION_COLUMN, row_number, value
            )
        )
    return int(number)


def _finite_float(value: object) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_progress_rows(progress_csv: Path) -> Dict[int, Dict[str, Optional[float]]]:
    """Read unique iteration rows and preserve missing metrics as ``None``."""

    progress_csv = Path(progress_csv).expanduser().resolve()
    if not progress_csv.is_file():
        raise FileNotFoundError(str(progress_csv))
    with progress_csv.open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {ITERATION_COLUMN, *METRIC_COLUMNS}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "{} is missing required columns: {}".format(
                    progress_csv, ", ".join(missing)
                )
            )
        rows: Dict[int, Dict[str, Optional[float]]] = {}
        for row_number, row in enumerate(reader, start=2):
            iteration = _parse_iteration(row.get(ITERATION_COLUMN), row_number)
            if iteration in rows:
                raise ValueError(
                    "Ambiguous duplicate training iteration {} in {}".format(
                        iteration, progress_csv
                    )
                )
            rows[iteration] = {
                column: _finite_float(row.get(column)) for column in METRIC_COLUMNS
            }
    if not rows:
        raise ValueError("{} contains no progress rows".format(progress_csv))
    return rows


def discover_checkpoint_paths(trial_dir: Path) -> Dict[int, Path]:
    """Return actual immediate ``checkpoint_*`` entries keyed by iteration."""

    trial_dir = _resolved_directory(trial_dir, "trial_dir")
    by_iteration: Dict[int, Path] = {}
    for path in sorted(trial_dir.glob("checkpoint_*"), key=lambda value: value.name):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None or not path.exists():
            continue
        iteration = int(match.group(1))
        resolved = path.resolve()
        if iteration in by_iteration:
            raise ValueError(
                "Ambiguous checkpoint iteration {}: {} and {}".format(
                    iteration, by_iteration[iteration], resolved
                )
            )
        by_iteration[iteration] = resolved
    if not by_iteration:
        raise FileNotFoundError(
            "No actual checkpoint_* entries were found in {}".format(trial_dir)
        )
    return by_iteration


def matched_checkpoint_candidates(trial_dir: Path) -> Tuple[List[CheckpointCandidate], dict]:
    """Intersect progress iterations with actual checkpoint iterations."""

    trial_dir = _resolved_directory(trial_dir, "trial_dir")
    progress_rows = read_progress_rows(trial_dir / "progress.csv")
    checkpoint_paths = discover_checkpoint_paths(trial_dir)
    shared = sorted(set(progress_rows).intersection(checkpoint_paths))
    if not shared:
        raise ValueError(
            "progress.csv and checkpoint_* have no shared iteration in {}".format(
                trial_dir
            )
        )
    candidates = [
        CheckpointCandidate(
            path=checkpoint_paths[iteration],
            iteration=iteration,
            metrics=progress_rows[iteration],
        )
        for iteration in shared
    ]
    discovery = {
        "progress_iterations": sorted(progress_rows),
        "checkpoint_iterations": sorted(checkpoint_paths),
        "matched_iterations": shared,
        "progress_only_iterations": sorted(set(progress_rows).difference(checkpoint_paths)),
        "checkpoint_only_iterations": sorted(set(checkpoint_paths).difference(progress_rows)),
        "unrankable_matched_iterations": [
            candidate.iteration for candidate in candidates if not candidate.is_rankable
        ],
    }
    return candidates, discovery


def checkpoint_state_paths(checkpoint_path: Path) -> List[Path]:
    """Return recognizable checkpoint state files in stable priority order."""

    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.is_file():
        return [checkpoint_path]
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(str(checkpoint_path))

    candidates = [
        checkpoint_path / "algorithm_state.pkl",
        checkpoint_path / "policies" / "default_policy" / "policy_state.pkl",
    ]
    candidates.extend(
        path
        for path in sorted(checkpoint_path.iterdir(), key=lambda value: value.name)
        if path.is_file() and LEGACY_STATE_PATTERN.fullmatch(path.name)
    )
    output: List[Path] = []
    seen = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(resolved)
    return output


def _checkpoint_record(
    candidate: CheckpointCandidate, source_trial: Path, roles: Sequence[str]
) -> dict:
    state_files = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in checkpoint_state_paths(candidate.path)
    ]
    primary_state = state_files[0] if state_files else None
    params_path = candidate.path.parent / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(
            "checkpoint selection requires sibling params.json: {}".format(
                params_path))
    tree_hash = sha256_tree(candidate.path)
    params_hash = sha256_file(params_path)
    package_hash = run_fingerprint({
        "checkpoint_tree_sha256": tree_hash,
        "params_json_sha256": params_hash,
    })
    return {
        "path": str(candidate.path),
        "resolved_path": str(candidate.path),
        "iteration": int(candidate.iteration),
        "metrics": {
            column: candidate.metrics[column] for column in METRIC_COLUMNS
        },
        "roles": list(roles),
        "source_trial": str(source_trial),
        "checkpoint_tree_sha256": tree_hash,
        "checkpoint_params_path": str(params_path.resolve()),
        "checkpoint_params_sha256": params_hash,
        "checkpoint_package_sha256": package_hash,
        "checkpoint_state_path": primary_state["path"] if primary_state else None,
        "checkpoint_state_sha256": primary_state["sha256"] if primary_state else None,
        "checkpoint_state_files": state_files,
    }


def select_checkpoints(
    trial_dir: Path, *, top: int = 5, include_final: bool = True
) -> dict:
    """Build a deduplicated, JSON-serializable checkpoint selection manifest."""

    if int(top) <= 0:
        raise ValueError("top must be positive")
    trial_dir = _resolved_directory(trial_dir, "trial_dir")
    candidates, discovery = matched_checkpoint_candidates(trial_dir)
    rankable = sorted(
        (candidate for candidate in candidates if candidate.is_rankable),
        key=lambda candidate: candidate.ranking_key(),
    )
    if not rankable:
        raise ValueError(
            "No matched checkpoint has all three finite C2 ranking metrics"
        )
    top_candidates = rankable[: int(top)]
    latest = max(candidates, key=lambda candidate: candidate.iteration)

    roles_by_path: Dict[Path, List[str]] = {}
    candidates_by_path: Dict[Path, CheckpointCandidate] = {}
    ordered_paths: List[Path] = []

    def add(candidate: CheckpointCandidate, role: str) -> None:
        if candidate.path not in roles_by_path:
            roles_by_path[candidate.path] = []
            candidates_by_path[candidate.path] = candidate
            ordered_paths.append(candidate.path)
        roles_by_path[candidate.path].append(role)

    for rank, candidate in enumerate(top_candidates, start=1):
        add(candidate, "top_{:02d}".format(rank))
    if include_final:
        add(latest, "final")

    checkpoints = [
        _checkpoint_record(
            candidates_by_path[path], trial_dir, roles_by_path[path]
        )
        for path in ordered_paths
    ]
    top_references = [
        {
            "rank": rank,
            "iteration": candidate.iteration,
            "path": str(candidate.path),
        }
        for rank, candidate in enumerate(top_candidates, start=1)
    ]
    final_reference = (
        {"iteration": latest.iteration, "path": str(latest.path)}
        if include_final
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_trial": str(trial_dir),
        "progress_csv": str((trial_dir / "progress.csv").resolve()),
        "selection_policy": {
            "top_requested": int(top),
            "top_selected": len(top_candidates),
            "include_final": bool(include_final),
            "ranking": [
                {"column": C2_SUCCESS_COLUMN, "direction": "descending"},
                {"column": T_CONV_COLUMN, "direction": "ascending"},
                {"column": J_SUCCESS_COLUMN, "direction": "ascending"},
                {"column": ITERATION_COLUMN, "direction": "descending"},
            ],
        },
        "discovery": discovery,
        "selection": {"top": top_references, "final": final_reference},
        "checkpoints": checkpoints,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    atomic_write_json(Path(path).expanduser().resolve(), payload)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank in-training C2 checkpoints and write a provenance JSON"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--trial-dir", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--include-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the latest matched checkpoint (default: true)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the JSON selection without writing --out",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.top <= 0:
        parser.error("--top must be positive")
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run is used")

    trial_dir = discover_trial_directory(
        run_dir=args.run_dir, trial_dir=args.trial_dir
    )
    payload = select_checkpoints(
        trial_dir, top=args.top, include_final=args.include_final
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        _write_json_atomic(args.out, payload)
        print(str(args.out.expanduser().resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
