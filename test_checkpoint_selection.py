"""Tests for deterministic, provenance-only checkpoint selection."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from eval.checkpoint_selection import (
    C2_SUCCESS_COLUMN,
    ITERATION_COLUMN,
    J_SUCCESS_COLUMN,
    T_CONV_COLUMN,
    discover_trial_directory,
    main,
    select_checkpoints,
)


FIELDS = [
    ITERATION_COLUMN,
    C2_SUCCESS_COLUMN,
    T_CONV_COLUMN,
    J_SUCCESS_COLUMN,
]


def write_progress(trial, rows, fieldnames=FIELDS):
    trial = Path(trial)
    trial.mkdir(parents=True, exist_ok=True)
    with (trial / "progress.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_checkpoint(trial, iteration, state=None):
    path = Path(trial) / "checkpoint_{:06d}".format(iteration)
    path.mkdir(parents=True)
    params = Path(trial) / "params.json"
    if not params.exists():
        params.write_text('{"model":{"custom_model":"unit"}}\n')
    content = state if state is not None else "state-{}".format(iteration).encode()
    (path / "algorithm_state.pkl").write_bytes(content)
    (path / "rllib_checkpoint.json").write_text('{"type": "Algorithm"}\n')
    return path


def row(iteration, success, t_conv, j_success):
    return {
        ITERATION_COLUMN: iteration,
        C2_SUCCESS_COLUMN: success,
        T_CONV_COLUMN: t_conv,
        J_SUCCESS_COLUMN: j_success,
    }


class CheckpointSelectionTest(unittest.TestCase):
    def test_metric_priority_intersection_final_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary) / "trial"
            write_progress(
                trial,
                [
                    row(1, 0.8, 100, 100),
                    row(2, 0.9, 600, 100),
                    row(3, 0.9, 500, 300),
                    row(4, 0.9, 500, 200),
                    row(5, 0.9, 500, 200),
                    row(6, 0.1, 50, 10),
                    row(7, 1.0, 1, 1),  # progress only
                ],
            )
            for iteration in range(1, 7):
                add_checkpoint(trial, iteration)
            add_checkpoint(trial, 8)  # checkpoint only

            payload = select_checkpoints(trial, top=5, include_final=True)

            self.assertEqual(
                [item["iteration"] for item in payload["selection"]["top"]],
                [5, 4, 3, 2, 1],
            )
            self.assertEqual(payload["selection"]["final"]["iteration"], 6)
            self.assertEqual(
                [item["iteration"] for item in payload["checkpoints"]],
                [5, 4, 3, 2, 1, 6],
            )
            self.assertEqual(payload["discovery"]["progress_only_iterations"], [7])
            self.assertEqual(payload["discovery"]["checkpoint_only_iterations"], [8])
            record = payload["checkpoints"][0]
            self.assertEqual(record["source_trial"], str(trial.resolve()))
            self.assertEqual(record["path"], str((trial / "checkpoint_000005").resolve()))
            self.assertEqual(
                record["checkpoint_state_sha256"],
                hashlib.sha256(b"state-5").hexdigest(),
            )
            self.assertEqual(len(record["checkpoint_tree_sha256"]), 64)
            self.assertEqual(len(record["checkpoint_package_sha256"]), 64)
            self.assertEqual(len(record["checkpoint_params_sha256"]), 64)

    def test_latest_is_deduplicated_when_already_top_ranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary) / "trial"
            write_progress(trial, [row(1, 0.5, 500, 500), row(2, 1.0, 100, 100)])
            add_checkpoint(trial, 1)
            add_checkpoint(trial, 2)

            payload = select_checkpoints(trial, top=1, include_final=True)

            self.assertEqual(len(payload["checkpoints"]), 1)
            self.assertEqual(payload["checkpoints"][0]["iteration"], 2)
            self.assertEqual(payload["checkpoints"][0]["roles"], ["top_01", "final"])

    def test_unrankable_latest_can_still_be_included_as_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary) / "trial"
            write_progress(trial, [row(1, 0.9, 100, 100), row(2, "", "", "")])
            add_checkpoint(trial, 1)
            add_checkpoint(trial, 2)

            payload = select_checkpoints(trial, top=5, include_final=True)

            self.assertEqual([item["iteration"] for item in payload["selection"]["top"]], [1])
            self.assertEqual(payload["selection"]["final"]["iteration"], 2)
            self.assertEqual(payload["discovery"]["unrankable_matched_iterations"], [2])
            final = next(item for item in payload["checkpoints"] if item["iteration"] == 2)
            self.assertTrue(all(value is None for value in final["metrics"].values()))

    def test_trial_discovery_fails_on_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            first = run / "trial_a"
            second = run / "trial_b"
            write_progress(first, [row(1, 1, 1, 1)])
            self.assertEqual(
                discover_trial_directory(run_dir=run), first.resolve()
            )
            write_progress(second, [row(1, 1, 1, 1)])
            with self.assertRaisesRegex(ValueError, "Ambiguous trial discovery"):
                discover_trial_directory(run_dir=run)

    def test_duplicate_progress_iteration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary) / "trial"
            write_progress(trial, [row(1, 1, 100, 100), row(1, 0.5, 200, 200)])
            add_checkpoint(trial, 1)
            with self.assertRaisesRegex(ValueError, "duplicate training iteration"):
                select_checkpoints(trial)

    def test_cli_dry_run_and_json_write_without_checkpoint_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / "run" / "trial"
            write_progress(trial, [row(1, 1.0, 100, 20)])
            checkpoint = add_checkpoint(trial, 1, state=b"immutable-state")
            original_files = sorted(path.relative_to(checkpoint) for path in checkpoint.rglob("*"))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["--run-dir", str(root / "run"), "--dry-run"])
            self.assertEqual(result, 0)
            dry_payload = json.loads(stdout.getvalue())
            self.assertEqual(dry_payload["selection"]["final"]["iteration"], 1)

            output = root / "selection" / "checkpoints.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--trial-dir",
                        str(trial),
                        "--top",
                        "1",
                        "--no-include-final",
                        "--out",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue().strip(), str(output.resolve()))
            payload = json.loads(output.read_text())
            self.assertIsNone(payload["selection"]["final"])
            self.assertEqual(payload["checkpoints"][0]["roles"], ["top_01"])
            self.assertEqual(
                sorted(path.relative_to(checkpoint) for path in checkpoint.rglob("*")),
                original_files,
            )
            self.assertEqual((checkpoint / "algorithm_state.pkl").read_bytes(), b"immutable-state")


if __name__ == "__main__":
    unittest.main()
