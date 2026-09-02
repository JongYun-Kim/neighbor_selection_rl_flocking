"""Unified command line for the maintained evaluation workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from utils.paths import repo_path


DEFAULT_ROOT = Path(os.environ.get(
    "ARTIFACT_ROOT", repo_path("test_results", "evaluation")
))
DEFAULT_DEVICE = os.environ.get("DEVICE", "cpu").strip().lower()
if DEFAULT_DEVICE == "gpu":
    DEFAULT_DEVICE = "cuda"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _run_id(value):
    if not RUN_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run ID must contain only letters, digits, dot, underscore, and hyphen"
        )
    return value


def _comma_ints(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 2 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("agent counts must be unique integers >=2")
    return values


def _comma_strings(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a unique comma-separated list")
    return values


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="Main-C2 checkpoint, population, validation, and analysis tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dispatch is delegated to checkpoint_selection.main before this parser is
    # invoked; registering the name here keeps the unified top-level help
    # complete without duplicating its option definitions.
    subparsers.add_parser(
        "checkpoints", help="rank a training run and export finalist checkpoints")

    c2 = subparsers.add_parser("c2", help="run the staged C2 dev/confirmation lane")
    c2.add_argument("--lane", choices=("dev", "confirm"), required=True)
    source = c2.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidates", type=Path)
    source.add_argument("--checkpoint", type=Path)
    c2.add_argument("--run-id", required=True, type=_run_id)
    c2.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    c2.add_argument("--workers", type=int, default=8)
    c2.add_argument("--seeds", help="custom A-B or comma list; marks the run non-official")
    c2.add_argument("--steps", type=int, default=6000)
    c2.add_argument("--bound", type=float, default=250.0)
    c2.add_argument("--n-agents", type=int, default=20)
    c2.add_argument("--repair-invalid", action="store_true")

    population = subparsers.add_parser(
        "population", help="N=10/20/40 full-trace policy population evaluation")
    population.add_argument("--checkpoint", required=True, type=Path)
    population.add_argument("--run-id", required=True, type=_run_id)
    population.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    population.add_argument("--num-agents", type=_comma_ints, default=[10, 20, 40])
    population.add_argument("--seeds", default="0-49")
    population.add_argument("--steps", type=int, default=6000)
    population.add_argument("--bound", type=float, default=250.0)
    population.add_argument("--policies", type=_comma_strings,
                            default=list(("deterministic", "stochastic", "pure_acs")))
    population.add_argument("--device", default=DEFAULT_DEVICE)
    population.add_argument("--batch-size", type=int)
    population.add_argument("--workers", type=int, default=8)
    population.add_argument("--repair-invalid", action="store_true")
    population.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="validate a population bundle")
    validate.add_argument("--run", required=True, type=Path)
    validate.add_argument("--deep", action="store_true")
    validate.add_argument(
        "--write-report",
        action="store_true",
        help="write validation.json; validation is read-only by default",
    )
    validate.add_argument(
        "--rebuild-summaries",
        action="store_true",
        help=("replace summary CSVs from validated episodes and write "
              "validation.json"),
    )

    # Analysis parsers intentionally keep their public surface here; the
    # implementation modules also expose main(argv) for direct use/testing.
    radii = subparsers.add_parser("radii", help="single-episode cutoff-radius view")
    radii.add_argument("--run", required=True, type=Path)
    radii.add_argument("--n", type=int, required=True)
    radii.add_argument("--seed", type=int, required=True)
    radii.add_argument(
        "--output",
        type=Path,
        help=("default: canonical bundle analysis/radii; required outside a "
              "legacy/noncanonical input"),
    )
    radii.add_argument("--animation", action="store_true")

    heatmaps = subparsers.add_parser("heatmaps", help="ranked population cutoff heatmaps")
    heatmaps.add_argument("--run", required=True, type=Path)
    heatmaps.add_argument(
        "--output",
        type=Path,
        help=("default: canonical bundle analysis/ranked_heatmaps; required "
              "outside a legacy/noncanonical input"),
    )
    heatmaps.add_argument("--t-max-seconds", type=float)
    heatmaps.add_argument("--with-entropies", action="store_true")

    control = subparsers.add_parser("control-effort", help="main-C2 L1/L2 effort analysis")
    control.add_argument("--run", required=True, type=Path)
    control.add_argument(
        "--output",
        type=Path,
        help=("default: canonical bundle analysis/control_effort; required "
              "outside a legacy/noncanonical input"),
    )
    control.add_argument("--animations", action="store_true")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "checkpoints":
        from eval.checkpoint_selection import main as checkpoint_main
        return checkpoint_main(argv[1:])

    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) <= 0:
        parser.error("--workers must be positive")
    if getattr(args, "steps", 1) <= 0:
        parser.error("--steps must be positive")

    if args.command == "c2":
        from eval.c2_suite import (
            candidates_from_json, evaluate_c2_lane, explicit_candidate)
        from eval.population import parse_int_range
        if args.lane == "confirm" and args.candidates:
            parser.error("confirmation requires --checkpoint, never --candidates")
        candidates = (candidates_from_json(args.candidates) if args.candidates
                      else [explicit_candidate(args.checkpoint)])
        seeds = parse_int_range(args.seeds) if args.seeds else None
        bundle = args.output_root.expanduser().resolve() / args.run_id / f"c2_{args.lane}"
        _, ranking = evaluate_c2_lane(
            candidates, args.lane, bundle, args.run_id, workers=args.workers,
            seeds=seeds, steps=args.steps, bound=args.bound, n_agents=args.n_agents,
            repair_invalid=args.repair_invalid)
        print(json.dumps({"bundle": str(bundle), "ranking": ranking}, indent=2,
                         sort_keys=True, allow_nan=False))
        return 0

    if args.command == "population":
        from eval.population import parse_int_range, run_population
        if args.device.startswith("cuda") and args.batch_size is None:
            parser.error("CUDA population evaluation requires explicit --batch-size")
        batch_size = 1 if args.batch_size is None else args.batch_size
        if batch_size <= 0:
            parser.error("--batch-size must be positive")
        bundle = args.output_root.expanduser().resolve() / args.run_id / "population"
        manifest = run_population(
            args.checkpoint, bundle, args.run_id, num_agents=args.num_agents,
            seeds=parse_int_range(args.seeds), horizon=args.steps, bound=args.bound,
            policies=args.policies, device=args.device, batch_size=batch_size,
            workers=args.workers, repair_invalid=args.repair_invalid,
            dry_run=args.dry_run)
        print(json.dumps({"bundle": str(bundle), "manifest": manifest}, indent=2,
                         sort_keys=True, allow_nan=False))
        return 0

    if args.command == "validate":
        from eval.population import validate_bundle
        result = validate_bundle(
            args.run,
            deep=args.deep,
            write_report=args.write_report,
            rebuild_summaries=args.rebuild_summaries,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.command == "radii":
        from eval.analysis.radii import run_from_bundle
        outputs = run_from_bundle(args.run, args.n, args.seed, output=args.output,
                                  animation=args.animation)
        print(json.dumps(outputs, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.command == "heatmaps":
        from eval.analysis.heatmaps import run_from_bundle
        outputs = run_from_bundle(args.run, output=args.output,
                                  t_max_seconds=args.t_max_seconds,
                                  with_entropies=args.with_entropies)
        print(json.dumps(outputs, indent=2, sort_keys=True, allow_nan=False))
        return 0

    if args.command == "control-effort":
        from eval.analysis.control_effort import run_from_bundle
        outputs = run_from_bundle(args.run, output=args.output,
                                  animations=args.animations)
        print(json.dumps(outputs, indent=2, sort_keys=True, allow_nan=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
