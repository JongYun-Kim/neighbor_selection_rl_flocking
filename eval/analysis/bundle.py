"""Small read-only helpers shared by bundle-facing analysis adapters."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _paths_from_csv(path: Path, root: Path) -> List[Path]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        path_column = next(
            (
                name
                for name in ("path", "episode_path", "artifact_path", "raw_path")
                if name in names
            ),
            None,
        )
        if path_column is None:
            raise ValueError("episodes CSV has no artifact path column: {}".format(path))
        paths = []
        for row in reader:
            item = Path(row[path_column])
            paths.append(item if item.is_absolute() else root / item)
        return paths


def discover_bundle(
    run: os.PathLike,
) -> Tuple[Path, Dict[str, Any], List[Path]]:
    """Return bundle root, manifest, and referenced full-episode files."""

    supplied = Path(run).expanduser().resolve()
    manifest_path = supplied if supplied.is_file() else supplied / "manifest.json"
    root = manifest_path.parent if manifest_path.is_file() else supplied
    if not root.is_dir():
        raise FileNotFoundError("bundle does not exist: {}".format(root))
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        status = manifest.get("status")
        if status is not None and status != "completed":
            raise ValueError(
                "bundle status is {!r}, not 'completed': {}".format(status, root)
            )

    csv_path = next(
        (
            candidate
            for candidate in (
                root / "episodes.csv",
                root / "summaries" / "episodes.csv",
            )
            if candidate.is_file()
        ),
        None,
    )
    if csv_path is not None:
        paths = _paths_from_csv(csv_path, root)
    else:
        episode_root = root / "episodes"
        search_root = episode_root if episode_root.is_dir() else root
        paths = sorted(search_root.rglob("*.npz"))

    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise FileNotFoundError("no episode NPZ files found under {}".format(root))
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "bundle references missing episode: {}".format(missing[0])
        )
    return root, manifest, unique


def is_canonical_population_bundle(manifest: Mapping[str, Any]) -> bool:
    """Return whether default in-bundle analysis output is safe to use."""
    return (
        manifest.get("schema_version") == "evaluation-bundle-2.0"
        and manifest.get("suite") == "population_v1"
        and manifest.get("status") == "completed"
    )


def resolve_analysis_output(
    root: Path,
    manifest: Mapping[str, Any],
    output: Optional[os.PathLike],
    *default_parts: str,
) -> Path:
    """Resolve an analysis directory without writing into legacy records.

    Maintained population bundles own their ``analysis/`` directory.  Legacy
    farms and ad-hoc episode collections are historical inputs, so they require
    an explicit destination outside their source root.
    """
    root = Path(root).resolve()
    if output is None:
        if not is_canonical_population_bundle(manifest):
            raise ValueError(
                "legacy/noncanonical analysis input requires an explicit "
                "analysis output directory outside the source bundle"
            )
        return root.joinpath(*default_parts)

    resolved = Path(output).expanduser().resolve()
    if not is_canonical_population_bundle(manifest):
        try:
            resolved.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "legacy/noncanonical analysis output must be outside the "
                "source bundle: {}".format(root)
            )
    return resolved


def deterministic_paths(paths: Sequence[Path]) -> List[Path]:
    """Use canonical policy directory names to avoid loading other populations.

    Historical bundles without a policy directory are returned unchanged and
    subsequently filtered from their normalized metadata.
    """

    # The policy is the directory immediately containing an episode file in
    # both maintained (``deterministic``) and dev (``best_deterministic``)
    # layouts.  Never scan absolute ancestors: a run ID such as
    # ``deterministic-ablation`` must not make its stochastic children pass.
    deterministic_segments = {"deterministic", "best_deterministic"}
    explicit = [
        path
        for path in paths
        if path.parent.name.strip().lower() in deterministic_segments
    ]
    return explicit if explicit else list(paths)


def _walk_named(value: Any, names: set) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names and not isinstance(child, (Mapping, list, tuple)):
                yield child
            yield from _walk_named(child, names)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_named(child, names)


def manifest_number(
    manifest: Mapping[str, Any],
    names: Sequence[str],
    *,
    num_agents: Optional[int] = None,
) -> Optional[float]:
    """Find one unambiguous numeric value, including legacy per-N configs."""

    evaluation = manifest.get("evaluation", {})
    configs = evaluation.get("config_by_num_agents", {}) if isinstance(
        evaluation, Mapping
    ) else {}
    if num_agents is not None and isinstance(configs, Mapping):
        selected = configs.get(str(int(num_agents)), configs.get(int(num_agents)))
        candidates = list(_walk_named(selected, set(names)))
        if candidates:
            return float(candidates[0])

    candidates = list(_walk_named(manifest, set(names)))
    numeric = []
    for value in candidates:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    first = numeric[0]
    if not all(value == first for value in numeric[1:]):
        return None
    return first


def expected_episode_count(manifest: Mapping[str, Any]) -> Optional[int]:
    spec = manifest.get("spec", {})
    seeds = spec.get("seeds") if isinstance(spec, Mapping) else None
    return len(seeds) if isinstance(seeds, list) and seeds else None


def manifest_agent_counts(manifest: Mapping[str, Any]) -> Tuple[int, ...]:
    spec = manifest.get("spec", {})
    values = spec.get("num_agents") if isinstance(spec, Mapping) else None
    if not isinstance(values, list):
        return tuple()
    return tuple(int(value) for value in values)
