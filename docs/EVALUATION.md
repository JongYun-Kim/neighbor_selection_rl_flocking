# Evaluation workflow

The repository-root `eval` package is the implementation of record. The unified
entry point is `python -m eval <command> ...`; the older per-module commands in
the README remain available for reproducibility. New checkpoint selection,
staged C2 evaluation, population recording, validation, and analysis should use
the unified entry point.

## `main_c2_v1`: criterion of record

An episode first fires C2 at state-boundary index `t` only when all three
conditions hold:

1. The minimum polarization over the last 50 state samples is **strictly
   greater than** `0.98`.
2. The proximity graph is connected throughout the last 300 state samples.
   A proximity edge exists only when distance is **strictly less than** `r0`.
3. Over those 300 samples, position entropy satisfies
   `(rolling_max - rolling_min) / rolling_mean < 0.05`, again a **strict**
   inequality.

The initial state is index 0 and post-step states are indices 1 through 6000.
The evaluator runs the entire fixed horizon and judges convergence offline; a
short trajectory is not evidence of failure. Evaluation uses
`is_training=False`, so reward is the unshaped control cost. For a successful
episode, `J = -sum(reward[1:t+1])`; failed episodes keep `J` undefined. Learned
policies use deterministic argmax for the criterion-of-record lane.

The only official confirmation configuration is:

| Field | Value |
|---|---|
| lane | `confirm` |
| agents | `N=20` |
| initial-position bound | `L=250` |
| seeds | `1500-1999` (500 episodes) |
| horizon | 6000 steps |
| inference | deterministic argmax, CPU |
| termination and judgment | fixed horizon, offline `main_c2_v1` |

Changing any of those fields produces a development or diagnostic result, not
a confirmation result. The `dev` lane defaults to seeds `1000-1031`, uses the
same protocol, and is explicitly non-official. Checkpoint-based commands default
to `checkpoints/dynamic_k_nn_best/checkpoint_000592`; pass `--checkpoint` to
override it. Confirmation still accepts exactly one checkpoint, and never a
candidate set.

## Checkpoint funnel

Start from one unambiguous Tune trial:

```bash
python -m eval checkpoints --trial-dir test_results/<trial> \
    --top 5 --include-final --out test_results/evaluation/candidates.json
```

`--run-dir` may be used when it contains exactly one trial. Discovery fails if
there are zero or multiple `progress.csv` files rather than guessing. The
selector intersects progress rows with checkpoint entries that actually exist,
ranks finite in-training metrics by:

1. `evaluation/custom_metrics/c2_success_mean`, descending;
2. `evaluation/custom_metrics/t_conv_mean`, ascending;
3. `evaluation/custom_metrics/J_success_mean`, ascending;
4. training iteration, descending.

It retains the top five and adds the latest/final matched checkpoint. If that
checkpoint is already in the top five it is de-duplicated and receives both
roles. The JSON contains resolved source paths, metrics, iteration, trial,
checkpoint-tree SHA-256, recognizable state-file SHA-256 values, the sibling
`params.json` hash, and a combined inference-package hash. It is a manifest
only: checkpoint data is never copied at this stage. Those hashes are
revalidated before a lane starts, and the confirmation bundle atomically
archives its one finalist.

Evaluate all candidates on `dev` and review `summaries/ranking.csv`. The canonical
training-curve-best checkpoint can then be confirmed without a checkpoint option;
pass `--checkpoint` when confirming a different selection:

```bash
python -m eval c2 --lane dev --candidates test_results/evaluation/candidates.json \
    --run-id model-dev --output-root test_results/evaluation --workers 24

python -m eval c2 --lane confirm \
    --run-id model-confirm --output-root test_results/evaluation --workers 24
```

Lane defaults should normally be left intact. `--seeds` and `--steps` exist for
diagnostics, but overriding them makes a run non-official.

## Full population suite

The default `population_v1` suite records the complete trajectories needed for
cutoff-radius, heatmap, entropy, and control-effort analysis:

| Axis | Default |
|---|---|
| population size | `N=10,20,40` |
| initial-position bound | `L=250` |
| environment seeds | `0-49` |
| horizon | 6000 steps |
| policies | `deterministic,stochastic,pure_acs` |
| backend | CPU, one episode per inference batch |

Every episode stores `T+1` states and scalar state series, plus `T` pointer
actions, realized binary masks, and control inputs. Deterministic and stochastic
learned policies and PureACS share the same initial state for each `(N, seed)`;
stochastic action seeds are derived and recorded separately. Each episode is
judged with `main_c2_v1`, but the population suite itself is an analysis suite,
not the official N=20 confirmation lane.

Summaries include per-arm Wilson/CVaR results, raw seed-paired deltas to
PureACS, and aggregate exact McNemar plus co-success paired-dJ
(Wilcoxon/sign-test) statistics.

```bash
python -m eval population --run-id model-population \
    --output-root test_results/evaluation \
    --num-agents 10,20,40 --seeds 0-49 --steps 6000 --bound 250 \
    --policies deterministic,stochastic,pure_acs --device cpu --workers 8
```

CPU results are the canonical production results. A CUDA run may be useful for
throughput diagnosis and batching experiments, but it must have a distinct run
ID and be labelled diagnostic; do not combine it with or silently substitute it
for the CPU population. Use `--device cuda:N --batch-size B` only for that lane.

## Configurable sensitivity suite

The sensitivity suite runs one-factor-at-a-time diagnostics from a versioned
YAML specification. The maintained presets cover the standard OAT study and the
small-turn-radius, time-refined study:

```bash
python -m eval sensitivity run \
    --config eval/sensitivity/configs/oat.yaml \
    --run-id dknn-oat --output-root test_results/evaluation

python -m eval sensitivity run \
    --config eval/sensitivity/configs/refined_near_zero.yaml \
    --run-id dknn-near-zero --output-root test_results/evaluation

python -m eval sensitivity validate \
    --run test_results/evaluation/dknn-oat/sensitivity
python -m eval sensitivity plot \
    --run test_results/evaluation/dknn-oat/sensitivity
```

The config is the source of truth for the baseline, factor values, policies,
seeds, and temporal profile. `--dry-run` previews the expanded settings and job
count. `--factors`, `--settings`, `--policies`, and `--seeds` select part of that
plan for a distinct filtered bundle, or repeat the identical selection when
resuming it; `--checkpoint`, `--device`,
`--batch-size`, `--workers`, and `--repair-invalid` are explicit execution
overrides. Filters are part of the bundle fingerprint, so separately filtered
shards need distinct run IDs and are not silently merged. Validate a completed
bundle before plotting it. Plotting requires `pure_acs` and at least one learned
policy evaluated on the same seeds. A filtered OAT plan records the factors
represented by its non-baseline `--settings` (a baseline-only plan retains the
requested `--factors`). Plots use that resolved list while the one shared
baseline setting keeps its canonical membership in all five OAT factors.

Both presets run `learned_deterministic`, `learned_stochastic`, and `pure_acs`
for seeds 0--99 and 6000 policy steps. Their explicit baseline is N=20,
speed=15, minimum turn radius=28.125, interaction/C2 radius=60, ACS
lambda/sigma=5/1, and initial bound=250. The OAT ranges are N=`10,20,40,80`;
turn radius=`14.0625,21.09375,28.125,42.1875,56.25`; interaction radius=
`30,45,60,90,120`; joint ACS gain multiplier=`0.5,0.75,1,1.5,2`; and initial
bound=`125,187.5,250,375,500`. The shared baseline is executed only once, for
20 unique OAT settings in total.

The refined preset tests turn radii
`14.0625,7.03125,3.515625,1.7578125,0.87890625,0.477464829275686`. It computes
`ceil(max_turn_rate * 0.1 / max_heading_change_per_substep)` dynamics steps per
policy interval, giving `2,4,8,16,32,59` for the checked-in limit. A full OAT
run is 36 million environment steps; the refined preset is 217.8 million
dynamics substeps, so use `--dry-run` and filtered smoke runs before launching
the complete jobs.

Sensitivity episodes use compact artifacts containing the scalar series,
outcome, and metadata needed for C2 summaries and paired comparisons. They do
not record full state, action, or control trajectories; use the population suite
when radius, heatmap, or control-effort analysis is required. The sensitivity
bundle writes `settings.csv`, `episodes.csv`, `aggregate.csv`,
`paired_vs_acs.csv`, and `dominance.csv` under `summaries/`.

All maintained sensitivity settings are judged offline with `main_c2_v1`, but
they are diagnostic results, not the official confirmation lane. In the refined
profile the policy still acts at 10 Hz: each 0.1-second policy interval is split
into smaller dynamics steps, each learned pointer action is held for that
interval, and its realized mask and ACS control are recomputed at every substep.
PureACS also recomputes its farthest-neighbor pointer at every substep.

The presets carry forward the experimental questions from the historical
sensitivity farm, not a claim that its published numbers are reproduced. The
current protocol, checkpoint, implementation, and sparse artifact contract are
authoritative for new runs; historical farm outputs remain read-only records.

## Artifacts, resume, and validation

Each C2, population, or sensitivity run owns one output bundle with a versioned
`manifest.json`, a fingerprint of its immutable specification, episode files,
summaries, and validation records. A rerun may reuse an episode only after its
shape and metadata match the manifest. A mismatched fingerprint fails instead
of appending incompatible data to an existing run. `--repair-invalid` is an
explicit recovery action: invalid files are moved under the bundle's
`quarantine/` tree before regeneration. File existence alone is never a resume
criterion.

The unified CLI lays bundles out as
`<output-root>/<run-id>/c2_dev`, `c2_confirm`, `population`, or `sensitivity`.
Pass the suite directory itself to its validation and analysis commands.

Atomic artifact writes honor the invoking process's umask and directory ACLs
(for example, umask `0002` produces group-readable/writable files and
directories). Replacing an existing file preserves its current mode. This
keeps shared result roots collaborative without weakening a deliberately
restrictive umask.

Validate a completed population before analysis; use `--deep` to recompute the
pointer-to-mask relationship, physical metric series, state/control
transitions, raw rewards, all persisted C2 result copies, and the archived
checkpoint package. Validation is read-only by default and verifies the
existing summary CSVs. `--write-report` explicitly records `validation.json`;
`--rebuild-summaries` is the explicit repair operation and also records that
report:

```bash
python -m eval validate \
    --run test_results/evaluation/model-population/population \
    --deep --write-report
```

New artifacts are written only under the designated artifact root. Source and
checkpoints are inputs and must remain read-only. Historical farm and study
outputs are also read-only records: compatibility readers may load their old
field names for analysis, but they do not rewrite or migrate files in place,
relabel an old protocol as `main_c2_v1`, or treat a legacy short horizon as a
valid 6000-step failure observation. When reading a legacy bundle, always pass
an explicit analysis `--output` under the new artifact root rather than writing
derived files beside the historical record. The full-trajectory population
analysis commands enforce this: a legacy/noncanonical input is rejected when
the output is omitted or resolves inside the source bundle.

## One-shot Docker runner

`docker/run_eval.sh` runs the same unified CLI in the foreground and removes
the container on exit. It mounts:

- the current source worktree at `/workspace/source`, read-only;
- `CHECKPOINT_ROOT` at `/workspace/checkpoints`, read-only;
- `ARTIFACT_ROOT` at `/workspace/artifacts`, read-write.

The defaults are the repository's `checkpoints/` and
`test_results/evaluation/` directories. The checkpoint root must already exist;
the artifact directory is created only for a real run. Paths passed to the eval
CLI must use the container names above, not host-absolute paths. Keep the same
`CHECKPOINT_ROOT` mapping between checkpoint selection and later evaluation,
because the selection JSON intentionally records resolved paths.

The image defaults to `uom-neighbor-selection-eval`, a separate tag from the
durable training runner's `uom-neighbor-selection`; therefore an evaluation
`--build` cannot retag the image another user's future training container will
start. Set `IMAGE_NAME` explicitly only when intentionally sharing a tag. No
GPU is exposed by default.
`DEVICE=cuda:0`, `GPU_REQUEST=...`, or an evaluator `--device cuda:0` explicitly
requests Docker GPU access; `DEVICE` is also the population and sensitivity CLI
default when `--device` is omitted. CUDA population and sensitivity runs require
an explicit `--batch-size`. The runner uses a unique container name, performs a
foreground one-shot `docker run --rm`, and forwards host-side git commit/dirty
provenance—including tracked diffs and untracked, non-ignored files—because a
linked worktree's host `.git` pointer is unavailable in the container.

Runner options must precede the eval command:

```bash
# Preview Docker itself; no directory, image, or container is changed.
./docker/run_eval.sh --build --dry-run checkpoints --trial-dir /workspace/checkpoints

# Build, then run. Everything after "checkpoints" belongs to python -m eval.
./docker/run_eval.sh --build checkpoints --trial-dir /workspace/checkpoints \
    --out /workspace/artifacts/candidates.json

# Here --dry-run belongs to the population CLI, not to the Docker runner.
./docker/run_eval.sh population --checkpoint /workspace/checkpoints/checkpoint_000848 \
    --run-id population-preview --output-root /workspace/artifacts --dry-run
```

## Copy-ready next-session workflow

Set the two host roots once. `CHECKPOINT_ROOT` may be a Tune run root or a
higher directory; adjust the `/workspace/checkpoints/...` suffix accordingly.

```bash
export CHECKPOINT_ROOT=/absolute/path/to/training-results
export ARTIFACT_ROOT=/absolute/path/to/evaluation-artifacts

# 1. Check the fully resolved container invocation, then create top-5+final JSON.
./docker/run_eval.sh --build --dry-run checkpoints \
    --run-dir /workspace/checkpoints/<run> \
    --out /workspace/artifacts/funnel/candidates.json
./docker/run_eval.sh --build checkpoints \
    --run-dir /workspace/checkpoints/<run> \
    --out /workspace/artifacts/funnel/candidates.json

# 2. Screen every candidate on the 32-seed development lane.
./docker/run_eval.sh c2 --lane dev \
    --candidates /workspace/artifacts/funnel/candidates.json \
    --run-id chosen-model-dev --output-root /workspace/artifacts --workers 24

# 3. Review chosen-model-dev/c2_dev/summaries/ranking.csv, then confirm one checkpoint.
./docker/run_eval.sh c2 --lane confirm \
    --checkpoint /workspace/checkpoints/<run>/<trial>/checkpoint_<iteration> \
    --run-id chosen-model-confirm --output-root /workspace/artifacts --workers 24

# 4. Record the complete N=10/20/40 population on CPU.
./docker/run_eval.sh population \
    --checkpoint /workspace/checkpoints/<run>/<trial>/checkpoint_<iteration> \
    --run-id chosen-model-population --output-root /workspace/artifacts \
    --num-agents 10,20,40 --seeds 0-49 --steps 6000 --bound 250 \
    --policies deterministic,stochastic,pure_acs --device cpu --workers 8

# 5. Deep-validate before producing any derived result.
./docker/run_eval.sh validate \
    --run /workspace/artifacts/chosen-model-population/population \
    --deep --write-report

# 6. Generate per-episode radii, 30-second population heatmaps with entropy,
#    and optional control-effort figures/animations from the validated bundle.
./docker/run_eval.sh radii \
    --run /workspace/artifacts/chosen-model-population/population --n 20 --seed 0
./docker/run_eval.sh heatmaps \
    --run /workspace/artifacts/chosen-model-population/population \
    --t-max-seconds 30 --with-entropies
./docker/run_eval.sh control-effort \
    --run /workspace/artifacts/chosen-model-population/population --animations
```

Omit `--t-max-seconds` for full-horizon heatmaps. Analysis commands consume the
recorded artifacts; they do not rerun the policy or alter the source bundle's
episode data. With the default output location, full, time-limited, and
time-limited-plus-entropy heatmaps use separate view directories under
`analysis/ranked_heatmaps/`, so all three four-figure sets can coexist.
