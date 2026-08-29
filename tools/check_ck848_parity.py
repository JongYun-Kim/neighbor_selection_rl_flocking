"""Config equivalence gate: `train_unified.py --profile dknn` == the ck848 recipe.

ck848 (checkpoints/legacy_distance_pointer_260818/.../checkpoint_000848) is the
best-performing policy this repo has produced and the target of the
reproduction line. Its exact training config is preserved next to it as
params.json; this script diffs a resolved unified-trainer config against it and
fails on any difference that is not on the whitelist below.

Usage:
  python train_unified.py --profile dknn --dry-run | python tools/check_ck848_parity.py
  python tools/check_ck848_parity.py --config resolved.json [--ref other/params.json]

Input is either the --dry-run payload ({"config": ..., "stop": ...}) or a bare
RLlib config dict. Exit 0 = equivalent, 1 = a field diverged (each printed).

The whitelist has three parts, all justified in REPO_CONSOLIDATION_PLAN.md 4.1:
  ALIASES          identifier renames that leave the code path identical
  NEW_TOPLEVEL     RLlib keys ck848 did not set, with the value each must hold
  ENV_NEW_EXPECTED env-schema fields added after ck848, each pinned to the
                   legacy default that reproduces ck848 behavior
Not visible in a config and therefore not checked here: the trainable class
(ck848 ran PPO, the unified trainer runs GradLoggingPPO, which only adds
gradient-norm logging) and the callbacks' no-op status on this path.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.paths import repo_path  # noqa: E402

DEFAULT_REF = repo_path(
    "checkpoints", "legacy_distance_pointer_260818",
    "PPO_neighbor_selection_flocking_env_8d07b_00000_0_2026-08-18_06-10-46",
    "params.json")

# Pre/post-rename identifier pairs. The Dynamic-k NN rename touched names only:
# the pointer -> mask rule and the model's parameter set are byte-identical, and
# ck848 loads strict=True under the new model class (LEGACY848 section 1).
ALIASES = [
    {"distance_pointer_neighbor_selector_rl", "dynamic_k_nn_neighbor_selector_rl"},
    {"distance_pointer", "dynamic_k_nn"},
]

# RLlib top-level keys absent from ck848's config, with the value each is
# allowed to take. A callable receives the value and returns True when the key
# is still equivalent to ck848's behavior.
NEW_TOPLEVEL = {
    "callbacks": (lambda v: isinstance(v, str) and v.endswith("C2Callbacks"),
                  "metrics logging only; no-op on the dknn path apart from "
                  "copying per_agent_rewards into the batch, which no loss reads"),
    "entropy_coeff": (lambda v: v == 0.0,
                      "explicit 0.0 == the RLlib default ck848 inherited"),
    "normalize_actions": (lambda v: v is False,
                          "inert for MultiDiscrete actions in either setting"),
    "evaluation_interval": (lambda v: v is None or (isinstance(v, int) and v > 0),
                            "in-training eval cadence; runs on separate workers"),
    "evaluation_duration": (lambda v: isinstance(v, int) and v > 0,
                            "episodes per eval round; off the training path"),
    "evaluation_duration_unit": (lambda v: v == "episodes",
                                 "C2 protocol counts episodes, not steps"),
    "evaluation_num_workers": (lambda v: isinstance(v, int) and v >= 0,
                               "dedicated eval workers; no training rollouts"),
    "evaluation_config": (
        lambda v: (isinstance(v, dict) and v.get("explore") is False
                   and v["env_config"]["config"]["env"]["is_training"] is False),
        "separate eval env: argmax actions, is_training=False"),
}

# Env-schema fields introduced after ck848. Each must hold the legacy default,
# i.e. the value at which the field does not exist as far as behavior goes.
ENV_NEW_EXPECTED = {
    "termination_mode": "legacy",
    "reward_mode": "legacy",
    "obs_position_scale": "legacy",
    "initial_position_bound_pool": None,
    "expose_aux_target": False,
    "expose_global_stats": False,
    "continuous_action": False,
    "acs_train_w_conn": 0.0,
    "acs_train_w_align": 0.0,
    # C2 constants, inert while termination_mode/reward_mode are "legacy".
    "c2_phi_goal": 0.98,
    "c2_align_window": 50,
    "c2_window": 300,
    "c2_eps": 0.05,
    "c2_w_pos": 4.0,
    "c2_w_vel": 0.2,
    "c2_w_ctrl": 0.1,
    "c2_success_bonus": 10.0,
}

EXPECTED_BUDGET = 8_000_000

FAILS = []


def fail(msg):
    FAILS.append(msg)


def same(a, b):
    """Equality with the rename aliases and list/tuple normalization."""
    if isinstance(a, str) and isinstance(b, str) and a != b:
        return any({a, b} <= al for al in ALIASES)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def check_toplevel(new, ref):
    """(1) every ck848 key reproduced, (2) every extra key whitelisted."""
    for k in sorted(ref):
        if k in ("env_config", "model"):
            continue                      # walked separately
        if k not in new:
            fail(f"top-level: missing key {k!r} (ck848 had {ref[k]!r})")
        elif not same(new[k], ref[k]):
            fail(f"top-level {k}: ck848 {ref[k]!r} != new {new[k]!r}")
    for k in sorted(set(new) - set(ref)):
        if k not in NEW_TOPLEVEL:
            fail(f"top-level {k}: not on the whitelist (value {new[k]!r})")
            continue
        cond, _why = NEW_TOPLEVEL[k]
        try:
            ok = cond(new[k])
        except Exception as exc:                       # malformed value
            ok, _why = False, f"{_why} [raised {exc!r}]"
        if not ok:
            fail(f"top-level {k}: value {new[k]!r} violates the whitelist "
                 f"condition ({_why})")


def check_model(new, ref):
    nm, rm = new.get("model", {}), ref["model"]
    if not same(nm.get("custom_model"), rm["custom_model"]):
        fail(f"model.custom_model: ck848 {rm['custom_model']!r} != "
             f"new {nm.get('custom_model')!r}")
    nc = nm.get("custom_model_config", {})
    rc = rm["custom_model_config"]
    for k in sorted(rc):
        if k not in nc:
            fail(f"model.custom_model_config: missing {k!r} (ck848 {rc[k]!r})")
        elif not same(nc[k], rc[k]):
            fail(f"model.custom_model_config.{k}: ck848 {rc[k]!r} != new {nc[k]!r}")
    for k in sorted(set(nc) - set(rc)):
        fail(f"model.custom_model_config.{k}: absent from ck848 "
             f"(new {nc[k]!r}) — the dknn model config must stay verbatim")


def check_env_config(new, ref):
    """(3) env dict: ck848 keys identical, new keys at their legacy defaults."""
    ne, re_ = new.get("env_config", {}), ref["env_config"]
    if not same(ne.get("seed_id"), re_["seed_id"]):
        fail(f"env_config.seed_id: ck848 {re_['seed_id']!r} != "
             f"new {ne.get('seed_id')!r}")
    nc = ne.get("config", {})
    rc = re_["config"]
    for section in ("control", "env"):
        n_s, r_s = nc.get(section, {}), rc[section]
        for k in sorted(r_s):
            if k not in n_s:
                fail(f"env_config.{section}: missing {k!r} (ck848 {r_s[k]!r})")
            elif not same(n_s[k], r_s[k]):
                fail(f"env_config.{section}.{k}: ck848 {r_s[k]!r} != new {n_s[k]!r}")
        extra = sorted(set(n_s) - set(r_s))
        if section == "control":
            for k in extra:
                fail(f"env_config.control.{k}: absent from ck848 (new {n_s[k]!r})")
            continue
        for k in extra:
            if k not in ENV_NEW_EXPECTED:
                fail(f"env_config.env.{k}: post-ck848 field with no expected "
                     f"legacy default recorded (new {n_s[k]!r})")
            elif n_s[k] != ENV_NEW_EXPECTED[k]:
                fail(f"env_config.env.{k}: expected the legacy default "
                     f"{ENV_NEW_EXPECTED[k]!r}, got {n_s[k]!r}")
        for k in sorted(set(ENV_NEW_EXPECTED) - set(n_s)):
            fail(f"env_config.env: expected post-ck848 field {k!r} to be present "
                 f"and set to {ENV_NEW_EXPECTED[k]!r}")


def check_stop(stop):
    got = stop.get("timesteps_total")
    if got != EXPECTED_BUDGET:
        fail(f"stop.timesteps_total: the ck848 recipe is {EXPECTED_BUDGET} steps, "
             f"got {got!r}")
    if "training_iteration" in stop:
        fail(f"stop.training_iteration: ck848 stopped on steps only, got "
             f"{stop['training_iteration']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="resolved config JSON (default: stdin)")
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help="ck848 params.json (default: the in-repo copy)")
    ap.add_argument("--skip-stop", action="store_true",
                    help="do not check the stop condition (for partial-budget "
                         "configs that are otherwise the ck848 recipe)")
    args = ap.parse_args()

    raw = (open(args.config).read() if args.config else sys.stdin.read()).strip()
    if not raw:
        raise SystemExit("no config on stdin; pipe `train_unified.py --dry-run`")
    payload = json.loads(raw)
    new = payload.get("config", payload)
    stop = payload.get("stop")
    with open(args.ref) as fh:
        ref = json.load(fh)

    check_toplevel(new, ref)
    check_model(new, ref)
    check_env_config(new, ref)
    if stop is not None and not args.skip_stop:
        check_stop(stop)

    ref_short = os.path.relpath(args.ref, repo_path())
    if FAILS:
        print(f"ck848 parity FAIL ({len(FAILS)} diffs) vs {ref_short}")
        for msg in FAILS:
            print(f"  - {msg}")
        return 1
    n_env = len(ref["env_config"]["config"]["env"])
    print(f"ck848 parity PASS vs {ref_short}")
    print(f"  top-level: {len(ref) - 2} keys reproduced, "
          f"{len(set(new) - set(ref))} whitelisted additions")
    print(f"  env: {n_env} ck848 fields identical, "
          f"{len(ENV_NEW_EXPECTED)} post-ck848 fields at their legacy defaults")
    print(f"  model: {len(ref['model']['custom_model_config'])} custom_model_config "
          f"fields identical; identifiers matched through the rename aliases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
