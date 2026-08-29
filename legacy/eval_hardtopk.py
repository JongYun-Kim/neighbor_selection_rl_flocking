"""
FAST PROBE: Hard top-K=10 selection from supervised attention scores vs Fully-Connected ACS.

Hypothesis: checkpoint `distaux_v2_rank10_260527` was trained with dist_aux_coef=1.0 so that
attention scores rank distance (nearest neighbors get the highest score). Applying a HARD
top-K=10 selection on those attention scores should reproduce KNN-like behaviour and BEAT the
fully-connected (FC) ACS baseline (which KNN beats by ~17%).

This script REUSES evaluate_checkpoint.py machinery (create_env, RLPolicy, PureACSPolicy,
monte_carlo_evaluation, print_comparison). It runs the env in BINARY mode (continuous_action=False).
"""

import os
# ---- GPU PIN: use ONLY GPU 1, set before importing torch/ray ----
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import pickle
import argparse
import numpy as np
import torch
from scipy import stats

from legacy.evaluate_checkpoint import (
    create_env,
    RLPolicy,
    PureACSPolicy,
    monte_carlo_evaluation,
    print_comparison,
)
from models.ppo import NeighborSelectionPPORLlib


CHECKPOINT = (
    "/workspace/test_results/distaux_v2_rank10_260527/"
    "GradLoggingPPO_ns_env_f6fbe_00000_0_2026-05-27_11-34-04/checkpoint_000100"
)
TOP_K = 10


def _build_binary_action_from_scores(scores, padding_mask, neighbor_masks, k):
    """
    Given per-agent neighbor scores (N, N) where HIGH = preferred neighbor, build a binary
    int8 action that selects the top-k highest-scoring VALID neighbors per agent + self-loop.

    Respects padding_mask (1-D, N) and neighbor_masks (2-D, N, N). Invalid entries are zeroed.

    Returns action (N, N) int8.
    """
    n = padding_mask.shape[0]
    action = np.zeros((n, n), dtype=np.int8)

    # valid[i, j] = both real AND visible. (Eval uses FC visibility + no padding, but be safe.)
    pad2d = padding_mask[:, None].astype(bool) & padding_mask[None, :].astype(bool)
    valid = pad2d & neighbor_masks.astype(bool)

    # Work on a copy of scores; force invalid + diagonal to -inf so they are never top-k picked.
    s = scores.astype(np.float64).copy()
    s[~valid] = -np.inf
    np.fill_diagonal(s, -np.inf)

    active_idx = np.where(padding_mask.astype(bool))[0]
    for i in active_idx:
        row = s[i]
        # candidate neighbors = finite-scored (valid, non-self) entries
        cand = np.where(np.isfinite(row))[0]
        if cand.size == 0:
            continue
        if cand.size <= k:
            chosen = cand
        else:
            # top-k highest scores among candidates
            chosen = cand[np.argpartition(row[cand], -k)[-k:]]
        action[i, chosen] = 1

    # self-loops for real agents
    action[active_idx, active_idx] = 1
    return action


class HardTopKPolicy:
    """
    Loads the model EXACTLY like RLPolicy, but in __call__ extracts raw attention scores from
    model.actor(obs_dict) and applies a hard top-K=10 selection (highest attention = nearest,
    per the dist_aux supervision) to produce a binary adjacency action.
    """

    def __init__(self, checkpoint_path, env, k=TOP_K):
        self.k = k
        # Reuse RLPolicy's loading machinery verbatim (params.json read, model build, weight restore).
        self._rl = RLPolicy(checkpoint_path, env, observation_type="ego_centric")
        self.model = self._rl.model
        self.device = self._rl.device

    def _build_obs_tensors(self, obs):
        """Replicate RLPolicy.__call__ obs preprocessing into the model's obs_dict."""
        obs_tensors = {
            "local_agent_infos": torch.from_numpy(
                obs["local_agent_infos"][np.newaxis]
            ).float().to(self.device),
            "neighbor_masks": torch.from_numpy(
                obs["neighbor_masks"][np.newaxis]
            ).float().to(self.device),
            "padding_mask": torch.from_numpy(
                obs["padding_mask"][np.newaxis]
            ).float().to(self.device),
        }
        if "global_agent_infos" in obs:
            obs_tensors["global_agent_infos"] = torch.from_numpy(
                obs["global_agent_infos"][np.newaxis]
            ).float().to(self.device)
        if "is_from_my_env" in obs:
            obs_tensors["is_from_my_env"] = torch.from_numpy(
                np.array([obs["is_from_my_env"]])
            ).float().to(self.device)
        return obs_tensors

    def __call__(self, obs):
        with torch.no_grad():
            obs_dict = self._build_obs_tensors(obs)
            att, *_ = self.model.actor(obs_dict)  # (B, N, N) raw attention; HIGH = near
            att = att.cpu().numpy()[0]  # (N, N)

        return _build_binary_action_from_scores(
            scores=att,
            padding_mask=obs["padding_mask"],
            neighbor_masks=obs["neighbor_masks"],
            k=self.k,
        )


class KNNTruePolicy:
    """
    Reference upper bound: select the k nearest neighbors by ACTUAL relative position.
    local_agent_infos[i, j, :2] = ego-rotated relative position of j as seen by i (normalized).
    Its Euclidean norm equals the true (normalized) relative distance (rotation invariant).
    Smaller distance = preferred => use NEGATIVE distance as the score for the shared top-k helper.
    """

    def __init__(self, k=TOP_K):
        self.k = k

    def __call__(self, obs):
        local = obs["local_agent_infos"]  # (N, N, obs_dim)
        rel_xy = local[:, :, :2]  # (N, N, 2) normalized ego-frame relative position
        dist = np.sqrt(np.sum(rel_xy ** 2, axis=-1))  # (N, N) relative distance, rotation-invariant
        score = -dist  # nearer => higher score
        return _build_binary_action_from_scores(
            scores=score,
            padding_mask=obs["padding_mask"],
            neighbor_masks=obs["neighbor_masks"],
            k=self.k,
        )


def _welch(rl_stats, fc_stats):
    """Welch's t-test on per-episode returns (rl vs fc)."""
    rl_ret = np.asarray(rl_stats["_episode_returns"], dtype=np.float64)
    fc_ret = np.asarray(fc_stats["_episode_returns"], dtype=np.float64)
    t, p = stats.ttest_ind(rl_ret, fc_ret, equal_var=False)
    return float(t), float(p)


def _summary(name, stats_dict, fc_stats=None):
    rew_m = stats_dict["episode_return_mean"]
    rew_s = stats_dict["episode_return_std"]
    vel = stats_dict.get("velocity_entropy_final_mean", np.nan)
    edges = stats_dict.get("mean_edges_per_agent_mean", np.nan)
    line = (f"{name:<16} reward={rew_m:8.3f} ± {rew_s:6.3f} | "
            f"vel_ent(final)={vel:7.4f} | edges/agent={edges:6.3f}")
    if fc_stats is not None:
        t, p = _welch(stats_dict, fc_stats)
        sig = "YES" if (p < 0.05 and rew_m > fc_stats["episode_return_mean"]) else "no"
        line += f" | t={t:7.3f} p={p:.4g} | beats_FC(p<.05)={sig}"
    print(line)
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen_episodes", type=int, default=150)
    parser.add_argument("--final_episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"torch.cuda.is_available()={torch.cuda.is_available()}")

    # Confirm continuous_action of the checkpoint (informational) but FORCE binary env.
    params_path = os.path.join(os.path.dirname(CHECKPOINT), "params.json")
    with open(params_path) as f:
        params = json.load(f)
    ckpt_cont = (params.get("env_config", {}).get("config", {})
                 .get("env", {}).get("continuous_action", None))
    print(f"Checkpoint trained continuous_action={ckpt_cont}; running env in BINARY mode for probe.")

    # ---- ENV: binary action, ego_centric obs ----
    env = create_env({"continuous_action": False}, observation_type="ego_centric")
    print(f"Env: continuous_action={env.continuous_action}, "
          f"num_agents_max={env.num_agents_max}, max_time_steps={env.config.env.max_time_steps}")

    # ---- Policies ----
    print("\nLoading HardTopKPolicy (model from checkpoint)...")
    hardtopk = HardTopKPolicy(CHECKPOINT, env, k=TOP_K)
    knn = KNNTruePolicy(k=TOP_K)
    fc = PureACSPolicy(continuous=False)

    # ---- Quick single-step edge sanity (mean edges/agent ~ k+1 self) ----
    obs0 = env.reset()
    a_htk = hardtopk(obs0)
    a_knn = knn(obs0)
    n = int(obs0["padding_mask"].sum())
    print(f"\n[sanity] one-step mean edges/agent (incl self): "
          f"HardTopK={a_htk.sum()/n:.3f}, KNN={a_knn.sum()/n:.3f} (expect ~{TOP_K+1})")
    # off-diagonal overlap between HardTopK and KNN selections
    overlap = ((a_htk == 1) & (a_knn == 1)).sum() - n  # subtract self-loops
    htk_off = a_htk.sum() - n
    print(f"[sanity] HardTopK vs KNN off-diagonal selection overlap: "
          f"{overlap}/{htk_off} = {100.0*overlap/max(htk_off,1):.1f}% "
          f"(high overlap => attention truly tracks distance)")

    # ============================ STAGE 1: SCREEN ============================
    print("\n" + "#" * 80)
    print(f"# STAGE 1 SCREEN: {args.screen_episodes} episodes each, deterministic")
    print("#" * 80)

    fc_stats = monte_carlo_evaluation(env, fc, args.screen_episodes, desc="FC (PureACS)")
    htk_stats = monte_carlo_evaluation(env, hardtopk, args.screen_episodes, desc="HardTopK")
    knn_stats = monte_carlo_evaluation(env, knn, args.screen_episodes, desc="KNN-True")

    print("\n----- print_comparison: HardTopK vs FC -----")
    print_comparison(htk_stats, fc_stats)
    print("\n----- print_comparison: KNN-True vs FC -----")
    print_comparison(knn_stats, fc_stats)

    print("\n========== STAGE 1 NUMERIC SUMMARY ==========")
    _summary("FC (PureACS)", fc_stats)
    _summary("KNN-True", knn_stats, fc_stats)
    htk_line = _summary("HardTopK", htk_stats, fc_stats)

    # ---- SANITY CHECK on FC / KNN absolute reward ranges ----
    fc_rew = fc_stats["episode_return_mean"]
    knn_rew = knn_stats["episode_return_mean"]
    fc_ok = (-300.0 <= fc_rew <= -255.0)   # FC varies ~-263..-293 across eval runs (HANDOFF Phase 11)
    knn_ok = (-245.0 <= knn_rew <= -205.0)  # expected near -225 (~17-20% better)
    print(f"\n[SANITY] FC reward={fc_rew:.3f} (expect ~-263..-293) -> {'OK' if fc_ok else 'OUT OF RANGE'}")
    print(f"[SANITY] KNN reward={knn_rew:.3f} (expect ~-225)      -> {'OK' if knn_ok else 'OUT OF RANGE'}")
    if not (fc_ok and knn_ok):
        print("\n!!! SANITY CHECK FAILED — binary env / baseline setup likely wrong. STOPPING. !!!")
        return

    # Decide whether HardTopK significantly beats FC at screen stage.
    t_htk, p_htk = _welch(htk_stats, fc_stats)
    beats = (p_htk < 0.05) and (htk_stats["episode_return_mean"] > fc_rew)
    print(f"\n[DECISION] HardTopK vs FC: t={t_htk:.3f}, p={p_htk:.4g}, "
          f"mean_diff={htk_stats['episode_return_mean']-fc_rew:+.3f} -> "
          f"{'SIGNIFICANTLY BEATS FC' if beats else 'does NOT significantly beat FC'}")

    # ============================ STAGE 2: DEFINITIVE ============================
    if beats:
        print("\n" + "#" * 80)
        print(f"# STAGE 2 DEFINITIVE: HardTopK vs FC at {args.final_episodes} episodes")
        print("#" * 80)
        fc2 = monte_carlo_evaluation(env, fc, args.final_episodes, desc="FC (PureACS) [final]")
        htk2 = monte_carlo_evaluation(env, hardtopk, args.final_episodes, desc="HardTopK [final]")
        print("\n----- print_comparison: HardTopK vs FC (FINAL) -----")
        print_comparison(htk2, fc2)
        print("\n========== STAGE 2 NUMERIC SUMMARY (FINAL) ==========")
        _summary("FC (PureACS)", fc2)
        _summary("HardTopK", htk2, fc2)
        t2, p2 = _welch(htk2, fc2)
        beats2 = (p2 < 0.05) and (htk2["episode_return_mean"] > fc2["episode_return_mean"])
        print(f"\n[FINAL DECISION] HardTopK vs FC ({args.final_episodes} ep): "
              f"t={t2:.3f}, p={p2:.4g}, "
              f"mean_diff={htk2['episode_return_mean']-fc2['episode_return_mean']:+.3f} -> "
              f"{'SIGNIFICANTLY BEATS FC' if beats2 else 'does NOT significantly beat FC'}")
    else:
        print("\n[Stage 2 skipped] HardTopK did not significantly beat FC at screen stage.")


if __name__ == "__main__":
    main()
