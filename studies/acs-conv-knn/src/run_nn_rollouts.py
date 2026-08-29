"""Roll out the project's winning NN checkpoint (Phase 14 hard top-K=10 +
dist_aux, trained-as-evaluated, deterministic) under the study harness.

Checkpoint: test_results/hardtopk10_distaux_260529/.../checkpoint_000010
Loading reuses evaluate_checkpoint.RLPolicy verbatim (argmax over (N,N,2) logits
-> binary int8 action; with the model's hard_top_k flag this reproduces exactly
the evaluated behavior). CPU-only: GPUs hidden before torch import.

Output: data/nn_hardtopk/*.npz (16 seeds x 6000 steps, N=20, L=250).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only inference
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import sys  # noqa: E402

STUDY = "/workspace/studies/acs-conv-knn"
sys.path.insert(0, os.path.join(STUDY, "src"))
sys.path.insert(0, "/workspace")

CKPT = ("/workspace/test_results/hardtopk10_distaux_260529/"
        "GradLoggingPPO_neighbor_selection_flocking_env_87313_00000_0_"
        "2026-05-29_09-34-02/checkpoint_000010")


def main():
    import torch
    torch.set_num_threads(4)
    from legacy.evaluate_checkpoint import RLPolicy   # heavy import (torch/ray)
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input
    from common import build_config, rollout, save_run

    cfg = build_config(n_agents=20, max_steps=6000, initial_position_bound=250.0)
    cfg.env.expose_aux_target = True                  # model may consume global infos
    tmp_env = NeighborSelectionFlockingEnv(config_to_env_input(cfg, seed_id=0))
    policy = RLPolicy(CKPT, tmp_env, observation_type="ego_centric")

    for rep in range(16):
        seed = 1000 + rep
        out = os.path.join(STUDY, "data", "nn_hardtopk", f"nn_L250_s{seed}.npz")
        if os.path.exists(out):
            continue
        rec, snaps, ts, meta = rollout(policy, cfg, seed, pos_stride=10,
                                       extra_meta=dict(policy="hardtopk10_distaux_ckpt10"))
        save_run(out, rec, snaps, ts, meta)
        import numpy as np
        T = meta["steps_done"]
        print(f"seed {seed}: sp_end={np.nanmean(rec['s_ent'][T-200:T]):7.2f} "
              f"phi_end={np.nanmean(rec['phi'][T-200:T]):.4f} "
              f"deg={np.nanmean(rec['deg_mean'][T-200:T]):.2f} "
              f"churn={np.nanmean(rec['churn'][T-200:T]):.4f} "
              f"comp={rec['n_comp_r0'][T]:.0f}", flush=True)
    print("nn_hardtopk batch done")


if __name__ == "__main__":
    main()
