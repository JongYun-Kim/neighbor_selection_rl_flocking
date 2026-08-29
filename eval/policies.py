"""Checkpoint -> deterministic policy adapters for the C2 eval harness.

One entry point, `load_policy(checkpoint_path, env)`: it reads the run's
params.json and dispatches to the adapter for that method — `C2Policy` for the
binary edge-selection models (models.ppo) and `DknnC2Policy` for the
cutoff-pointer models (models.ppo_dynamic_k_nn). Both return argmax actions in
the encoding the eval env expects, so callers stay method-agnostic.

`induced_mask_from_obs` reconstructs the pointer -> (N,N) selection mask from
the observation, for diagnostics that need the selection graph of a pointer
policy without asking the env.

Split out of eval.eval_c2 unchanged; eval_c2 keeps the forensics wrapper, the
judge, the rollout driver and the CLI.
"""
import json
import os

import numpy as np


# Cutoff-pointer (Dynamic-k NN) identifiers, including the pre-rename
# "distance_pointer" names written by runs that predate the Dynamic-k NN
# refactor. The pointer -> mask rule (env.cutoff_indices_to_binary_action) is
# unchanged across that rename, so those checkpoints take the same code path.
DKNN_MODEL_NAMES = ("dynamic_k_nn_neighbor_selector_rl",
                    "distance_pointer_neighbor_selector_rl")
DKNN_ACTION_TYPES = ("dynamic_k_nn", "distance_pointer")


def is_dknn_params(params):
    """True if a run's params.json describes a cutoff-pointer policy."""
    custom_model = params.get("model", {}).get("custom_model", "")
    action_type = (params.get("env_config", {}).get("config", {})
                   .get("env", {}).get("action_type", "binary_vector"))
    return custom_model in DKNN_MODEL_NAMES or action_type in DKNN_ACTION_TYPES
# ---------------------------------------------------------------- policy load
class C2Policy:
    """Deterministic (argmax) policy from an RLlib checkpoint of this study.

    Standalone version of evaluate_checkpoint.RLPolicy that also feeds the
    "global_stats" obs key consumed by use_global_stats models.
    """

    def __init__(self, checkpoint_path, env):
        import pickle
        import torch
        self.torch = torch
        params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
        with open(params_path) as f:
            params = json.load(f)
        mc = params["model"]["custom_model_config"]
        from models.ppo import NeighborSelectionPPORLlib
        N = env.num_agents_max
        self.model = NeighborSelectionPPORLlib(
            obs_space=env.observation_space, action_space=env.action_space,
            num_outputs=2 * N * N, model_config={"custom_model_config": mc},
            name="eval_policy")
        with open(os.path.join(checkpoint_path, "policies", "default_policy",
                               "policy_state.pkl"), "rb") as f:
            state = pickle.load(f)
        torch_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                       for k, v in state["weights"].items()}
        missing, unexpected = self.model.load_state_dict(torch_state, strict=False)
        if missing or unexpected:
            print(f"WARN load_state_dict: missing={missing} unexpected={unexpected}")
        self.model.eval()
        self.N = N

    def __call__(self, obs):
        torch = self.torch
        with torch.no_grad():
            t = {
                "local_agent_infos": torch.from_numpy(obs["local_agent_infos"][None]).float(),
                "neighbor_masks": torch.from_numpy(obs["neighbor_masks"][None]).float(),
                "padding_mask": torch.from_numpy(obs["padding_mask"][None]).float(),
                "is_from_my_env": torch.from_numpy(np.array([True])),
            }
            for k in ("global_agent_infos", "global_stats"):
                if k in obs:
                    t[k] = torch.from_numpy(obs[k][None]).float()
            logits, _ = self.model.forward({"obs": t}, state=[], seq_lens=None)
            lg = logits.numpy()[0].reshape(self.N, self.N, 2)
            return np.argmax(lg, axis=-1).astype(np.int8)


class DknnC2Policy:
    """Deterministic (argmax) cutoff-pointer policy from a dynamic_k_nn
    checkpoint: per-ego N-way argmax over pointer logits -> (N,) int64 pointer,
    fed to the env as-is (mask conversion happens inside the env)."""

    def __init__(self, checkpoint_path, env, custom_model_config):
        import pickle
        import torch
        self.torch = torch
        from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib
        N = env.num_agents_max
        self.model = DynamicKNNPPORLlib(
            obs_space=env.observation_space, action_space=env.action_space,
            num_outputs=N * N,
            model_config={"custom_model_config": custom_model_config},
            name="eval_policy")
        with open(os.path.join(checkpoint_path, "policies", "default_policy",
                               "policy_state.pkl"), "rb") as f:
            state = pickle.load(f)
        torch_state = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                       for k, v in state["weights"].items()}
        # strict=True on purpose: the pointer model and the binary-vector model
        # share an identical parameter set, so a mis-dispatched checkpoint would
        # load silently under strict=False and then have its N*N pointer logits
        # reinterpreted as N*N*2 binary logits. Only the head arity differs, and
        # only this assertion sees it.
        self.model.load_state_dict(torch_state, strict=True)
        self.model.eval()
        self.N = N

    def __call__(self, obs):
        torch = self.torch
        with torch.no_grad():
            t = {
                "local_agent_infos": torch.from_numpy(obs["local_agent_infos"][None]).float(),
                "neighbor_masks": torch.from_numpy(obs["neighbor_masks"][None]).float(),
                "padding_mask": torch.from_numpy(obs["padding_mask"][None]).float(),
                "is_from_my_env": torch.from_numpy(np.array([True])),
            }
            logits, _ = self.model.forward({"obs": t}, state=[], seq_lens=None)
            lg = logits.numpy()[0].reshape(self.N, self.N)
            return np.argmax(lg, axis=-1).astype(np.int64)


def induced_mask_from_obs(obs, pointer):
    """Replicate the env's cutoff-pointer -> (N,N) mask conversion from the
    observation (obs distances = world distances x 2/L; the '<=' cutoff rule
    is scale-invariant). Diagnostics-grade twin of env info['binary_action']."""
    pm = obs["padding_mask"].astype(bool)
    N = pm.shape[0]
    rel = obs["local_agent_infos"][:, :, :2].astype(np.float64)
    d = np.sqrt((rel ** 2).sum(-1))  # (N, N) ego-row distances, self = 0
    mask = np.zeros((N, N), dtype=np.int8)
    for i in np.where(pm)[0]:
        thr = d[i, pointer[i]]
        row = pm & (d[i] <= thr)
        row[i] = True
        mask[i] = row.astype(np.int8)
    return mask


def load_policy(checkpoint_path, env):
    """params.json-driven method dispatch (Q8/B6 adapter)."""
    params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
    with open(params_path) as f:
        params = json.load(f)
    if is_dknn_params(params):
        mc = params["model"]["custom_model_config"]
        return DknnC2Policy(checkpoint_path, env, mc)
    return C2Policy(checkpoint_path, env)
