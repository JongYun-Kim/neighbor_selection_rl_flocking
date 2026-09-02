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
import pickle
from pathlib import Path

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


# ------------------------------------------------------ full-population adapter
def resolve_dynamic_checkpoint(path):
    """Resolve a Tune checkpoint or a historical evaluation archive.

    Returns a dictionary containing a canonical checkpoint directory, model
    config, observation scale, and enough information to load the state.  The
    historical ``weights/best_01`` bundles deliberately do not contain
    ``params.json``; their ``metadata.json`` is the authoritative substitute.
    """
    given = Path(path).expanduser().resolve()
    if (given.is_dir() and (given / "metadata.json").is_file()
            and (given / "model_state_dict.pt").is_file()):
        archive_root, checkpoint = given, given / "checkpoint"
    elif given.is_dir() and (given / "checkpoint").is_dir():
        archive_root, checkpoint = given, given / "checkpoint"
    else:
        checkpoint = given
        archive_root = checkpoint.parent

    params_path = checkpoint.parent / "params.json"
    policy_state = checkpoint / "policies" / "default_policy" / "policy_state.pkl"
    if checkpoint.is_dir() and params_path.is_file() and policy_state.is_file():
        with params_path.open(encoding="utf-8") as stream:
            params = json.load(stream)
        if not is_dknn_params(params):
            raise ValueError(f"not a Dynamic-k NN checkpoint: {checkpoint}")
        env_cfg = params.get("env_config", {}).get("config", {}).get("env", {})
        return {
            "checkpoint": checkpoint,
            "archive_root": checkpoint.parent,
            "kind": "rllib",
            "model_config": params["model"]["custom_model_config"],
            "obs_position_scale": env_cfg.get("obs_position_scale", "legacy"),
            "params": params,
            "config_file": params_path,
            "policy_state": policy_state,
        }

    metadata_path = archive_root / "metadata.json"
    state_path = archive_root / "model_state_dict.pt"
    if metadata_path.is_file() and state_path.is_file() and checkpoint.exists():
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        env_cfg = metadata.get("training_env_config", {}).get("env", {})
        return {
            "checkpoint": checkpoint,
            "archive_root": archive_root,
            "kind": "state_dict_archive",
            "model_config": metadata["model_config"],
            "obs_position_scale": env_cfg.get("obs_position_scale", "legacy"),
            "metadata": metadata,
            "config_file": metadata_path,
            "state_dict": state_path,
        }

    raise FileNotFoundError(
        "checkpoint needs parent params.json + policy_state.pkl, or an archive "
        f"with metadata.json + model_state_dict.pt: {given}"
    )


class DynamicKNNInferencePolicy:
    """Strict-load Dynamic-k policy supporting argmax and categorical sampling."""

    def __init__(self, checkpoint_path, env, device="cpu"):
        import torch
        from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib

        self.torch = torch
        self.device = torch.device(device)
        self.source = resolve_dynamic_checkpoint(checkpoint_path)
        n_agents = env.num_agents_max
        self.model = DynamicKNNPPORLlib(
            obs_space=env.observation_space,
            action_space=env.action_space,
            num_outputs=n_agents * n_agents,
            model_config={"custom_model_config": self.source["model_config"]},
            name="population_eval_policy",
        )
        if self.source["kind"] == "rllib":
            with self.source["policy_state"].open("rb") as stream:
                state = pickle.load(stream)
            weights = {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in state["weights"].items()
            }
        else:
            weights = torch.load(str(self.source["state_dict"]), map_location="cpu")
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.n_agents = int(n_agents)

    def logits(self, observations):
        """Return ``(B,N,N)`` pointer logits for a list of env observations."""
        torch = self.torch
        if isinstance(observations, dict):
            observations = [observations]
        tensors = {}
        for key in ("local_agent_infos", "neighbor_masks", "padding_mask",
                    "is_from_my_env"):
            values = np.stack([np.asarray(obs[key]) for obs in observations], axis=0)
            tensor = torch.as_tensor(values)
            if key in ("local_agent_infos", "neighbor_masks", "padding_mask"):
                tensor = tensor.float()
            tensors[key] = tensor.to(self.device)
        with torch.no_grad():
            flat, _ = self.model.forward({"obs": tensors}, state=[], seq_lens=None)
        return flat.reshape(len(observations), self.n_agents, self.n_agents)

    def actions(self, observations, mode="deterministic", generators=None):
        torch = self.torch
        logits = self.logits(observations)
        if mode == "deterministic":
            actions = torch.argmax(logits, dim=-1)
        elif mode == "stochastic":
            if generators is None or len(generators) != logits.shape[0]:
                raise ValueError("stochastic inference needs one generator per episode")
            probs = torch.softmax(logits, dim=-1)
            actions = torch.stack([
                torch.multinomial(probs[index], 1, replacement=True,
                                  generator=generators[index]).squeeze(-1)
                for index in range(logits.shape[0])
            ])
        else:
            raise ValueError(f"unsupported learned action mode: {mode}")
        return actions.detach().cpu().numpy().astype(np.int64, copy=False)

    def make_generator(self, seed):
        torch = self.torch
        try:
            generator = torch.Generator(device=self.device)
        except TypeError:
            if self.device.type != "cpu":
                raise RuntimeError("this Torch version lacks a CUDA Generator")
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator
