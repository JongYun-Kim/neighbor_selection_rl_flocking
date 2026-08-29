"""Canonical identifiers for Dynamic-k nearest-neighbor selection.

The persistent names written by new Dynamic-k NN training runs. Runs that
predate the rename wrote "distance_pointer" in the same slots; the pointer ->
mask rule is unchanged across it, so eval/eval_c2.py accepts both spellings
and dispatches them to the same code path.

This was a two-file package (dynamic_k_nn/{__init__,identifiers}.py) whose
__init__ only re-exported the five constants below.
"""

ACTION_TYPE = "dynamic_k_nn"
ACTION_ENCODING = "cutoff_agent_pointer"
MODEL_ID = "dynamic_k_nn_neighbor_selector_rl"
EXPERIMENT_NAME = "dynamic_k_nn_neighbor_selection"
WANDB_PROJECT = "nb-selection-dynamic-k-nn"

__all__ = [
    "ACTION_ENCODING",
    "ACTION_TYPE",
    "EXPERIMENT_NAME",
    "MODEL_ID",
    "WANDB_PROJECT",
]
