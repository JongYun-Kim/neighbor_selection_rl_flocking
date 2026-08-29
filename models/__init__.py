"""The two live neighbor-selection policy models.

NeighborSelectionPPORLlib is the binary edge-selection head ((N,N,2) logits);
DynamicKNNPPORLlib is the cutoff-pointer head ((N,N) logits -> per-ego pointer).
Dormant variants live under legacy/ (ppo_centralized, beta_dist).
"""
from models.ppo import NeighborSelectionPPORLlib
from models.ppo_dynamic_k_nn import DynamicKNNPPORLlib

__all__ = [
    "NeighborSelectionPPORLlib",
    "DynamicKNNPPORLlib",
]
