"""
Heuristic baseline neighbor selection strategies for flocking environments.

This module provides four heuristic baselines for neighbor selection:
1. Random Neighbor Selection: Randomly select neighbors based on probability
2. Distance-Based Neighbor Selection: Select neighbors within a distance threshold
3. Fixed-Nearest Neighbor Selection: Select k nearest neighbors
4. Fixed-Farthest Neighbor Selection: Select k farthest neighbors

All baselines:
- Accept environment observations in standard format
- Return actions in the format expected by the environment
- Handle padding agents correctly using padding_mask
- Respect communication graph using neighbor_masks
- Maintain self-loop representation (diagonal = 1)
"""

import numpy as np
from typing import Dict, Optional


class RandomNeighborSelection:
    """
    Randomly select neighbors based on a specified probability.

    Only valid (non-padding, within neighbor mask) neighbors are considered.
    Each valid neighbor is selected independently with probability p.
    """

    def __init__(self, selection_probability: float = 0.5, seed: Optional[int] = None):
        """
        Initialize Random Neighbor Selection baseline.

        Args:
            selection_probability: Probability of selecting each valid neighbor (default: 0.5)
            seed: Random seed for reproducibility (optional)
        """
        assert 0.0 <= selection_probability <= 1.0, "selection_probability must be in [0, 1]"
        self.selection_probability = selection_probability
        self.rng = np.random.RandomState(seed)

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate random neighbor selection action.

        Args:
            obs: Environment observation containing:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array

        Returns:
            action: (num_agents_max, num_agents_max) integer array with neighbor selections
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize action with zeros
        action = np.zeros((num_agents_max, num_agents_max), dtype=np.int8)

        # Set self-loops to 1 (required by environment)
        np.fill_diagonal(action, 1)

        # Create valid neighbor mask (neighbor_masks AND both agents are not padding)
        # padding_mask_2d[i, j] = True if both i and j are real agents
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d

        # Remove self-loops from valid neighbors for random selection
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # Randomly select valid neighbors
        random_selection = self.rng.rand(num_agents_max, num_agents_max) < self.selection_probability
        action = action | (valid_neighbors_no_self & random_selection).astype(np.int8)

        return action


class DistanceBasedNeighborSelection:
    """
    Select neighbors whose pairwise distance is below a threshold.

    Only valid (non-padding, within neighbor mask) neighbors are considered.
    Distances are computed from relative positions in the observation.
    """

    def __init__(self, distance_threshold: float, periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None):
        """
        Initialize Distance-Based Neighbor Selection baseline.

        Args:
            distance_threshold: Maximum distance for neighbor selection
            periodic_boundary: Whether the environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
        """
        assert distance_threshold > 0, "distance_threshold must be positive"
        self.distance_threshold = distance_threshold
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate distance-based neighbor selection action.

        Args:
            obs: Environment observation containing:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array

        Returns:
            action: (num_agents_max, num_agents_max) integer array with neighbor selections
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Extract relative positions from observation
        # For non-periodic: obs_dim=4, positions are indices 0:2 (normalized by l/2)
        # For periodic: obs_dim=6, positions are sin/cos transformed indices 0:4
        if self.periodic_boundary:
            # For periodic boundaries, positions are [cos(x), sin(x), cos(y), sin(y)]
            # We need to reconstruct distances from sin/cos representation
            # This is complex, so we'll use a simplified approach
            # Note: The environment would have already filtered by comm_range in neighbor_masks
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            # Positions are normalized: rel_pos / (l/2), where l is initial_position_bound
            # We need to denormalize to get actual distances
            rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)

            # Compute pairwise distances
            distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)

        # Create valid neighbor mask
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d

        # Select neighbors within distance threshold
        within_threshold = distances <= self.distance_threshold
        action = (within_threshold & valid_neighbors).astype(np.int8)

        # Ensure self-loops are set
        np.fill_diagonal(action, 1)

        return action


class FixedNearestNeighborSelection:
    """
    Select the k nearest valid neighbors for each agent.

    If an agent has fewer than k valid neighbors, all available neighbors are selected.
    """

    def __init__(self, k: int, periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None):
        """
        Initialize Fixed-Nearest Neighbor Selection baseline.

        Args:
            k: Number of nearest neighbors to select
            periodic_boundary: Whether the environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
        """
        assert k > 0, "k must be positive"
        self.k = k
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate k-nearest neighbor selection action.

        Args:
            obs: Environment observation containing:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array

        Returns:
            action: (num_agents_max, num_agents_max) integer array with neighbor selections
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Extract relative positions and compute distances
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)
            distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)

        # Create valid neighbor mask (excluding self-loops for k-nearest computation)
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # For each agent, select k nearest neighbors
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            # Get valid neighbors for agent i
            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # No valid neighbors

            # Get distances to valid neighbors
            distances_i = distances[i].copy()
            distances_i[~valid_mask_i] = np.inf  # Set invalid neighbors to infinity

            # Find k nearest neighbors
            # Use argpartition for efficiency (only partial sort)
            num_valid = valid_mask_i.sum()
            k_actual = min(self.k, num_valid)

            if k_actual > 0:
                nearest_indices = np.argpartition(distances_i, k_actual - 1)[:k_actual]
                action[i, nearest_indices] = 1

        return action


class FixedFarthestNeighborSelection:
    """
    Select the k farthest valid neighbors for each agent.

    If an agent has fewer than k valid neighbors, all available neighbors are selected.
    """

    def __init__(self, k: int, periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None):
        """
        Initialize Fixed-Farthest Neighbor Selection baseline.

        Args:
            k: Number of farthest neighbors to select
            periodic_boundary: Whether the environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
        """
        assert k > 0, "k must be positive"
        self.k = k
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate k-farthest neighbor selection action.

        Args:
            obs: Environment observation containing:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array

        Returns:
            action: (num_agents_max, num_agents_max) integer array with neighbor selections
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Extract relative positions and compute distances
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)
            distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)

        # Create valid neighbor mask (excluding self-loops for k-farthest computation)
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # For each agent, select k farthest neighbors
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            # Get valid neighbors for agent i
            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # No valid neighbors

            # Get distances to valid neighbors
            distances_i = distances[i].copy()
            distances_i[~valid_mask_i] = -np.inf  # Set invalid neighbors to -infinity

            # Find k farthest neighbors
            num_valid = valid_mask_i.sum()
            k_actual = min(self.k, num_valid)

            if k_actual > 0:
                # Use argpartition with negative distances to get farthest
                farthest_indices = np.argpartition(-distances_i, k_actual - 1)[:k_actual]
                action[i, farthest_indices] = 1

        return action


# Convenience function to create baselines
def create_baseline(baseline_type: str, **kwargs):
    """
    Factory function to create baseline policies.

    Args:
        baseline_type: One of 'random', 'distance', 'nearest', 'farthest'
        **kwargs: Arguments to pass to the baseline constructor

    Returns:
        Baseline policy object

    Example:
        >>> baseline = create_baseline('random', selection_probability=0.5, seed=42)
        >>> baseline = create_baseline('nearest', k=5)
    """
    baselines = {
        'random': RandomNeighborSelection,
        'distance': DistanceBasedNeighborSelection,
        'nearest': FixedNearestNeighborSelection,
        'farthest': FixedFarthestNeighborSelection,
    }

    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Must be one of {list(baselines.keys())}")

    return baselines[baseline_type](**kwargs)
