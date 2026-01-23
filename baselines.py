"""
Heuristic baseline neighbor selection strategies for flocking environments.

This module provides five heuristic baselines for neighbor selection:
1. Random Neighbor Selection: Randomly select neighbors based on probability
2. Distance-Based Neighbor Selection: Select neighbors within a distance threshold
3. Fixed-Nearest Neighbor Selection: Select k nearest neighbors
4. Fixed-Farthest Neighbor Selection: Select k farthest neighbors
5. Metric-Topological Interaction (MTI): Adaptive neighbor selection with mode switching

All baselines:
- Accept environment observations in standard format
- Return actions in the format expected by the environment
- Handle padding agents correctly using padding_mask
- Respect communication graph using neighbor_masks
- Maintain self-loop representation (diagonal = 1)
"""

import numpy as np
from scipy.spatial import Voronoi
from typing import Dict, Optional


class VoronoiNeighborSelection:
    """
    Reference: (2010 Ginelli) Relevance of Metric-Free Interactions in Flocking Phenomena

    Select neighbors based on Voronoi tessellation.
    Use vor.ridge_points to determine neighbors, which is adjacent points sharing a Voronoi ridge.
    """

    def __init__(self):
        pass

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate Voronoi-based neighbor selection action.

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

        # Create valid neighbor mask
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d

        # Get indices of real agents
        real_agent_indices = np.where(padding_mask)[0]
        num_real_agents = len(real_agent_indices)

        # Voronoi requires at least 3 points in 2D
        if num_real_agents < 3:
            # Fall back to selecting all valid neighbors
            valid_neighbors_no_self = valid_neighbors.copy()
            np.fill_diagonal(valid_neighbors_no_self, False)
            action = action | valid_neighbors_no_self.astype(np.int8)
            return action

        # Reconstruct positions from local_agent_infos
        # Use agent 0 as reference and compute absolute positions from relative positions
        # local_agent_infos[i, j, 0:2] = normalized relative position of j from i's perspective
        # We pick a reference agent (first real agent) and use its view to get positions
        ref_agent = real_agent_indices[0]
        positions = np.zeros((num_agents_max, 2))
        
        # Position of reference agent is at origin (we only need relative positions for Voronoi)
        positions[ref_agent] = [0.0, 0.0]
        
        # Get positions of other agents relative to reference agent
        for j in real_agent_indices:
            if j != ref_agent:
                positions[j] = local_agent_infos[ref_agent, j, :2]

        # Extract positions for real agents only
        real_positions = positions[real_agent_indices]  # (num_real_agents, 2)

        # Compute Voronoi tessellation
        vor = Voronoi(real_positions)

        # Build Voronoi neighbor adjacency from ridge_points
        voronoi_adj = np.zeros((num_agents_max, num_agents_max), dtype=bool)
        for a, b in vor.ridge_points:
            # Map back to original indices
            orig_a = real_agent_indices[a]
            orig_b = real_agent_indices[b]
            voronoi_adj[orig_a, orig_b] = True
            voronoi_adj[orig_b, orig_a] = True

        # Select neighbors that are both Voronoi neighbors AND valid (in neighbor_masks)
        selected = voronoi_adj & valid_neighbors
        action = action | selected.astype(np.int8)

        return action


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


class MetricTopologicalInteractionSelection:
    """
    Metric-Topological Interaction (MTI) neighbor selection baseline.

    Based on: "Metric–topological interaction model of collective behavior"
    (Niizato & Gunji, 2011)

    Each agent maintains an internal mode (TOP or MET) that switches based on
    heading alignment with neighbors. This creates adaptive neighbor selection
    that responds to local swarm dynamics.

    - TOP (topological) mode: Select k nearest neighbors
    - MET (metric) mode: Select all neighbors within distance threshold R

    Mode switching:
    - TOP → MET: When heading alignment with neighbors is below threshold a
    - MET → TOP: When two random neighbors have heading difference above threshold b

    This is a STATEFUL baseline that remembers each agent's mode across time steps.
    """

    MODE_TOP = 0  # Topological mode
    MODE_MET = 1  # Metric mode

    def __init__(self, k: int, distance_threshold: float,
                 threshold_a: float, threshold_b: float,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialize MTI baseline.

        Args:
            k: Number of nearest neighbors in TOP mode
            distance_threshold: Distance threshold R for MET mode (normalized units)
            threshold_a: TOP→MET switch threshold (radians), heading difference below this triggers switch
            threshold_b: MET→TOP switch threshold (radians), heading difference above this triggers switch
            periodic_boundary: Whether environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
            seed: Random seed for reproducibility in MET mode sampling
        """
        assert k > 0, "k must be positive"
        assert distance_threshold > 0, "distance_threshold must be positive"
        assert threshold_a >= 0, "threshold_a must be non-negative"
        assert threshold_b >= 0, "threshold_b must be non-negative"

        self.k = k
        self.distance_threshold = distance_threshold
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

        # State: modes for each agent (initialized on first call)
        self.modes = None
        self.num_agents_max = None

    def reset(self, num_agents_max: Optional[int] = None):
        """
        Reset agent modes to TOP (topological mode).

        Call this at the start of each episode, or let the baseline auto-detect resets.

        Args:
            num_agents_max: Maximum number of agents (inferred from obs if not provided)
        """
        if num_agents_max is not None:
            self.num_agents_max = num_agents_max
            self.modes = np.full(num_agents_max, self.MODE_TOP, dtype=np.int8)
        else:
            self.modes = None
            self.num_agents_max = None

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate MTI neighbor selection action.

        Args:
            obs: Environment observation containing:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array
                  where obs_dim=4 for non-periodic: [rel_x, rel_y, cos(rel_heading), sin(rel_heading)]

        Returns:
            action: (num_agents_max, num_agents_max) integer array with neighbor selections
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize or reset modes if needed (auto-detect environment reset)
        if self.modes is None or self.num_agents_max != num_agents_max:
            self.reset(num_agents_max)

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Compute distances
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)
            distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)

        # Extract relative headings from observation
        # local_agent_infos[:, :, 2] = cos(theta_j - theta_i)
        # local_agent_infos[:, :, 3] = sin(theta_j - theta_i)
        rel_heading_cos = local_agent_infos[:, :, 2]
        rel_heading_sin = local_agent_infos[:, :, 3]

        # Compute relative heading angles: theta_j - theta_i
        rel_headings = np.arctan2(rel_heading_sin, rel_heading_cos)  # (num_agents_max, num_agents_max)

        # Valid neighbors (excluding self-loops for selection)
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # New modes for next time step
        new_modes = self.modes.copy()

        # Process each agent
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                # No valid neighbors, keep current mode and self-loop only
                continue

            # Select neighbors based on current mode
            if self.modes[i] == self.MODE_TOP:
                # TOP mode: select k nearest neighbors
                selected_neighbors = self._select_top_neighbors(i, distances, valid_mask_i)
                action[i, selected_neighbors] = 1

                # Check switching condition: TOP → MET
                if len(selected_neighbors) > 0:
                    new_modes[i] = self._check_top_to_met_switch(
                        i, selected_neighbors, rel_headings)

            else:  # MODE_MET
                # MET mode: select all neighbors within distance threshold
                selected_neighbors = self._select_met_neighbors(i, distances, valid_mask_i)
                action[i, selected_neighbors] = 1

                # Check switching condition: MET → TOP
                if len(selected_neighbors) >= 2:
                    new_modes[i] = self._check_met_to_top_switch(
                        i, selected_neighbors, rel_headings)

        # Update modes for next time step
        self.modes = new_modes

        return action

    def _select_top_neighbors(self, agent_i: int, distances: np.ndarray,
                             valid_mask: np.ndarray) -> np.ndarray:
        """Select k nearest neighbors in TOP mode."""
        distances_i = distances[agent_i].copy()
        distances_i[~valid_mask] = np.inf

        num_valid = valid_mask.sum()
        k_actual = min(self.k, num_valid)

        if k_actual > 0:
            nearest_indices = np.argpartition(distances_i, k_actual - 1)[:k_actual]
            return nearest_indices
        return np.array([], dtype=np.int32)

    def _select_met_neighbors(self, agent_i: int, distances: np.ndarray,
                             valid_mask: np.ndarray) -> np.ndarray:
        """Select all neighbors within distance threshold in MET mode."""
        distances_i = distances[agent_i]
        within_threshold = (distances_i <= self.distance_threshold) & valid_mask
        return np.where(within_threshold)[0]

    def _check_top_to_met_switch(self, agent_i: int, selected_neighbors: np.ndarray,
                                 rel_headings: np.ndarray) -> int:
        """
        Check if agent should switch from TOP to MET mode.

        Compute circular mean of neighbors' headings relative to agent i,
        then check if heading difference is below threshold_a.
        """
        # Get relative headings of selected neighbors: (theta_j - theta_i)
        neighbor_rel_headings = rel_headings[agent_i, selected_neighbors]

        # Compute circular mean of relative headings
        # mean(theta_j) - theta_i = atan2(mean(sin(theta_j - theta_i)), mean(cos(theta_j - theta_i)))
        mean_cos = np.mean(np.cos(neighbor_rel_headings))
        mean_sin = np.mean(np.sin(neighbor_rel_headings))
        mean_rel_heading = np.arctan2(mean_sin, mean_cos)

        # Heading difference between agent and mean of neighbors
        # |theta_i - mean(theta_j)| = |-(mean(theta_j) - theta_i)| = |mean_rel_heading|
        heading_diff = abs(mean_rel_heading)

        # Switch to MET if heading difference is small (well-aligned)
        if heading_diff <= self.threshold_a:
            return self.MODE_MET
        return self.MODE_TOP

    def _check_met_to_top_switch(self, agent_i: int, selected_neighbors: np.ndarray,
                                 rel_headings: np.ndarray) -> int:
        """
        Check if agent should switch from MET to TOP mode.

        Randomly sample two distinct neighbors and check their heading difference.
        If fewer than 2 neighbors, stay in MET mode.
        """
        if len(selected_neighbors) < 2:
            # Not enough neighbors to sample, stay in MET mode
            return self.MODE_MET

        # Randomly sample two distinct neighbors
        sampled = self.rng.choice(selected_neighbors, size=2, replace=False)
        j1, j2 = sampled[0], sampled[1]

        # Get relative headings: (theta_j1 - theta_i) and (theta_j2 - theta_i)
        rel_heading_j1 = rel_headings[agent_i, j1]
        rel_heading_j2 = rel_headings[agent_i, j2]

        # Compute theta_j1 - theta_j2 = (theta_j1 - theta_i) - (theta_j2 - theta_i)
        heading_diff_j1_j2 = rel_heading_j1 - rel_heading_j2

        # Wrap to [-pi, pi]
        heading_diff_j1_j2 = self._wrap_to_pi(heading_diff_j1_j2)

        # Switch to TOP if neighbors have large heading difference (poor alignment)
        if abs(heading_diff_j1_j2) > self.threshold_b:
            return self.MODE_TOP
        return self.MODE_MET

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))


# Convenience function to create baselines
def create_baseline(baseline_type: str, **kwargs):
    """
    Factory function to create baseline policies.

    Args:
        baseline_type: One of 'random', 'distance', 'nearest', 'farthest', 'mti'
        **kwargs: Arguments to pass to the baseline constructor

    Returns:
        Baseline policy object

    Example:
        >>> baseline = create_baseline('random', selection_probability=0.5, seed=42)
        >>> baseline = create_baseline('nearest', k=5)
        >>> baseline = create_baseline('mti', k=5, distance_threshold=0.5,
        ...                           threshold_a=0.1, threshold_b=0.5, seed=42)
    """
    baselines = {
        'random': RandomNeighborSelection,
        'distance': DistanceBasedNeighborSelection,
        'nearest': FixedNearestNeighborSelection,
        'farthest': FixedFarthestNeighborSelection,
        'mti': MetricTopologicalInteractionSelection,
        'voronoi': VoronoiNeighborSelection,
    }

    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Must be one of {list(baselines.keys())}")

    return baselines[baseline_type](**kwargs)
