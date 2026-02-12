"""
Heuristic baseline neighbor selection strategies for flocking environments.

This module provides eight heuristic baselines for neighbor selection:
1. Voronoi Neighbor Selection: Select neighbors based on Voronoi tessellation
2. Random Neighbor Selection: Randomly select neighbors based on probability
3. Distance-Based Neighbor Selection: Select neighbors within a distance threshold
4. Fixed-Nearest Neighbor Selection: Select k nearest neighbors
5. Fixed-Farthest Neighbor Selection: Select k farthest neighbors
6. Metric-Topological Interaction (MTI): Adaptive neighbor selection with mode switching
7. Highest-Degree Neighbor Selection: Select neighbors with highest degree (most connections)
8. Modified Fixed Number of Neighbors (MFNN): Select nearest neighbor in angular sectors

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


class HighestDegreeNeighborSelection:
    """
    Select neighbors with the highest degree (most connections).

    Based on: "Research on swarm consistent performance of improved Vicsek model
    with neighbors' degree"

    This baseline selects the beta neighbors that have the most connections to other
    agents. The intuition is that well-connected neighbors may be more informative
    for achieving consensus or alignment.

    Degree is computed as the number of valid neighbors each agent has.
    """

    def __init__(self, beta: int, periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None):
        """
        Initialize Highest-Degree Neighbor Selection baseline.

        Args:
            beta: Number of highest-degree neighbors to select
            periodic_boundary: Whether environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
        """
        assert beta > 0, "beta must be positive"
        self.beta = beta
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate highest-degree neighbor selection action.

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

        # Compute degree for each agent (number of valid neighbors, including self)
        # degree[j] = number of agents that have j as a neighbor
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d

        # Degree of agent j = sum of column j in valid_neighbors
        degrees = valid_neighbors.sum(axis=0)  # (num_agents_max,)

        # Compute distances for tie-breaking
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]
            distances = np.linalg.norm(rel_positions, axis=2)

        # Create valid neighbor mask (excluding self-loops)
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # For each agent, select beta neighbors with highest degree
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # No valid neighbors

            # Get valid neighbor indices
            valid_neighbor_indices = np.where(valid_mask_i)[0]

            # Get degrees of valid neighbors
            neighbor_degrees = degrees[valid_neighbor_indices]

            # Get distances to valid neighbors (for tie-breaking)
            neighbor_distances = distances[i, valid_neighbor_indices]

            # Sort by degree (descending), then by distance (ascending) for tie-breaking
            # Create sorting key: negative degree (for descending), then distance (for ascending)
            sort_keys = np.column_stack((-neighbor_degrees, neighbor_distances))
            sorted_indices = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))

            # Select top beta neighbors
            beta_actual = min(self.beta, len(valid_neighbor_indices))
            selected_local_indices = sorted_indices[:beta_actual]
            selected_neighbors = valid_neighbor_indices[selected_local_indices]

            action[i, selected_neighbors] = 1

        return action


class ModifiedFixedNumberNeighbors:
    """
    Modified Fixed Number of Neighbors (MFNN) selection.

    Based on: "Enhancing synchronization of self-propelled particles via
    modified rule of fixed number of neighbors"

    This baseline divides the space around each agent into (k-1) equal angular
    sectors and selects the nearest neighbor in each sector. This ensures
    spatial diversity in neighbor selection, preventing clustering of selected
    neighbors in one direction.

    The angular sectors are defined relative to the agent's local coordinate frame.
    """

    def __init__(self, k: int, periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None):
        """
        Initialize Modified Fixed Number of Neighbors baseline.

        Args:
            k: Maximum number of neighbors to select (space divided into k-1 sectors)
            periodic_boundary: Whether environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
        """
        assert k > 1, "k must be at least 2 to create sectors"
        self.k = k
        self.num_sectors = k - 1
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate MFNN neighbor selection action.

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

        # Compute distances and angles
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary distance computation not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)
            distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)

            # Compute angles of neighbors relative to agent
            # Note: This is in the world frame, not rotated to agent's heading
            # For a more accurate implementation aligned with agent heading,
            # we would need absolute heading information
            angles = np.arctan2(rel_positions[:, :, 1], rel_positions[:, :, 0])  # (num_agents_max, num_agents_max)

        # Create valid neighbor mask (excluding self-loops)
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # Define angular sectors
        # Divide [-π, π] into num_sectors equal sectors
        sector_boundaries = np.linspace(-np.pi, np.pi, self.num_sectors + 1)

        # For each agent, select nearest neighbor in each sector
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # No valid neighbors

            # Get angles and distances to valid neighbors
            valid_neighbor_indices = np.where(valid_mask_i)[0]
            neighbor_angles = angles[i, valid_neighbor_indices]
            neighbor_distances = distances[i, valid_neighbor_indices]

            # For each sector, find the nearest neighbor
            for sector_idx in range(self.num_sectors):
                sector_min = sector_boundaries[sector_idx]
                sector_max = sector_boundaries[sector_idx + 1]

                # Find neighbors in this sector
                # Handle the wrap-around at ±π
                if sector_idx == self.num_sectors - 1:
                    # Last sector: include right boundary
                    in_sector = (neighbor_angles >= sector_min) & (neighbor_angles <= sector_max)
                else:
                    in_sector = (neighbor_angles >= sector_min) & (neighbor_angles < sector_max)

                if not in_sector.any():
                    continue  # No neighbors in this sector

                # Find nearest neighbor in this sector
                sector_distances = neighbor_distances[in_sector]
                sector_neighbor_indices = valid_neighbor_indices[in_sector]

                nearest_idx_in_sector = np.argmin(sector_distances)
                nearest_neighbor = sector_neighbor_indices[nearest_idx_in_sector]

                action[i, nearest_neighbor] = 1

        return action


class ActiveSearchNeighborSelection:
    """
    Active Search 기반 이웃 선택 베이스라인.

    ============================================================================
    논문 참조: "Active Search Promotes the Collective Behavior of Underwater
               Robots With Limited Field of View"
               (Zhou et al., IEEE Robotics and Automation Letters, Vol. 10, No. 6, June 2025)
    ============================================================================

    이 베이스라인은 논문의 Section II에서 제안된 Active Search 모델의
    이웃 선택 메커니즘(Eq. 2, 3, 6, 9, 10)을 구현합니다.

    핵심 아이디어:
    - 시야 내에 이웃이 있을 때: 시야에서 벗어날 가능성이 높은 이웃과 정렬
    - 시야 내에 이웃이 없을 때: 과거에 이웃을 많이 본 방향으로 탐색(회전)
    - 이를 통해 제한된 시야각에서도 군집 유지 및 집단 행동 창발

    파라미터:
    - time_window (ΔT): Active search를 위한 과거 관찰 시간 윈도우
    - alignment_enabled (bool): Velocity alignment 활성화 여부
    - search_enabled (bool): Active search 활성화 여부
    """

    def __init__(self,
                 time_window: int = 10,
                 alignment_enabled: bool = True,
                 search_enabled: bool = True,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Active Search 베이스라인 초기화.

        Args:
            time_window: Active search를 위한 과거 관찰 시간 윈도우 (타임스텝 수)
            alignment_enabled: Velocity alignment 활성화 여부
            search_enabled: Active search 활성화 여부
            periodic_boundary: 주기적 경계 사용 여부
            boundary_size: 주기적 경계 크기
            seed: 재현성을 위한 랜덤 시드
        """
        assert time_window > 0, "time_window must be positive"

        self.time_window = time_window
        self.alignment_enabled = alignment_enabled
        self.search_enabled = search_enabled
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, \
                "boundary_size required for periodic boundaries"

        self.observation_history = []
        self.num_agents_max = None

    def reset(self):
        """
        에피소드 시작 시 상태 초기화.

        Standard reset method with no parameters.
        """
        self.observation_history = []
        self.num_agents_max = None

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Active Search 기반 이웃 선택 액션 생성.

        Args:
            obs: 환경 관측값 딕셔너리

        Returns:
            action: (num_agents_max, num_agents_max) integer array
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Auto-detect episode reset
        if self.num_agents_max is None or self.num_agents_max != num_agents_max:
            self.reset()
            self.num_agents_max = num_agents_max

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Extract relative positions
        if self.periodic_boundary:
            raise NotImplementedError(
                "Periodic boundary not yet implemented for Active Search baseline"
            )
        else:
            rel_positions = local_agent_infos[:, :, :2]

        # Create valid neighbor masks
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # Store observation history for active search
        if self.search_enabled:
            self.observation_history.append({
                'rel_positions': rel_positions.copy(),
                'valid_neighbors': valid_neighbors_no_self.copy()
            })

            if len(self.observation_history) > self.time_window:
                self.observation_history.pop(0)

        # Select neighbors for each agent
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue

            valid_mask_i = valid_neighbors_no_self[i]

            # Case 1: Neighbors exist
            if valid_mask_i.any():
                if self.alignment_enabled and len(self.observation_history) >= 2:
                    # Velocity alignment: select neighbor with highest p_j^i
                    selected = self._select_neighbor_with_alignment(
                        i, rel_positions, valid_mask_i
                    )
                else:
                    # Random selection
                    valid_indices = np.where(valid_mask_i)[0]
                    selected = self.rng.choice(valid_indices)

                action[i, selected] = 1

            # Case 2: No neighbors (keep self-loop only)
            else:
                pass

        return action

    def _select_neighbor_with_alignment(
        self,
        agent_i: int,
        rel_positions: np.ndarray,
        valid_mask: np.ndarray
    ) -> int:
        """
        Velocity alignment를 위한 이웃 선택.

        Selection criterion: p_j^i = ∠(θ_j^i(t), θ_j^i(t-τ)) · θ_j^i(t)

        Args:
            agent_i: 현재 에이전트 인덱스
            rel_positions: 상대 위치 배열
            valid_mask: 유효한 이웃 마스크 (self-loop 제외)

        Returns:
            selected: 선택된 이웃의 인덱스
        """
        current_rel_pos = rel_positions[agent_i]

        # Get past relative positions
        if len(self.observation_history) >= 2:
            past_obs = self.observation_history[-2]
            past_rel_pos = past_obs['rel_positions'][agent_i]
        else:
            past_rel_pos = current_rel_pos

        # Calculate angles
        theta_current = np.arctan2(current_rel_pos[:, 1], current_rel_pos[:, 0])
        theta_past = np.arctan2(past_rel_pos[:, 1], past_rel_pos[:, 0])

        # Calculate p_j^i
        angle_diff = np.abs(self._wrap_to_pi(theta_current - theta_past))
        angle_from_center = np.abs(theta_current)
        p_values = angle_diff * angle_from_center

        # Exclude invalid neighbors
        p_values[~valid_mask] = -np.inf

        # Handle case where all values are -inf
        if np.all(np.isinf(p_values)):
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                selected = valid_indices[0]
            else:
                selected = agent_i  # Fallback (should not happen)
        else:
            selected = np.argmax(p_values)

        return selected

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        """각도를 [-π, π] 범위로 래핑."""
        return np.arctan2(np.sin(angle), np.cos(angle))


class GazingPreferenceNeighborSelection:
    """
    Gazing Preference 기반 이웃 선택 베이스라인.

    ============================================================================
    논문 참조: "Gazing Preference Induced Controllable Milling Behavior in Swarm Robotics"
               (Zhou et al., IEEE Robotics and Automation Letters, Vol. 10, No. 6, June 2025)
    ============================================================================

    이 베이스라인은 논문의 Section II에서 제안된 gazing preference 모델의
    이웃 선택 메커니즘(Eq. 5, 6)을 구현합니다.

    핵심 아이디어:
    - 각 로봇은 gazing preference 방향 g_i를 가짐
    - gazing preference 방향에 가까운 이웃이 더 높은 확률로 선택됨
    - 이를 통해 milling(회전) 행동을 유도

    ============================================================================
    [논문 vs 구현 차이점]

    1. 속도 방향 vs heading 방향:
       - 논문: v_i (속도 벡터)를 기준으로 gazing preference 방향 g_i 계산
       - 구현: 환경에서 절대 속도 정보가 없으므로, 로봇의 heading 방향을
              속도 방향의 대리(proxy)로 사용. 이는 로봇이 heading 방향으로
              이동한다는 일반적인 가정에 기반함.

    2. 이웃 선택 방식:
       - 논문: Eq. 6의 확률에 따라 단일 이웃 s 선택 후 Eq. 5의 힘 적용
       - 구현: 환경의 neighbor selection 프레임워크에 맞춰 확률적 단일 선택 구현
    ============================================================================

    파라미터:
    - theta_g (θ^g): gazing preference 각도 (velocity 방향 기준, 라디안)
                     양수: 반시계 방향 (counterclockwise milling 유도)
                     음수: 시계 방향 (clockwise milling 유도)
    - lambda_coef (λ): 확률 밀도 함수 형태 계수 (값이 클수록 선호 각도에 집중)
    - beta: gazing preference 힘의 강도 (선택적, 디버깅/분석용)
    """

    def __init__(self,
                 theta_g: float = np.pi / 3,
                 lambda_coef: float = 3.0,
                 beta: float = 0.5,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Gazing Preference 베이스라인 초기화.

        =========================================================================
        [논문 참조] Section II, Fig. 1(b), (d) 및 Eq. (6)

        논문의 시뮬레이션 파라미터 (Section III.A):
        - θ^g ∈ [-π/2, π/2]: gazing preference 각도
        - λ ∈ {1, 3, 5}: 확률 밀도 함수 계수
        - β ∈ [0, 1]: gazing preference 힘 계수
        =========================================================================

        Args:
            theta_g (float):
                θ^g - gazing preference 각도 (라디안)
                [논문 참조] Section II, Fig. 1(b): "the angle θ^g between g_i and v_i
                may not be zero"
                [논문 참조] Section III.A: "the preference angle θ^g ∈ [-π/2, π/2]"
                [논문 참조] Fig. 3: θ^g = π/3 → counterclockwise milling (M → 1)
                                   θ^g = -π/3 → clockwise milling (M → -1)
                기본값: π/3 (논문 Fig. 3에서 사용한 대표적인 값)

            lambda_coef (float):
                λ - 확률 밀도 함수의 형태를 제어하는 계수
                [논문 참조] Eq. (6): "The parameter λ > 0 is a coefficient that
                controls the shape of the probability density function (PDF)"
                [논문 참조] Fig. 1(d): "the larger the value of λ, the higher the
                probability that a robot near the gazing preference angle will be chosen"
                [논문 참조] Fig. 4: λ ∈ {1, 3, 5}에서 실험
                기본값: 3.0 (논문 Fig. 3에서 사용한 값)

            beta (float):
                β - gazing preference 힘의 강도 계수
                [논문 참조] Eq. (5): "β is a coefficient that adjusts the strength
                of the gazing preference term"
                [논문 참조] Fig. 3: β = 0.7 사용
                [논문 참조] Section III.B: "the milling behavior occurs mainly in
                the region where β is relatively small, i.e., where β is between
                0.1 and 0.5"
                기본값: 0.5 (논문에서 권장하는 범위 내)
                [구현 참고] 이 파라미터는 neighbor selection에서 직접 사용되지 않으나,
                           논문과의 일관성 및 추후 확장을 위해 포함

            periodic_boundary (bool): 주기적 경계 사용 여부
            boundary_size (float, optional): 주기적 경계 크기
            seed (int, optional): 재현성을 위한 랜덤 시드
        """
        # =====================================================================
        # [파라미터 검증]
        # [논문 참조] Section III.A의 파라미터 범위 기반
        # =====================================================================
        assert -np.pi / 2 <= theta_g <= np.pi / 2, \
            f"theta_g must be in [-π/2, π/2], got {theta_g}"
        # [논문 참조] Eq. (6): "The parameter λ > 0"
        assert lambda_coef > 0, \
            f"lambda_coef must be positive, got {lambda_coef}"
        # [논문 참조] Fig. 4: β ∈ [0, 1] 범위에서 실험
        assert 0 <= beta <= 1, \
            f"beta must be in [0, 1], got {beta}"

        # 논문의 표기법과 일치하도록 변수명 설정
        self.theta_g = theta_g      # θ^g: gazing preference 각도
        self.lambda_coef = lambda_coef  # λ: 확률 밀도 계수
        self.beta = beta            # β: gazing preference 힘 계수
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, \
                "boundary_size required for periodic boundaries"

        # 에이전트 수 추적 (환경 리셋 감지용)
        self.num_agents_max = None

    def reset(self):
        """
        에피소드 시작 시 상태 초기화.

        Standard reset method with no parameters.

        =========================================================================
        [구현 참고] 이 베이스라인은 stateless하므로 별도의 상태 초기화 불필요.
                   단, 환경 리셋 감지를 위해 num_agents_max 추적.
        =========================================================================
        """
        self.num_agents_max = None

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Gazing Preference 기반 이웃 선택 액션 생성.

        =========================================================================
        [논문 참조] Section II, Eq. (5), (6)의 알고리즘 구현

        알고리즘 개요:
        1. 각 에이전트 i의 gazing preference 방향 g_i 계산
        2. 각 이웃 j에 대해 φ_ij (g_i와 이웃 방향 사이의 각도) 계산
        3. Eq. (6)에 따라 선택 확률 p_i(j) 계산
        4. 확률에 따라 이웃 선택
        =========================================================================

        Args:
            obs: 환경 관측값 딕셔너리:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array
                  obs_dim=4 (비주기적): [rel_x, rel_y, cos(rel_heading), sin(rel_heading)]

        Returns:
            action: (num_agents_max, num_agents_max) integer array
                - action[i, j] = 1: 에이전트 i가 에이전트 j를 이웃으로 선택
                - 각 에이전트는 gazing preference 확률에 따라 단일 이웃 선택
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # =====================================================================
        # [환경 리셋 감지]
        # Auto-detect episode reset - check both None and size change
        # =====================================================================
        if self.num_agents_max is None or self.num_agents_max != num_agents_max:
            self.reset()
            self.num_agents_max = num_agents_max

        # =====================================================================
        # [Step 1] 액션 초기화 (self-loop 필수)
        # =====================================================================
        action = np.eye(num_agents_max, dtype=np.int8)

        # =====================================================================
        # [Step 2] 상대 위치 추출 및 거리 계산
        # [논문 참조] Eq. (6): "φ_is = ∠(r̃_s - r_i, g_i)"
        # r̃_s - r_i는 에이전트 i에서 이웃 s로의 상대 위치 벡터
        # =====================================================================
        if self.periodic_boundary:
            raise NotImplementedError(
                "Periodic boundary not yet implemented for Gazing Preference baseline"
            )
        else:
            # ================================================================
            # [논문 참조] Section II: "d_ij = ||r_i - r̃_j||"
            # 환경의 관측값은 이미 상대 위치를 제공: rel_pos[i,j] = pos_j - pos_i
            # ================================================================
            rel_positions = local_agent_infos[:, :, :2]  # (N, N, 2)
            distances = np.linalg.norm(rel_positions, axis=2)  # (N, N)

        # =====================================================================
        # [Step 3] 유효한 이웃 마스크 생성
        # =====================================================================
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # =====================================================================
        # [Step 4] 각 에이전트에 대해 gazing preference 기반 이웃 선택
        # =====================================================================
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # 패딩 에이전트는 스킵

            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # 유효한 이웃이 없으면 스킵

            # ================================================================
            # [Step 4.1] 유효한 이웃 인덱스 추출
            # [논문 참조] Fig. 1(b): "The set N_i includes the neighbors of robot i
            # with which it interacts at any given time"
            # ================================================================
            valid_neighbor_indices = np.where(valid_mask_i)[0]

            # ================================================================
            # [Step 4.2] Gazing preference 방향 g_i 계산
            #
            # [논문 참조] Section II, Fig. 1(b):
            # "g_i does not necessarily align with the direction of the velocity
            # vector v_i. In other words, the angle θ^g between g_i and v_i may
            # not be zero."
            #
            # [구현] 환경에서 절대 속도/heading 정보가 없으므로, 로봇이 heading
            # 방향(= local coordinate frame의 +x 방향)으로 이동한다고 가정.
            #
            # 에이전트 i의 로컬 좌표계에서:
            # - velocity 방향 v_i = (1, 0) (로컬 +x 방향)
            # - gazing preference 방향 g_i = v_i를 θ^g만큼 회전
            #
            # [논문과의 차이] 논문은 글로벌 좌표계의 속도를 사용하지만,
            # 여기서는 로컬 좌표계를 사용. 상대 위치도 로컬 좌표계로 주어지므로
            # 알고리즘의 기하학적 관계는 동일하게 유지됨.
            # ================================================================
            g_i_x = np.cos(self.theta_g)
            g_i_y = np.sin(self.theta_g)
            g_i = np.array([g_i_x, g_i_y])  # gazing preference 방향 벡터

            # ================================================================
            # [Step 4.3] 각 이웃 j에 대해 φ_ij 계산
            #
            # [논문 참조] Eq. (6): "φ_is = ∠(r̃_s - r_i, g_i) ∈ [0, π)"
            # "represents the angle between the position of robot s, denoted by
            # r̃_s, and the gazing preference direction g_i"
            #
            # [수식 해석]
            # - r̃_s - r_i: 에이전트 i에서 이웃 s로의 방향 벡터
            # - g_i: gazing preference 방향 벡터
            # - φ_is: 두 벡터 사이의 각도 (0 ~ π)
            # ================================================================
            neighbor_directions = rel_positions[i, valid_neighbor_indices]  # (K, 2)

            # 이웃 방향 벡터의 단위 벡터 계산
            neighbor_distances = distances[i, valid_neighbor_indices]  # (K,)
            safe_distances = np.maximum(neighbor_distances, 1e-8)
            neighbor_unit_directions = neighbor_directions / safe_distances[:, np.newaxis]  # (K, 2)

            # φ_ij = arccos(g_i · neighbor_direction_unit)
            # [논문 참조] Eq. (6): φ_is ∈ [0, π)
            dot_products = np.sum(neighbor_unit_directions * g_i, axis=1)
            # 수치 안정성을 위해 클리핑
            dot_products = np.clip(dot_products, -1.0, 1.0)
            phi_values = np.arccos(dot_products)  # (K,), 범위: [0, π]

            # ================================================================
            # [Step 4.4] 선택 확률 p_i(j) 계산
            #
            # [논문 참조] Eq. (6):
            #   p_i(s) = (π - φ_is)^λ / Σ_{z∈N_i} (π - φ_iz)^λ
            #
            # "For a given λ, as φ_is decreases, the probability of selecting
            # robot s increases"
            #
            # [수식 해석]
            # - (π - φ_is): φ가 작을수록 값이 커짐 (gazing direction에 가까울수록 선호)
            # - λ: 값이 클수록 확률 분포가 더 뾰족해짐 (선호 방향에 더 집중)
            # ================================================================
            weights = np.power(np.pi - phi_values, self.lambda_coef)  # (K,)

            # ================================================================
            # [Step 4.5] 확률 정규화 및 이웃 선택
            #
            # [논문 참조] Eq. (6): 분모 Σ_{z∈N_i} (π - φ_iz)^λ로 정규화
            # ================================================================
            sum_weights = np.sum(weights)

            if sum_weights > 1e-8:
                # 정상적인 확률 분포로 선택
                probs = weights / sum_weights
                # ============================================================
                # [논문 참조] Section II, Eq. (5), (6):
                # 논문에서는 확률에 따라 단일 이웃 s를 선택하여 gazing preference
                # 힘을 적용함. 이 구현에서도 단일 이웃 선택.
                # ============================================================
                selected_idx = self.rng.choice(len(valid_neighbor_indices), p=probs)
                selected_neighbor = valid_neighbor_indices[selected_idx]
                action[i, selected_neighbor] = 1
            else:
                # 모든 가중치가 0에 가까운 경우 (모든 이웃이 gazing direction 반대편)
                # [구현 결정] 균등 확률로 랜덤 선택 (논문에서 이 경우 명시하지 않음)
                selected_neighbor = self.rng.choice(valid_neighbor_indices)
                action[i, selected_neighbor] = 1

        return action


class MotionSalienceThresholdSelection:
    """
    Motion Salience Threshold (MST) 기반 이웃 선택 베이스라인.

    ============================================================================
    논문 참조: "Tuning responsivity-persistence trade-off in swarm robotics:
               A motion salience threshold approach"
               (Li et al., Robotics and Autonomous Systems, 2025)
    ============================================================================

    이 베이스라인은 논문의 Section 2에서 제안된 MST 모델을 구현합니다.
    개체는 두 가지 상태(일반 조정 상태, 고정점 조정 상태) 사이를 전환하며,
    Motion Salience가 임계값 C를 초과하면 고정점 조정 상태로 전환됩니다.

    - 일반 조정 상태 (A_i = 0): 모든 이웃을 선택 (velocity averaging용)
    - 고정점 조정 상태 (A_i = 1): Motion Salience가 가장 높은 이웃만 선택

    이것은 STATEFUL 베이스라인으로, 시간 스텝 간 각 에이전트의 상태를 기억합니다.

    ============================================================================
    [환경 적응]
    obs에는 속도 정보가 없으므로 env.state에서 직접 접근:
    - env.state["agent_states"][:, 2:4] → 절대 속도 (vx, vy)
    ============================================================================
    """

    # =========================================================================
    # [논문 참조] Section 2.2, Figure 1
    # - STATE_GENERAL (A_i = 0): 일반 조정 상태 - 이웃들의 속도 평균화
    # - STATE_FIXED (A_i = 1): 고정점 조정 상태 - 특정 이웃의 속도만 따름
    # =========================================================================
    STATE_GENERAL = 0  # 일반 조정 상태 (General coordination state)
    STATE_FIXED = 1    # 고정점 조정 상태 (Fixed-point coordination state)

    def __init__(self,
                 motion_salience_threshold: float,
                 velocity_diff_threshold: float = 0.2,
                 k_neighbors: int = 7,
                 time_interval: float = 1.0,
                 scaling_factor: float = 100.0,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        MST 베이스라인 초기화.

        =========================================================================
        [논문 참조] Section 2.1 및 2.2
        =========================================================================

        Args:
            motion_salience_threshold (float):
                임계값 C - 상태 전환을 결정하는 Motion Salience 임계값
                [논문 참조] Section 2.2, Eq (2): "threshold C is established to
                determine the transition between these states"

            velocity_diff_threshold (float):
                δ (delta) - 고정점 상태에서 일반 상태로 복귀하기 위한 속도 차이 임계값
                [논문 참조] Section 2.2, Eq (2): "Should the velocity difference
                between individual i and j decrease below a predefined limit δ"
                [논문 참조] Section 3: "we use δ = 0.2"
                기본값: 0.2 (논문에서 사용한 값)

            k_neighbors (int):
                k - 위상적 이웃 수 (topological neighborhood size)
                [논문 참조] Section 2, paragraph 2: "Each individual is modeled to
                interact with its k nearest neighbors, with k = 7 selected for
                this study"
                기본값: 7 (논문에서 사용한 값)

            time_interval (float):
                τ (tau) - Motion Salience 계산을 위한 시간 간격
                [논문 참조] Section 2.1, Eq (1): "time interval [t-τ, t]"
                [논문 참조] Section 2.1: "with a time interval τ = 1 second"
                기본값: 1.0 (논문에서 사용한 값)

            scaling_factor (float):
                s - Motion Salience 값을 조정하기 위한 스케일링 팩터
                [논문 참조] Section 2.1, Eq (1): "The term s is a scaling factor"
                [논문 참조] Section 2.1: "we apply a scaling factor s = 100"
                기본값: 100.0 (논문에서 사용한 값)

            periodic_boundary (bool): 주기적 경계 사용 여부
            boundary_size (float, optional): 주기적 경계 크기
            seed (int, optional): 재현성을 위한 랜덤 시드
        """
        # =====================================================================
        # [파라미터 검증]
        # 논문에서 명시적으로 범위를 지정하지 않았지만, 물리적으로 의미 있는 범위로 제한
        # =====================================================================
        assert motion_salience_threshold >= 0, \
            "motion_salience_threshold (C) must be non-negative"
        assert velocity_diff_threshold > 0, \
            "velocity_diff_threshold (δ) must be positive"
        assert k_neighbors > 0, \
            "k_neighbors must be positive"
        assert time_interval > 0, \
            "time_interval (τ) must be positive"
        assert scaling_factor > 0, \
            "scaling_factor (s) must be positive"

        self.C = motion_salience_threshold   # 임계값 C
        self.delta = velocity_diff_threshold  # δ
        self.k = k_neighbors                  # k
        self.tau = time_interval              # τ
        self.s = scaling_factor               # s
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, \
                "boundary_size required for periodic boundaries"

        # =====================================================================
        # [상태 변수]
        # 논문에서는 각 개체가 상태를 유지한다고 명시함 (Figure 1, Section 2.2)
        # =====================================================================
        self.agent_states = None      # A_i(t): 각 에이전트의 현재 상태 (0 또는 1)
        self.target_neighbors = None  # j*: 고정점 상태에서 따르는 대상 이웃 인덱스
        self.prev_directions = None   # x̂^(t-τ)_ij: 이전 시간 스텝의 방향 벡터
        self.num_agents_max = None
        self.env = None  # Environment reference for accessing velocities

    def set_env(self, env):
        """
        환경 참조 설정 (속도 정보 접근용).

        Args:
            env: NeighborSelectionFlockingEnv 인스턴스
        """
        self.env = env

    def reset(self):
        """
        에피소드 시작 시 상태 초기화.

        Standard reset method with no parameters.

        =========================================================================
        [논문 참조] 논문에서 초기 상태에 대해 명시적으로 언급하지 않음.
        [구현 결정] 모든 에이전트를 일반 조정 상태(A_i = 0)로 초기화.
                   이는 시스템이 외부 자극 없이 시작한다는 가정에 기반함.
        =========================================================================
        """
        self.agent_states = None
        self.target_neighbors = None
        self.prev_directions = None
        self.num_agents_max = None

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        MST 기반 이웃 선택 액션 생성.

        =========================================================================
        [논문 참조] Section 2.1 및 2.2의 전체 알고리즘 흐름
        =========================================================================

        Args:
            obs: 환경 관측값 딕셔너리:
                - 'neighbor_masks': (num_agents_max, num_agents_max) boolean array
                - 'padding_mask': (num_agents_max,) boolean array
                - 'local_agent_infos': (num_agents_max, num_agents_max, obs_dim) array
                  obs_dim=4 (비주기적): [rel_x, rel_y, cos(rel_heading), sin(rel_heading)]

        Returns:
            action: (num_agents_max, num_agents_max) integer array
                - action[i, j] = 1: 에이전트 i가 에이전트 j를 이웃으로 선택
                - 일반 상태: k개의 가장 가까운 이웃 선택
                - 고정점 상태: Motion Salience가 가장 높은 이웃만 선택
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # =====================================================================
        # [환경 참조 확인]
        # =====================================================================
        if self.env is None:
            raise RuntimeError(
                "Must call set_env(env) before using this baseline. "
                "MST requires velocity information from env.state."
            )

        # =====================================================================
        # [상태 초기화] 환경 리셋 감지
        # Auto-detect episode reset - check both None and size change
        # =====================================================================
        if self.num_agents_max is None or self.num_agents_max != num_agents_max:
            # 모든 에이전트를 일반 조정 상태로 초기화
            self.agent_states = np.zeros(num_agents_max, dtype=np.int8)
            # 타겟 이웃은 -1로 초기화 (유효하지 않은 인덱스)
            self.target_neighbors = np.full(num_agents_max, -1, dtype=np.int32)
            self.prev_directions = None
            self.num_agents_max = num_agents_max

        # 액션 초기화 (self-loop 포함)
        action = np.eye(num_agents_max, dtype=np.int8)

        # =====================================================================
        # [Step 1] 상대 위치 및 방향 벡터 계산
        # [논문 참조] Section 2.1, Eq (1): "x̂^t_ij and x̂^(t-τ)_ij capture these
        #            directional changes"
        # =====================================================================
        if self.periodic_boundary:
            raise NotImplementedError(
                "Periodic boundary not yet implemented for MST baseline"
            )
        else:
            # 상대 위치 벡터: x_ij = pos_j - pos_i
            rel_positions = local_agent_infos[:, :, :2]  # (N, N, 2)

            # 거리 계산
            distances = np.linalg.norm(rel_positions, axis=2)  # (N, N)

            # ================================================================
            # [논문 참조] Section 2.1, Eq (1): 단위 벡터 x̂_ij 계산
            # "unit vectors x̂^t_ij and x̂^(t-τ)_ij capture these directional changes"
            # ================================================================
            safe_distances = np.maximum(distances, 1e-8)
            current_directions = rel_positions / safe_distances[:, :, np.newaxis]

        # =====================================================================
        # [Step 1.5] 속도 정보 가져오기 (env.state에서)
        # [환경 적응] obs에는 속도 정보가 없으므로 env.state에서 직접 접근
        # =====================================================================
        all_velocities = self.env.state["agent_states"][:, 2:4]  # (N, 2)

        # 각 에이전트 i에서 본 모든 에이전트 j의 속도
        # absolute_velocities[i, j] = v_j
        absolute_velocities = np.tile(all_velocities[np.newaxis, :, :], (num_agents_max, 1, 1))  # (N, N, 2)

        # 상대 속도: v_ij = v_j - v_i
        ego_velocities = all_velocities[:, np.newaxis, :]  # (N, 1, 2)
        rel_velocities = absolute_velocities - ego_velocities  # (N, N, 2)

        # =====================================================================
        # [Step 2] Motion Salience M_j 계산
        # [논문 참조] Section 2.1, Eq (1):
        #   M_j = (∠(x̂^t_ij, x̂^(t-τ)_ij) / τ) × s
        #
        # 여기서:
        # - ∠(a, b)는 두 단위 벡터 사이의 각도 (0 ~ π)
        # - τ는 시간 간격
        # - s는 스케일링 팩터
        # =====================================================================
        if self.prev_directions is None:
            # 첫 번째 스텝: Motion Salience 계산 불가
            # [구현 결정] 이전 방향 정보가 없으므로 M_j = 0으로 설정
            motion_salience = np.zeros((num_agents_max, num_agents_max))
        else:
            # 두 단위 벡터 사이의 각도 계산
            # cos(angle) = dot(x̂^t, x̂^(t-τ))
            dot_product = np.sum(
                current_directions * self.prev_directions, axis=2
            )
            # 수치 안정성을 위해 [-1, 1] 범위로 클리핑
            dot_product = np.clip(dot_product, -1.0, 1.0)

            # angle = arccos(dot_product), 범위: [0, π]
            angle_diff = np.arccos(dot_product)

            # ================================================================
            # [논문 참조] Section 2.1, Eq (1):
            #   M_j = (∠(x̂^t_ij, x̂^(t-τ)_ij) / τ) × s
            # ================================================================
            motion_salience = (angle_diff / self.tau) * self.s

        # =====================================================================
        # [Step 3] 유효한 이웃 마스크 생성
        # =====================================================================
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # =====================================================================
        # [Step 4] 위상적 이웃 (k-nearest neighbors) 결정
        # [논문 참조] Section 2, paragraph 2: "Each individual is modeled to
        #            interact with its k nearest neighbors"
        # [논문 참조] "Topological interactions involve selecting neighbors based
        #            on their positions within a network, focusing on a fixed
        #            number of nearest neighbors, regardless of physical distance"
        # =====================================================================
        k_nearest_masks = self._compute_k_nearest_neighbors(
            distances, valid_neighbors_no_self, padding_mask
        )

        # =====================================================================
        # [Step 5] 각 에이전트의 상태 전환 및 이웃 선택
        # [논문 참조] Section 2.2, Eq (2) 및 Eq (3)
        # =====================================================================
        new_states = self.agent_states.copy()
        new_targets = self.target_neighbors.copy()

        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # 패딩 에이전트는 스킵

            valid_mask_i = valid_neighbors_no_self[i]
            k_nearest_mask_i = k_nearest_masks[i]

            if not valid_mask_i.any():
                continue  # 유효한 이웃이 없으면 스킵

            # ================================================================
            # [논문 참조] Section 2, paragraph 2:
            # "we adopt a topological framework for defining the neighborhood
            #  N_i of agent i"
            # k개의 가장 가까운 이웃들의 인덱스
            # ================================================================
            topological_neighbors = np.where(k_nearest_mask_i)[0]

            if len(topological_neighbors) == 0:
                continue

            # ================================================================
            # [논문 참조] Section 2.1: "neighbor j" is in N_i
            # N_i 내에서의 Motion Salience 값들
            # ================================================================
            ms_in_neighborhood = motion_salience[i, topological_neighbors]
            max_ms = np.max(ms_in_neighborhood)
            j_star = topological_neighbors[np.argmax(ms_in_neighborhood)]

            current_state = self.agent_states[i]

            # ================================================================
            # [논문 참조] Section 2.2, Eq (2): 상태 전환 로직
            #
            # A_i(t+1) =
            #   1  if max_{j∈N_i} M_j > C ∧ A_i(t) = 0
            #   0  if |v_i - v_j*| < δ ∧ A_i(t) = 1
            #   A_i(t)  otherwise
            # ================================================================

            if current_state == self.STATE_GENERAL:
                # 일반 상태 → 고정점 상태 전환 조건
                # [논문 참조] Eq (2), 첫 번째 조건:
                # "if max_{j∈N_i} M_j > C ∧ A_i(t) = 0"
                if max_ms > self.C:
                    new_states[i] = self.STATE_FIXED
                    # [논문 참조] Section 2.2:
                    # "alignment of velocities with the neighbor possessing
                    #  the highest M_j, identified as j* = arg max(M_j)"
                    new_targets[i] = j_star

            elif current_state == self.STATE_FIXED:
                # 고정점 상태 → 일반 상태 복귀 조건
                # [논문 참조] Eq (2), 두 번째 조건:
                # "if |v_i - v_j*| < δ ∧ A_i(t) = 1"

                j_target = self.target_neighbors[i]

                # 타겟 이웃이 여전히 유효한지 확인
                if j_target < 0 or not valid_mask_i[j_target]:
                    # [구현 결정] 타겟이 유효하지 않으면 일반 상태로 복귀
                    # 논문에서 이 경우를 명시적으로 다루지 않음
                    new_states[i] = self.STATE_GENERAL
                    new_targets[i] = -1
                else:
                    # ========================================================
                    # 속도 차이 계산
                    # [논문 참조] Section 2.2: "|v_i - v_j*| < δ"
                    #
                    # rel_velocities[i, j] = v_j - v_i 이므로
                    # |v_i - v_j*| = |v_j* - v_i| = |rel_velocities[i, j*]|
                    # ========================================================
                    velocity_diff = np.linalg.norm(rel_velocities[i, j_target])

                    # [논문 참조] Section 3: "we use δ = 0.2"
                    # 속도 차이가 임계값보다 작으면 일반 상태로 복귀
                    if velocity_diff < self.delta:
                        new_states[i] = self.STATE_GENERAL
                        new_targets[i] = -1

            # ================================================================
            # [논문 참조] Section 2.2, Eq (3): 상태에 따른 이웃 선택
            #
            # v^align_i =
            #   (1/|N_i|) * Σ_{j∈N_i} v_j    if A_i(t) = 0 (일반 상태)
            #   v_j*                          if A_i(t) = 1 (고정점 상태)
            #
            # [구현 해석] 이 논문은 속도 정렬(velocity alignment)을 다루지만,
            # 현재 환경에서는 "이웃 선택"을 액션으로 반환해야 함.
            #
            # 일반 상태: 모든 k-nearest 이웃 선택 (velocity averaging에 해당)
            # 고정점 상태: j* 이웃만 선택 (single neighbor following에 해당)
            # ================================================================

            if new_states[i] == self.STATE_GENERAL:
                # [논문 참조] Eq (3), 첫 번째 경우:
                # "the individual averages the velocities of all neighboring
                #  individuals"
                # → k개의 위상적 이웃 모두 선택
                action[i, topological_neighbors] = 1

            else:  # STATE_FIXED
                # [논문 참조] Eq (3), 두 번째 경우:
                # "the individual follows the velocity of a specific individual"
                # → j* 이웃만 선택
                target = new_targets[i]
                if target >= 0:
                    action[i, target] = 1

        # =====================================================================
        # [Step 6] 상태 업데이트
        # =====================================================================
        self.agent_states = new_states
        self.target_neighbors = new_targets
        self.prev_directions = current_directions.copy()

        return action

    def _compute_k_nearest_neighbors(
        self,
        distances: np.ndarray,
        valid_mask: np.ndarray,
        padding_mask: np.ndarray
    ) -> np.ndarray:
        """
        각 에이전트에 대해 k개의 가장 가까운 이웃을 계산.

        =========================================================================
        [논문 참조] Section 2, paragraph 2:
        "Each individual is modeled to interact with its k nearest neighbors,
         with k = 7 selected for this study"
        =========================================================================

        Args:
            distances: (N, N) 거리 행렬
            valid_mask: (N, N) 유효한 이웃 마스크 (self-loop 제외)
            padding_mask: (N,) 패딩 마스크

        Returns:
            k_nearest_masks: (N, N) k-nearest 이웃 마스크
        """
        num_agents_max = distances.shape[0]
        k_nearest_masks = np.zeros((num_agents_max, num_agents_max), dtype=bool)

        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue

            valid_mask_i = valid_mask[i]
            if not valid_mask_i.any():
                continue

            distances_i = distances[i].copy()
            distances_i[~valid_mask_i] = np.inf

            num_valid = valid_mask_i.sum()
            k_actual = min(self.k, num_valid)

            if k_actual > 0:
                # argpartition으로 효율적인 k-nearest 찾기
                nearest_indices = np.argpartition(distances_i, k_actual - 1)[:k_actual]
                k_nearest_masks[i, nearest_indices] = True

        return k_nearest_masks


class VisualAttentionNeighborSelection:
    """
    Strict implementation of Visual Attention-Based Neighbor Selection (DLN-S).

    Based on: "Emergence of Collective Behaviors for the Swarm Robotics Through
    Visual Attention-Based Selective Interaction" (Zheng et al., 2024)

    This baseline strictly follows the literal notation of Equations (1), (2), (129), (132), and (135).
    Normalization of vectors is NOT applied as it is not explicitly denoted in Eq (135).

    FIXED: Velocity vector estimation from position differences (per paper).
    - Estimates velocities from position changes: v_j ≈ (pos_j(t) - pos_j(t-τ)) / τ
    - No env.state access needed (position-based only)
    - Correctly implements Eq. 135: chi = (1 + v_j · x_ij) / 2
    """

    def __init__(self, selection_preference: float = 1.0,
                 robot_radius: float = 1.0,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        assert selection_preference > 0, "alpha must be positive"
        assert robot_radius > 0, "radius must be positive"

        self.alpha = selection_preference
        self.r = robot_radius
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

        self.prev_visual_angles = None
        self.prev_chis = None
        self.prev_rel_positions = None
        self.prev_positions = None  # For velocity estimation
        self.num_agents_max = None

    def reset(self):
        """
        Reset internal state for new episode.

        Standard reset method with no parameters.
        Can be called explicitly or auto-triggered when episode reset detected.
        """
        self.prev_visual_angles = None
        self.prev_chis = None
        self.prev_rel_positions = None
        self.prev_positions = None
        self.num_agents_max = None

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Auto-detect episode reset
        if self.num_agents_max is None or self.num_agents_max != num_agents_max:
            self.reset()
            self.num_agents_max = num_agents_max

        action = np.eye(num_agents_max, dtype=np.int8)

        # --- 1. Compute Distances ---
        if self.periodic_boundary:
             raise NotImplementedError("Strict DLN for Periodic boundary not implemented")
        else:
            # Relative positions x_ij(t) = pos_j - pos_i (Raw Vector)
            rel_positions = local_agent_infos[:, :, :2]
            distances = np.linalg.norm(rel_positions, axis=2)

        safe_distances = np.maximum(distances, 1e-6)

        # --- 2. Calculate Visual Projection theta (Eq. 132) ---
        # theta = 2 * arctan(r / d)
        current_thetas = 2 * np.arctan(self.r / safe_distances)

        # --- 3. Calculate Departing Desire chi (Eq. 135) ---
        # Equation 135: chi = (1 + v_j . x_ij) / 2
        # Use raw vectors x_ij and v_j directly as per literal notation.

        # Estimate velocities from position differences (per paper Section II-A)
        # "we estimate the velocity of neighbor j based on the position differences"
        if self.prev_positions is not None and self.prev_rel_positions is not None:
            # Reconstruct absolute positions from relative positions
            # Since rel_positions[i,j] are normalized, we need to denormalize first
            # But for velocity estimation, we can work with position differences directly

            # Estimate velocity: v_j ≈ (pos_j(t) - pos_j(t-1)) / dt
            # We approximate this using changes in relative positions
            # delta_rel_pos[i,j] = rel_pos[i,j](t) - rel_pos[i,j](t-1)
            delta_rel_positions = rel_positions - self.prev_rel_positions  # (N, N, 2)

            # For velocity estimation, assume dt = 1 time step
            estimated_velocities = delta_rel_positions  # (N, N, 2)

            # Dot product: v_j . x_ij (Raw Dot Product)
            # No normalization applied as per Eq. 135 literal notation.
            alignment = np.sum(estimated_velocities * rel_positions, axis=2)
        else:
            # First step: no velocity estimate available, use zero alignment
            alignment = np.zeros((num_agents_max, num_agents_max))

        # Eq 135: chi = (1 + alignment) / 2
        current_chis = (1.0 + alignment) / 2.0

        # --- 4. Logic Branch based on History ---
        if self.prev_visual_angles is None:
            self._update_state(current_thetas, current_chis, rel_positions)
            return self._fallback_random_selection(num_agents_max, neighbor_masks, padding_mask)

        # --- 5. Calculate Departing Tendency theta_dot (Eq. 129) ---
        theta_diff = self.prev_visual_angles - current_thetas
        safe_prev_theta = np.maximum(self.prev_visual_angles, 1e-6)
        theta_dot = theta_diff / safe_prev_theta

        # --- 6. Calculate Final DLN c_ij (Eq. 1) ---
        # c_ij = theta_dot * chi(t-tau) * chi(t)
        dln_scores = theta_dot * self.prev_chis * current_chis
        dln_scores = np.maximum(0.0, dln_scores)

        # --- 7. Probabilistic Selection (Eq. 2) ---
        # p ~ (c_ij)^alpha
        # Handle potential large values from raw dot product
        # (Though proportional selection handles scale, avoid overflow if needed)
        weights = np.power(dln_scores, self.alpha)

        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        for i in range(num_agents_max):
            if not padding_mask[i]: continue

            valid_mask_i = valid_neighbors[i]
            if not valid_mask_i.any(): continue

            weights_i = weights[i].copy()
            weights_i[~valid_mask_i] = 0.0

            sum_w = np.sum(weights_i)

            if sum_w > 1e-8:
                probs = weights_i / sum_w
                candidates = np.arange(num_agents_max)
                selected = self.rng.choice(candidates, p=probs)
                action[i, selected] = 1
            else:
                candidates = np.where(valid_mask_i)[0]
                if len(candidates) > 0:
                    selected = self.rng.choice(candidates)
                    action[i, selected] = 1

        # --- 8. Update State ---
        self._update_state(current_thetas, current_chis, rel_positions)

        return action

    def _update_state(self, thetas, chis, rel_pos):
        self.prev_visual_angles = thetas
        self.prev_chis = chis
        self.prev_rel_positions = rel_pos

    def _fallback_random_selection(self, num_agents_max, neighbor_masks, padding_mask):
        action = np.eye(num_agents_max, dtype=np.int8)
        valid = neighbor_masks & (padding_mask[:, None] & padding_mask[None, :])
        np.fill_diagonal(valid, False)

        for i in range(num_agents_max):
            if not padding_mask[i]: continue
            candidates = np.where(valid[i])[0]
            if len(candidates) > 0:
                choice = self.rng.choice(candidates)
                action[i, choice] = 1
        return action


# Convenience function to create baselines
def create_baseline(baseline_type: str, **kwargs):
    """
    Factory function to create baseline policies.

    Args:
        baseline_type: One of 'random', 'distance', 'nearest', 'farthest', 'mti',
                      'voronoi', 'highest_degree', 'mfnn', 'visual_attention',
                      'active_search', 'gazing_preference', 'motion_salience'
        **kwargs: Arguments to pass to the baseline constructor

    Returns:
        Baseline policy object

    Example:
        >>> baseline = create_baseline('random', selection_probability=0.5, seed=42)
        >>> baseline = create_baseline('nearest', k=5)
        >>> baseline = create_baseline('mti', k=5, distance_threshold=0.5,
        ...                           threshold_a=0.1, threshold_b=0.5, seed=42)
        >>> baseline = create_baseline('highest_degree', beta=5)
        >>> baseline = create_baseline('mfnn', k=6)
        >>> baseline = create_baseline('gazing_preference', theta_g=np.pi/3, lambda_coef=3.0)
        >>> baseline = create_baseline('motion_salience', motion_salience_threshold=1.0, k_neighbors=7)
    """
    baselines = {
        'random': RandomNeighborSelection,
        'distance': DistanceBasedNeighborSelection,
        'nearest': FixedNearestNeighborSelection,
        'farthest': FixedFarthestNeighborSelection,
        'mti': MetricTopologicalInteractionSelection,
        'voronoi': VoronoiNeighborSelection,
        'highest_degree': HighestDegreeNeighborSelection,
        'mfnn': ModifiedFixedNumberNeighbors,
        'visual_attention': VisualAttentionNeighborSelection,
        'active_search': ActiveSearchNeighborSelection,
        'gazing_preference': GazingPreferenceNeighborSelection,
        'motion_salience': MotionSalienceThresholdSelection,
    }

    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Must be one of {list(baselines.keys())}")

    return baselines[baseline_type](**kwargs)
