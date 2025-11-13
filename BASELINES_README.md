# Heuristic Baseline Neighbor Selection Strategies

This directory contains implementations of heuristic baseline neighbor selection strategies for the flocking environment.

## Overview

The baseline heuristics provide simple, interpretable policies for neighbor selection that can be used as:
- Performance baselines for evaluating RL agents
- Ablation study components
- Quick prototyping tools
- Sanity checks for environment implementation

## Available Baselines

### 1. Random Neighbor Selection

Randomly selects neighbors based on a specified probability.

**Parameters:**
- `selection_probability` (float): Probability of selecting each valid neighbor (default: 0.5)
- `seed` (int, optional): Random seed for reproducibility

**Usage:**
```python
from baselines import RandomNeighborSelection

baseline = RandomNeighborSelection(selection_probability=0.5, seed=42)
action = baseline(obs)
```

**Characteristics:**
- Simple and unbiased
- No distance information required
- Suitable for testing environment dynamics with random connectivity

### 2. Distance-Based Neighbor Selection

Selects neighbors whose pairwise distance is below a given threshold.

**Parameters:**
- `distance_threshold` (float): Maximum distance for neighbor selection
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary (required if `periodic_boundary=True`)

**Usage:**
```python
from baselines import DistanceBasedNeighborSelection

baseline = DistanceBasedNeighborSelection(
    distance_threshold=0.5,  # Note: in normalized units
    periodic_boundary=False
)
action = baseline(obs)
```

**Characteristics:**
- Proximity-based selection
- Maintains local connectivity
- Threshold should be specified in normalized units (positions are normalized by `initial_position_bound / 2`)

**Note:** Currently supports non-periodic boundaries only. Periodic boundary support is planned for future updates.

### 3. Fixed-Nearest Neighbor Selection

Selects the k nearest valid neighbors for each agent.

**Parameters:**
- `k` (int): Number of nearest neighbors to select
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary

**Usage:**
```python
from baselines import FixedNearestNeighborSelection

baseline = FixedNearestNeighborSelection(k=5, periodic_boundary=False)
action = baseline(obs)
```

**Characteristics:**
- Fixed connectivity degree per agent
- Promotes local clustering
- Gracefully handles cases where an agent has fewer than k valid neighbors (selects all available)

### 4. Fixed-Farthest Neighbor Selection

Selects the k farthest valid neighbors for each agent.

**Parameters:**
- `k` (int): Number of farthest neighbors to select
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary

**Usage:**
```python
from baselines import FixedFarthestNeighborSelection

baseline = FixedFarthestNeighborSelection(k=5, periodic_boundary=False)
action = baseline(obs)
```

**Characteristics:**
- Anti-clustering behavior
- Encourages long-range connections
- Useful for studying exploration vs. exploitation in flocking

## Factory Function

A convenience factory function is provided to create baselines:

```python
from baselines import create_baseline

# Create random baseline
baseline = create_baseline('random', selection_probability=0.5, seed=42)

# Create distance-based baseline
baseline = create_baseline('distance', distance_threshold=0.5, periodic_boundary=False)

# Create k-nearest baseline
baseline = create_baseline('nearest', k=5, periodic_boundary=False)

# Create k-farthest baseline
baseline = create_baseline('farthest', k=3, periodic_boundary=False)
```

## Complete Example

Here's a complete example of using a baseline with the flocking environment:

```python
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config, load_dict
from baselines import FixedNearestNeighborSelection

# Load environment configuration
config_dict = load_dict('envs/default_env_config.yaml')
config = Config(**config_dict)
env_context = config_to_env_input(config, seed_id=42)

# Create environment
env = NeighborSelectionFlockingEnv(env_context)

# Create baseline policy
baseline = FixedNearestNeighborSelection(k=5, periodic_boundary=False)

# Run episode
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = baseline(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
```

## Implementation Details

### Observation Format

All baselines expect observations in the standard environment format:

```python
obs = {
    'local_agent_infos': np.ndarray,  # Shape: (num_agents_max, num_agents_max, obs_dim)
                                       # Contains relative positions and headings
    'neighbor_masks': np.ndarray,      # Shape: (num_agents_max, num_agents_max)
                                       # Boolean array indicating valid neighbors
    'padding_mask': np.ndarray,        # Shape: (num_agents_max,)
                                       # Boolean array indicating real vs padding agents
    'is_from_my_env': np.bool_
}
```

### Action Format

All baselines return actions in the format expected by the environment:

- **Shape:** `(num_agents_max, num_agents_max)`
- **Data type:** `np.int8` (or compatible integer type)
- **Semantics:** `action[i, j] = 1` means agent `i` selects agent `j` as a neighbor
- **Constraints:**
  - Self-loops required: `action[i, i] = 1` for all `i`
  - Respects neighbor masks: `action[i, j] = 1` only if `neighbor_masks[i, j] = True`
  - Respects padding masks: `action[i, j] = 1` only if both `padding_mask[i]` and `padding_mask[j]` are `True`

### Handling Edge Cases

All baselines correctly handle:

1. **Padding agents:** Actions for padding agents are set appropriately (self-loops only)
2. **Limited neighbors:** If an agent has fewer than k valid neighbors, all available neighbors are selected
3. **Self-loops:** Always maintained as required by the environment
4. **Neighbor mask constraints:** Only neighbors within the communication range (as defined by `neighbor_masks`) can be selected

### Distance Computation

For distance-based baselines (distance, nearest, farthest):

- Distances are computed from the relative positions in `local_agent_infos[:, :, :2]`
- For **non-periodic boundaries**: Positions are normalized by `initial_position_bound / 2`
  - Distance thresholds should be specified in these normalized units
  - Example: If `initial_position_bound = 250`, a physical distance of 125 corresponds to normalized distance of 1.0
- For **periodic boundaries**: Not yet implemented (planned for future updates)

### Efficiency Considerations

- All implementations use vectorized NumPy operations where possible
- k-nearest and k-farthest use `np.argpartition` for O(n) selection instead of full sorting
- Mask operations are performed using boolean indexing for efficiency

## Testing

To verify the baselines work correctly, run:

```bash
python test_baselines.py
```

This will test all four baselines with the flocking environment and verify:
- Actions have correct format
- Actions respect neighbor masks and padding masks
- Self-loops are correctly maintained
- Baselines can run for multiple steps without errors

## Known Limitations

1. **Periodic boundaries:** Distance computation for periodic boundaries is not yet implemented for distance-based, nearest, and farthest baselines
2. **Distance normalization:** Distance thresholds must be specified in normalized units (divided by `initial_position_bound / 2`)
3. **Static policies:** These are heuristic policies that don't learn or adapt over time

## Future Enhancements

Potential improvements for future versions:

1. Support for periodic boundary distance computation
2. Adaptive k-nearest (variable k based on local density)
3. Hybrid strategies (e.g., nearest within a distance threshold)
4. Probabilistic k-nearest (sample k neighbors from distance distribution)
5. Velocity-aware selection (consider relative velocities, not just positions)
6. Dynamic distance thresholds based on environment state

## Citation

If you use these baselines in your research, please cite the repository and mention the specific baseline used.
