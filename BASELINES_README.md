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

### 5. Metric-Topological Interaction (MTI) Selection

Adaptive neighbor selection based on local heading alignment, implementing the model from:

**"Metric–topological interaction model of collective behavior"** (Niizato & Gunji, 2011)

Each agent maintains an internal mode (TOP or MET) that switches based on heading alignment with neighbors. This creates context-aware neighbor selection that adapts to local swarm dynamics.

**Parameters:**
- `k` (int): Number of nearest neighbors in TOP (topological) mode
- `distance_threshold` (float): Distance threshold R for MET (metric) mode (normalized units)
- `threshold_a` (float): TOP→MET switch threshold in radians (alignment threshold)
- `threshold_b` (float): MET→TOP switch threshold in radians (dispersion threshold)
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary
- `seed` (int, optional): Random seed for reproducibility in MET mode sampling

**Usage:**
```python
from baselines import MetricTopologicalInteractionSelection

baseline = MetricTopologicalInteractionSelection(
    k=5,                      # Select 5 nearest neighbors in TOP mode
    distance_threshold=0.5,   # Select neighbors within 0.5 in MET mode
    threshold_a=0.1,          # Switch to MET when aligned (< 0.1 rad difference)
    threshold_b=0.5,          # Switch to TOP when dispersed (> 0.5 rad difference)
    seed=42
)
action = baseline(obs)

# For episode resets (optional, auto-detected):
baseline.reset()
```

**Characteristics:**
- **Stateful**: Maintains each agent's mode (TOP/MET) across time steps
- **Adaptive**: Switches modes based on local heading alignment
- **TOP mode**: Selects k nearest neighbors (promotes local clustering)
- **MET mode**: Selects all neighbors within distance threshold (promotes cohesion)

**Mode Switching Logic:**
- **TOP → MET**: When agent's heading is well-aligned with its k nearest neighbors (heading difference ≤ threshold_a)
- **MET → TOP**: When two randomly sampled neighbors have large heading difference (> threshold_b), indicating local dispersion

**Implementation Notes:**
- Heading information is extracted from relative heading observations: `[cos(θ_j - θ_i), sin(θ_j - θ_i)]`
- Circular mean is used for averaging heading angles
- Automatically detects episode resets when `num_agents_max` changes
- Handles edge cases: padding agents, insufficient neighbors, etc.

**Use Cases:**
- Studying emergent adaptive behavior in flocking
- Comparing context-aware vs. static neighbor selection
- Understanding the role of local alignment information in collective behavior

### 6. Highest-Degree Neighbor Selection

Selects neighbors with the highest degree (most connections), based on:

**"Research on swarm consistent performance of improved Vicsek model with neighbors' degree"**

This baseline prioritizes well-connected neighbors, with the intuition that agents with more connections may be more informative for achieving consensus or alignment in the swarm.

**Parameters:**
- `beta` (int): Number of highest-degree neighbors to select
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary

**Usage:**
```python
from baselines import HighestDegreeNeighborSelection

baseline = HighestDegreeNeighborSelection(beta=5, periodic_boundary=False)
action = baseline(obs)
```

**Characteristics:**
- **Degree-aware**: Selects neighbors based on their connectivity
- **Implicit information flow**: Well-connected neighbors may have access to more swarm information
- **Tie-breaking**: When multiple neighbors have the same degree, nearest neighbors are preferred

**Implementation Details:**
- Degree is computed as the total number of valid neighbors each agent has
- Neighbors are sorted by degree (descending), then by distance (ascending) for tie-breaking
- Handles cases where fewer than beta neighbors are available (selects all)
- Padding agents are excluded from degree calculations

**Use Cases:**
- Studying the role of network topology in collective behavior
- Comparing degree-based vs. distance-based selection
- Investigating consensus formation through well-connected agents

### 7. Modified Fixed Number of Neighbors (MFNN)

Spatially-distributed neighbor selection with angular sectors, based on:

**"Enhancing synchronization of self-propelled particles via modified rule of fixed number of neighbors"**

This baseline divides the space around each agent into (k-1) equal angular sectors and selects the nearest neighbor in each sector. This ensures spatial diversity in neighbor selection, preventing clustering of selected neighbors in one direction.

**Parameters:**
- `k` (int): Maximum number of neighbors to select (space divided into k-1 sectors)
- `periodic_boundary` (bool): Whether the environment uses periodic boundaries (default: False)
- `boundary_size` (float, optional): Size of periodic boundary

**Usage:**
```python
from baselines import ModifiedFixedNumberNeighbors

baseline = ModifiedFixedNumberNeighbors(k=6, periodic_boundary=False)
# This creates 5 angular sectors and selects nearest neighbor in each
action = baseline(obs)
```

**Characteristics:**
- **Spatial diversity**: Ensures neighbors are distributed across different directions
- **Angular partitioning**: Divides 360° space into equal sectors
- **Distance-based within sectors**: Selects nearest neighbor in each sector
- **Adaptive selection**: May select fewer than k-1 neighbors if some sectors are empty

**Implementation Details:**
- Space is divided into k-1 equal angular sectors spanning [-π, π]
- For each sector, the nearest neighbor (by Euclidean distance) is selected
- Sectors are defined in the agent's local coordinate frame
- Empty sectors (no neighbors) are skipped
- Handles edge cases: fewer neighbors than sectors, padding agents

**Use Cases:**
- Ensuring balanced spatial coverage in neighbor selection
- Preventing over-clustering in one direction
- Studying the effect of spatial diversity on synchronization
- Comparing with standard k-nearest which may cluster neighbors

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

# Create MTI baseline
baseline = create_baseline('mti', k=5, distance_threshold=0.5,
                          threshold_a=0.1, threshold_b=0.5, seed=42)

# Create highest-degree baseline
baseline = create_baseline('highest_degree', beta=5)

# Create MFNN baseline
baseline = create_baseline('mfnn', k=6)
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
