# Guide for Heuristic Baseline Developers

This guide explains how to add new heuristic neighbor selection baselines to this repository. It covers the required interfaces, data formats, common pitfalls, and testing procedures.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Required Interface](#required-interface)
3. [Understanding Observations](#understanding-observations)
4. [Action Format Requirements](#action-format-requirements)
5. [Handling Masks](#handling-masks)
6. [Distance and Heading Extraction](#distance-and-heading-extraction)
7. [Common Pitfalls](#common-pitfalls)
8. [Testing Your Baseline](#testing-your-baseline)
9. [Adding to Factory Function](#adding-to-factory-function)
10. [Template Skeleton](#template-skeleton)

## Quick Start

To add a new baseline:

1. Create a new class in `baselines.py`
2. Implement `__init__` and `__call__` methods
3. Add to the `create_baseline` factory function
4. Update `BASELINES_README.md` with documentation
5. Add test case in `test_baselines.py`
6. Run tests to verify correctness

## Required Interface

All baseline classes must implement these methods:

### `__init__(self, ...)`

Initialize your baseline with any parameters needed.

**Requirements:**
- Accept parameters for your selection strategy
- Validate parameters (use assertions)
- Initialize random number generator if needed (use `np.random.RandomState(seed)`)
- Handle periodic boundary settings if applicable

**Example:**
```python
def __init__(self, threshold: float, seed: Optional[int] = None):
    assert threshold > 0, "threshold must be positive"
    self.threshold = threshold
    self.rng = np.random.RandomState(seed)
```

### `__call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray`

Generate neighbor selection action from observation.

**Input:** Observation dictionary (see [Understanding Observations](#understanding-observations))

**Output:** Action array of shape `(num_agents_max, num_agents_max)` with dtype `np.int8` or compatible integer type

**Requirements:**
- Must respect `neighbor_masks` and `padding_mask`
- Must set diagonal to 1 (self-loops required)
- Must handle all edge cases gracefully

### `reset(self)` (Optional)

For stateful baselines, implement a reset method to reinitialize state between episodes.

**Example:**
```python
def reset(self):
    self.internal_state = self._initialize_state()
```

## Understanding Observations

The observation dictionary contains:

```python
obs = {
    'local_agent_infos': np.ndarray,  # Shape: (num_agents_max, num_agents_max, obs_dim)
    'neighbor_masks': np.ndarray,      # Shape: (num_agents_max, num_agents_max), dtype: bool
    'padding_mask': np.ndarray,        # Shape: (num_agents_max,), dtype: bool
    'is_from_my_env': np.bool_         # Scalar boolean
}
```

### `local_agent_infos`

Contains relative information between agents. Shape: `(num_agents_max, num_agents_max, obs_dim)`

**For non-periodic boundaries** (`obs_dim=4`):
- `local_agent_infos[i, j, 0:2]`: Normalized relative position of agent j from agent i's perspective
  - Normalized by `initial_position_bound / 2`
  - `rel_pos[i, j] = (pos[j] - pos[i]) / (initial_position_bound / 2)`
- `local_agent_infos[i, j, 2]`: `cos(heading_j - heading_i)` (relative heading cosine)
- `local_agent_infos[i, j, 3]`: `sin(heading_j - heading_i)` (relative heading sine)

**For periodic boundaries** (`obs_dim=6`):
- `local_agent_infos[i, j, 0:4]`: `[cos(x), sin(x), cos(y), sin(y)]` (sin/cos transformed positions)
- `local_agent_infos[i, j, 4:6]`: Relative heading `[cos(heading_j - heading_i), sin(heading_j - heading_i)]`

**Important:** For padding agents (where `padding_mask[i]` or `padding_mask[j]` is False), the values are zeros.

### `neighbor_masks`

Boolean array indicating valid communication links. Shape: `(num_agents_max, num_agents_max)`

- `neighbor_masks[i, j] = True`: Agent j is within communication range of agent i
- `neighbor_masks[i, j] = False`: Agent j is NOT a valid neighbor of agent i
- **Always includes self-loops:** `neighbor_masks[i, i] = True` for active agents
- Already accounts for communication range limits

**Key constraint:** Your action can ONLY select neighbors where `neighbor_masks[i, j] = True`.

### `padding_mask`

Boolean array indicating real vs. padding agents. Shape: `(num_agents_max,)`

- `padding_mask[i] = True`: Agent i is a real agent
- `padding_mask[i] = False`: Agent i is a padding agent (inactive)

**Key constraint:** Your action must only select real agents (both source and target must have `padding_mask = True`).

## Action Format Requirements

### Shape and Type
- **Shape:** `(num_agents_max, num_agents_max)`
- **Data type:** `np.int8` (or compatible integer type)
- **Semantics:** `action[i, j] = 1` means agent i selects agent j as a neighbor

### Required Constraints

1. **Self-loops:** `action[i, i] = 1` for all `i`
   ```python
   np.fill_diagonal(action, 1)
   ```

2. **Respect neighbor_masks:** `action[i, j] = 1` only if `neighbor_masks[i, j] = True`
   ```python
   assert np.all((neighbor_masks | ~action)), "Action violates neighbor_masks"
   ```

3. **Respect padding_mask:** `action[i, j] = 1` only if both `padding_mask[i]` and `padding_mask[j]` are True
   ```python
   padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
   assert np.all((padding_mask_2d | ~action)), "Action violates padding_mask"
   ```

## Handling Masks

### Creating Valid Neighbor Mask

Always start by creating a valid neighbor mask that combines both constraints:

```python
# Valid neighbors: in communication range AND both agents are real
padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
valid_neighbors = neighbor_masks & padding_mask_2d

# For selection (excluding self-loops)
valid_neighbors_no_self = valid_neighbors.copy()
np.fill_diagonal(valid_neighbors_no_self, False)
```

### Processing Each Agent

When implementing per-agent logic:

```python
for i in range(num_agents_max):
    if not padding_mask[i]:
        continue  # Skip padding agents

    valid_mask_i = valid_neighbors_no_self[i]

    if not valid_mask_i.any():
        continue  # No valid neighbors, keep self-loop only

    # Your selection logic here
    selected_indices = your_selection_function(i, valid_mask_i)
    action[i, selected_indices] = 1
```

## Distance and Heading Extraction

### Computing Distances

For non-periodic boundaries:

```python
# Extract relative positions (normalized)
rel_positions = local_agent_infos[:, :, :2]  # (num_agents_max, num_agents_max, 2)

# Compute pairwise distances (in normalized units)
distances = np.linalg.norm(rel_positions, axis=2)  # (num_agents_max, num_agents_max)
```

**Note:** Distances are in normalized units (divided by `initial_position_bound / 2`). If you need to compare with physical distances, adjust your thresholds accordingly.

### Extracting Relative Headings

```python
# Extract cos and sin of relative headings
rel_heading_cos = local_agent_infos[:, :, 2]  # cos(theta_j - theta_i)
rel_heading_sin = local_agent_infos[:, :, 3]  # sin(theta_j - theta_i)

# Compute relative heading angles
rel_headings = np.arctan2(rel_heading_sin, rel_heading_cos)  # (num_agents_max, num_agents_max)
```

### Computing Circular Mean of Headings

When averaging angles, use circular statistics:

```python
# For a set of relative headings
neighbor_rel_headings = rel_headings[agent_i, selected_neighbors]

# Circular mean
mean_cos = np.mean(np.cos(neighbor_rel_headings))
mean_sin = np.mean(np.sin(neighbor_rel_headings))
mean_heading = np.arctan2(mean_sin, mean_cos)
```

### Wrapping Angles to [-π, π]

Always wrap angle differences:

```python
def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

# Usage
angle_diff = theta1 - theta2
angle_diff = wrap_to_pi(angle_diff)
```

## Common Pitfalls

### 1. Not Setting Diagonal to 1

**Problem:**
```python
action = np.zeros((num_agents_max, num_agents_max), dtype=np.int8)
# Forgot to set diagonal!
```

**Solution:**
```python
action = np.eye(num_agents_max, dtype=np.int8)  # Start with identity
# OR
action = np.zeros((num_agents_max, num_agents_max), dtype=np.int8)
np.fill_diagonal(action, 1)
```

### 2. Violating neighbor_masks

**Problem:**
```python
# Selecting neighbors without checking neighbor_masks
action[i, j] = 1  # j might not be in neighbor_masks[i]!
```

**Solution:**
```python
# Always filter with valid_neighbors
within_threshold = (distances[i] <= threshold) & valid_neighbors[i]
action[i, within_threshold] = 1
```

### 3. Not Handling Padding Agents

**Problem:**
```python
for i in range(num_agents_max):
    # Forgot to check padding_mask[i]
    selected = select_neighbors(i)
    action[i, selected] = 1
```

**Solution:**
```python
for i in range(num_agents_max):
    if not padding_mask[i]:
        continue  # Skip padding agents
    selected = select_neighbors(i)
    action[i, selected] = 1
```

### 4. Incorrect dtype or shape

**Problem:**
```python
action = np.zeros((num_agents, num_agents))  # Wrong shape! Should be num_agents_max
# OR
action = np.zeros((num_agents_max, num_agents_max), dtype=np.float32)  # Wrong dtype!
```

**Solution:**
```python
action = np.zeros((num_agents_max, num_agents_max), dtype=np.int8)
```

### 5. Not Handling Empty Neighbor Sets

**Problem:**
```python
selected_neighbors = select_k_nearest(i, distances, valid_mask_i)
# What if valid_mask_i has no True values?
action[i, selected_neighbors] = 1  # Might fail!
```

**Solution:**
```python
if not valid_mask_i.any():
    continue  # No valid neighbors, keep self-loop only

selected_neighbors = select_k_nearest(i, distances, valid_mask_i)
action[i, selected_neighbors] = 1
```

### 6. Non-Deterministic Randomness

**Problem:**
```python
selected = np.random.choice(candidates)  # Uses global random state!
```

**Solution:**
```python
def __init__(self, seed=None):
    self.rng = np.random.RandomState(seed)

def __call__(self, obs):
    selected = self.rng.choice(candidates)  # Reproducible
```

### 7. Naive Angle Averaging

**Problem:**
```python
mean_heading = np.mean(headings)  # WRONG for circular data!
```

**Solution:**
```python
mean_cos = np.mean(np.cos(headings))
mean_sin = np.mean(np.sin(headings))
mean_heading = np.arctan2(mean_sin, mean_cos)  # Circular mean
```

## Testing Your Baseline

### Manual Validation

Test your baseline with a simple script:

```python
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config, load_dict
from baselines import YourNewBaseline

# Create environment
config_dict = load_dict('envs/default_env_config.yaml')
config = Config(**config_dict)
env_context = config_to_env_input(config, seed_id=42)
env = NeighborSelectionFlockingEnv(env_context)

# Create baseline
baseline = YourNewBaseline(param1=value1, param2=value2)

# Test
obs = env.reset()
action = baseline(obs)

# Validate action
num_agents_max = obs['neighbor_masks'].shape[0]
assert action.shape == (num_agents_max, num_agents_max), f"Wrong shape: {action.shape}"
assert np.issubdtype(action.dtype, np.integer), f"Wrong dtype: {action.dtype}"
assert np.all(np.diag(action) == 1), "Self-loops must be 1"
assert np.all((obs['neighbor_masks'] | ~action)), "Violates neighbor_masks"

padding_mask_2d = obs['padding_mask'][:, np.newaxis] & obs['padding_mask'][np.newaxis, :]
assert np.all((padding_mask_2d | ~action)), "Violates padding_mask"

print("Manual validation passed!")
```

### Add to Test Suite

Add your baseline to `test_baselines.py`:

```python
def main():
    # ... existing tests ...

    # Test your new baseline
    baseline_your_new = YourNewBaseline(param1=value1, param2=value2)
    test_baseline(env, baseline_your_new, "Your New Baseline", num_steps=10)
```

### Run Tests

```bash
python test_baselines.py
```

Expected output:
- No assertion errors
- Actions have correct format
- Environment steps successfully for multiple timesteps
- No crashes or exceptions

## Adding to Factory Function

Update the `create_baseline` function in `baselines.py`:

```python
def create_baseline(baseline_type: str, **kwargs):
    """
    Factory function to create baseline policies.

    Args:
        baseline_type: One of 'random', 'distance', 'nearest', 'farthest', 'mti', 'your_type'
        **kwargs: Arguments to pass to the baseline constructor
    """
    baselines = {
        'random': RandomNeighborSelection,
        'distance': DistanceBasedNeighborSelection,
        'nearest': FixedNearestNeighborSelection,
        'farthest': FixedFarthestNeighborSelection,
        'mti': MetricTopologicalInteractionSelection,
        'your_type': YourNewBaseline,  # Add here
    }

    if baseline_type not in baselines:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Must be one of {list(baselines.keys())}")

    return baselines[baseline_type](**kwargs)
```

## Template Skeleton

Here's a minimal template for a new baseline:

```python
class YourNewBaseline:
    """
    Brief description of your baseline.

    Detailed explanation of the selection strategy.

    Parameters:
        param1 (type): Description
        param2 (type): Description
        seed (int, optional): Random seed for reproducibility
    """

    def __init__(self, param1: float, param2: int,
                 periodic_boundary: bool = False,
                 boundary_size: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialize baseline.

        Args:
            param1: Description
            param2: Description
            periodic_boundary: Whether environment uses periodic boundaries
            boundary_size: Size of periodic boundary (required if periodic_boundary=True)
            seed: Random seed
        """
        # Validate parameters
        assert param1 > 0, "param1 must be positive"
        assert param2 > 0, "param2 must be positive"

        self.param1 = param1
        self.param2 = param2
        self.periodic_boundary = periodic_boundary
        self.boundary_size = boundary_size
        self.rng = np.random.RandomState(seed)

        if periodic_boundary:
            assert boundary_size is not None, "boundary_size required for periodic boundaries"

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate neighbor selection action.

        Args:
            obs: Environment observation

        Returns:
            action: (num_agents_max, num_agents_max) integer array
        """
        neighbor_masks = obs['neighbor_masks']
        padding_mask = obs['padding_mask']
        local_agent_infos = obs['local_agent_infos']
        num_agents_max = neighbor_masks.shape[0]

        # Initialize action with self-loops
        action = np.eye(num_agents_max, dtype=np.int8)

        # Compute distances (if needed)
        if self.periodic_boundary:
            raise NotImplementedError("Periodic boundary not yet implemented")
        else:
            rel_positions = local_agent_infos[:, :, :2]
            distances = np.linalg.norm(rel_positions, axis=2)

        # Create valid neighbor mask
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        valid_neighbors = neighbor_masks & padding_mask_2d
        valid_neighbors_no_self = valid_neighbors.copy()
        np.fill_diagonal(valid_neighbors_no_self, False)

        # Process each agent
        for i in range(num_agents_max):
            if not padding_mask[i]:
                continue  # Skip padding agents

            valid_mask_i = valid_neighbors_no_self[i]

            if not valid_mask_i.any():
                continue  # No valid neighbors

            # YOUR SELECTION LOGIC HERE
            # Example: select all valid neighbors within a distance
            selected = (distances[i] <= self.param1) & valid_mask_i
            action[i, selected] = 1

        return action
```

## Periodic Boundary Support

If you want to add periodic boundary support:

1. **Check the observation format:**
   - For periodic: `obs_dim = 6`, positions are `[cos(x), sin(x), cos(y), sin(y)]`
   - You need to reconstruct distances from sin/cos representation

2. **Use environment utilities:**
   ```python
   from utils.utils import get_rel_pos_dist_in_periodic_boundary
   ```

3. **Example pattern:**
   ```python
   if self.periodic_boundary:
       # Reconstruct positions from sin/cos
       # Compute distances accounting for wrap-around
       # This is non-trivial - see env.py for reference
       raise NotImplementedError("Periodic boundary distance computation not yet implemented")
   else:
       # Standard distance computation
       rel_positions = local_agent_infos[:, :, :2]
       distances = np.linalg.norm(rel_positions, axis=2)
   ```

## Best Practices

1. **Vectorize when possible:** Use NumPy operations instead of nested loops
2. **Use efficient sorting:** `np.argpartition` instead of `np.argsort` when you only need k smallest/largest
3. **Document thoroughly:** Explain your algorithm and any non-obvious implementation details
4. **Handle edge cases:** Empty neighbor sets, insufficient neighbors, padding agents, etc.
5. **Validate inputs:** Use assertions in `__init__` to catch invalid parameters early
6. **Maintain consistency:** Follow the same patterns as existing baselines
7. **Test extensively:** Test with different numbers of agents, communication ranges, and configurations

## Further Reading

- See `envs/env.py` lines 935-1002 for observation construction details
- See `envs/env.py` lines 394-451 for how actions are processed in `step()`
- See existing baselines in `baselines.py` for implementation examples
- See `test_baselines.py` for testing patterns

## Questions?

If you have questions or encounter issues:
1. Check existing baselines for similar functionality
2. Review this guide and the environment implementation in `envs/env.py`
3. Run tests to identify specific constraint violations
4. Open an issue with a minimal reproduction example
