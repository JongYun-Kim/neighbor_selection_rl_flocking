"""
Test script for baseline neighbor selection heuristics.

This script demonstrates how to use the baseline heuristics with the
flocking environment and verifies they produce valid actions.
"""

import numpy as np
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config
from baselines import (
    RandomNeighborSelection,
    DistanceBasedNeighborSelection,
    FixedNearestNeighborSelection,
    FixedFarthestNeighborSelection,
    create_baseline
)


def test_baseline(env, baseline, baseline_name, num_steps=10):
    """
    Test a baseline policy in the environment.

    Args:
        env: Flocking environment instance
        baseline: Baseline policy object
        baseline_name: Name of the baseline for logging
        num_steps: Number of steps to run

    Returns:
        Total reward accumulated
    """
    print(f"\n{'='*60}")
    print(f"Testing: {baseline_name}")
    print(f"{'='*60}")

    obs = env.reset()
    print(f"Environment reset. Num agents: {env.num_agents}")
    print(f"Observation keys: {obs.keys()}")
    print(f"Padding mask: {obs['padding_mask']}")
    print(f"Neighbor masks shape: {obs['neighbor_masks'].shape}")

    total_reward = 0
    for step in range(num_steps):
        # Get action from baseline
        action = baseline(obs)

        # Validate action format
        assert action.shape == (env.num_agents_max, env.num_agents_max), \
            f"Action shape {action.shape} != expected {(env.num_agents_max, env.num_agents_max)}"
        assert np.issubdtype(action.dtype, np.integer), \
            f"Action dtype {action.dtype} is not integer"
        assert np.all(np.diag(action) == 1), \
            "Self-loops must all be 1"

        # Check that action respects neighbor masks
        invalid_selections = action & ~obs['neighbor_masks']
        assert not invalid_selections.any(), \
            f"Action selects invalid neighbors (not in neighbor_masks)"

        # Check that action respects padding mask
        padding_mask_2d = obs['padding_mask'][:, np.newaxis] & obs['padding_mask'][np.newaxis, :]
        invalid_padding = action & ~padding_mask_2d
        assert not invalid_padding.any(), \
            f"Action selects padding agents"

        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Count selected neighbors per agent
        num_selected = action.sum(axis=1) - 1  # Subtract self-loop
        active_agents = obs['padding_mask']
        avg_selected = num_selected[active_agents].mean() if active_agents.any() else 0

        print(f"Step {step+1}: reward={reward:.4f}, "
              f"avg_neighbors_selected={avg_selected:.2f}, done={done}")

        if done:
            print(f"Episode finished at step {step+1}")
            break

    print(f"Total reward: {total_reward:.4f}")
    return total_reward


def main():
    """Run tests for all baseline heuristics."""
    print("Baseline Heuristics Test Suite")
    print("="*60)

    # Create environment configuration
    # Load default config and modify for testing
    from envs.env import load_dict

    config_dict = load_dict('envs/default_env_config.yaml')

    # Modify for faster testing
    config_dict['env']['num_agents_pool'] = [5, 10]  # Smaller pool for testing
    config_dict['env']['max_time_steps'] = 50  # Fewer steps
    config_dict['env']['task_type'] = 'acs'  # Use ACS task

    config = Config(**config_dict)

    # Create environment
    env_context = config_to_env_input(config, seed_id=42)
    env = NeighborSelectionFlockingEnv(env_context)

    print(f"\nEnvironment created:")
    print(f"  Task type: {config.env.task_type}")
    print(f"  Num agents range: [{env.num_agents_min}, {env.num_agents_max}]")
    print(f"  Communication range: {config.env.comm_range}")
    print(f"  Max time steps: {config.env.max_time_steps}")

    # Test 1: Random Neighbor Selection
    baseline_random = RandomNeighborSelection(selection_probability=0.5, seed=42)
    test_baseline(env, baseline_random, "Random Neighbor Selection (p=0.5)", num_steps=10)

    # Test 2: Distance-Based Neighbor Selection
    # Note: distance_threshold should be in normalized units (divided by l/2)
    # For testing, use a threshold that's reasonable for normalized distances
    baseline_distance = DistanceBasedNeighborSelection(
        distance_threshold=0.5,  # Normalized distance
        periodic_boundary=False
    )
    test_baseline(env, baseline_distance, "Distance-Based Selection (threshold=0.5)", num_steps=10)

    # Test 3: Fixed-Nearest Neighbor Selection
    baseline_nearest = FixedNearestNeighborSelection(k=3, periodic_boundary=False)
    test_baseline(env, baseline_nearest, "Fixed-Nearest Neighbor Selection (k=3)", num_steps=10)

    # Test 4: Fixed-Farthest Neighbor Selection
    baseline_farthest = FixedFarthestNeighborSelection(k=3, periodic_boundary=False)
    test_baseline(env, baseline_farthest, "Fixed-Farthest Neighbor Selection (k=3)", num_steps=10)

    # Test 5: Using factory function
    print(f"\n{'='*60}")
    print("Testing factory function")
    print(f"{'='*60}")
    baseline_factory = create_baseline('random', selection_probability=0.7, seed=123)
    test_baseline(env, baseline_factory, "Random (via factory, p=0.7)", num_steps=10)

    print(f"\n{'='*60}")
    print("All tests completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
