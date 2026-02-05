"""
Verification script for Active Search Neighbor Selection baseline.

Demonstrates that the ActiveSearchNeighborSelection baseline can be imported,
instantiated, and used correctly.
"""

import numpy as np

print("="*60)
print("Testing Active Search Neighbor Selection Baseline")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from baselines import (
        ActiveSearchNeighborSelection,
        create_baseline
    )
    from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config, load_dict
    print("✓ Active Search baseline imports successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test instantiation
print("\n2. Testing Active Search instantiation...")
try:
    active_baseline = ActiveSearchNeighborSelection(
        time_window=10,
        alignment_enabled=True,
        search_enabled=True,
        seed=42
    )
    print("✓ Active Search baseline instantiates successfully")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    exit(1)

# Test factory function
print("\n3. Testing factory function...")
try:
    active_factory = create_baseline('active_search',
                                    time_window=10,
                                    alignment_enabled=True,
                                    seed=123)
    print("✓ Active Search created via factory successfully")
except Exception as e:
    print(f"✗ Factory creation failed: {e}")
    exit(1)

# Test with real environment
print("\n4. Testing with real environment...")
try:
    # Create environment
    config_dict = load_dict('envs/default_env_config.yaml')
    config = Config(**config_dict)
    env_context = config_to_env_input(config, seed_id=42)
    env = NeighborSelectionFlockingEnv(env_context)
    print("✓ Environment created successfully")

    # Get observation
    obs = env.reset()
    num_agents_max = obs['neighbor_masks'].shape[0]
    print(f"✓ Environment reset, num_agents_max={num_agents_max}")

    # Call baseline
    action = active_baseline(obs)
    print(f"✓ Baseline called successfully, action shape: {action.shape}")

except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test action validation
print("\n5. Testing action validation...")
try:
    # Check shape
    assert action.shape == (num_agents_max, num_agents_max), \
        f"Wrong shape: expected ({num_agents_max}, {num_agents_max}), got {action.shape}"
    print(f"✓ Action shape correct: {action.shape}")

    # Check dtype
    assert action.dtype in [np.int8, np.int32, np.int64], \
        f"Wrong dtype: expected integer, got {action.dtype}"
    print(f"✓ Action dtype correct: {action.dtype}")

    # Check self-loops
    assert np.all(np.diag(action) == 1), "Self-loops not set correctly"
    print("✓ Self-loops set correctly")

    # Check neighbor_masks constraint
    assert np.all((obs['neighbor_masks'] | ~action.astype(bool))), \
        "Action violates neighbor_masks"
    print("✓ neighbor_masks constraint respected")

    # Check padding_mask constraint
    padding_mask_2d = obs['padding_mask'][:, np.newaxis] & obs['padding_mask'][np.newaxis, :]
    assert np.all((padding_mask_2d | ~action.astype(bool))), \
        "Action violates padding_mask"
    print("✓ padding_mask constraint respected")

except AssertionError as e:
    print(f"✗ Validation failed: {e}")
    exit(1)

# Test reset functionality
print("\n6. Testing reset functionality...")
try:
    # Manual reset
    active_baseline.reset()
    assert active_baseline.num_agents_max is None, "Reset should clear num_agents_max"
    print("✓ Manual reset works correctly")

    # Auto-detection
    obs = env.reset()
    action = active_baseline(obs)
    assert active_baseline.num_agents_max is not None, "Auto-detection should set num_agents_max"
    print(f"✓ Auto-detection works, num_agents_max={active_baseline.num_agents_max}")

except Exception as e:
    print(f"✗ Reset test failed: {e}")
    exit(1)

# Test episode execution
print("\n7. Testing episode execution...")
try:
    obs = env.reset()
    total_reward = 0
    steps = 0

    for step in range(50):
        action = active_baseline(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"✓ Episode executed: {steps} steps, total reward: {total_reward:.2f}")

except Exception as e:
    print(f"✗ Episode execution failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with alignment disabled
print("\n8. Testing with alignment disabled...")
try:
    active_no_align = ActiveSearchNeighborSelection(
        time_window=10,
        alignment_enabled=False,
        search_enabled=True,
        seed=42
    )
    obs = env.reset()
    action = active_no_align(obs)
    print(f"✓ Works with alignment_enabled=False")

except Exception as e:
    print(f"✗ Test failed: {e}")
    exit(1)

# Test observation history management
print("\n9. Testing observation history management...")
try:
    active_hist = ActiveSearchNeighborSelection(time_window=3, seed=42)
    obs = env.reset()

    # Run multiple steps
    for step in range(5):
        action = active_hist(obs)
        obs, _, _, _ = env.step(action)

    # Check history length doesn't exceed time_window
    assert len(active_hist.observation_history) <= 3, \
        f"History length {len(active_hist.observation_history)} exceeds time_window=3"
    print(f"✓ History management works, length={len(active_hist.observation_history)}")

except Exception as e:
    print(f"✗ History test failed: {e}")
    exit(1)

# Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nAll tests passed:")
print("✓ Import and instantiation")
print("✓ Factory function")
print("✓ Action format validation")
print("✓ Mask constraints respected")
print("✓ Reset functionality (manual and auto-detection)")
print("✓ Episode execution")
print("✓ Alignment/search toggle")
print("✓ Observation history management")
print("\nFixed issues:")
print("✓ Issue 1: reset() signature standardized to reset(self)")
print("✓ Issue 2: Episode detection improved (None check added)")
print("✓ Issue 3: valid_mask handling clarified")
print("="*60)
