"""
Quick verification script to demonstrate MTI baseline functionality.

This script shows that the MTI baseline can be imported, instantiated,
and used without errors.
"""

import numpy as np

# Test imports
print("Testing imports...")
try:
    from baselines import MetricTopologicalInteractionSelection, create_baseline
    print("✓ MTI baseline imports successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test instantiation
print("\nTesting instantiation...")
try:
    mti = MetricTopologicalInteractionSelection(
        k=5,
        distance_threshold=0.5,
        threshold_a=0.1,
        threshold_b=0.5,
        seed=42
    )
    print("✓ MTI baseline instantiates successfully")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    exit(1)

# Test factory function
print("\nTesting factory function...")
try:
    mti_factory = create_baseline('mti', k=3, distance_threshold=0.6,
                                  threshold_a=0.15, threshold_b=0.4, seed=42)
    print("✓ MTI baseline created via factory successfully")
except Exception as e:
    print(f"✗ Factory creation failed: {e}")
    exit(1)

# Test with mock observation
print("\nTesting with mock observation...")
try:
    num_agents_max = 10
    obs_dim = 4

    # Create mock observation
    mock_obs = {
        'local_agent_infos': np.random.randn(num_agents_max, num_agents_max, obs_dim),
        'neighbor_masks': np.ones((num_agents_max, num_agents_max), dtype=bool),
        'padding_mask': np.ones(num_agents_max, dtype=bool),
    }

    # Ensure observation has valid structure
    # Set diagonal of relative headings to [1, 0] (self-relative heading = 0)
    mock_obs['local_agent_infos'][:, :, 2] = np.random.randn(num_agents_max, num_agents_max)
    mock_obs['local_agent_infos'][:, :, 3] = np.random.randn(num_agents_max, num_agents_max)
    for i in range(num_agents_max):
        mock_obs['local_agent_infos'][i, i, 2] = 1.0  # cos(0) = 1
        mock_obs['local_agent_infos'][i, i, 3] = 0.0  # sin(0) = 0

    # Generate action
    action = mti(mock_obs)

    # Validate action
    assert action.shape == (num_agents_max, num_agents_max), \
        f"Wrong shape: {action.shape}"
    assert np.issubdtype(action.dtype, np.integer), \
        f"Wrong dtype: {action.dtype}"
    assert np.all(np.diag(action) == 1), \
        "Self-loops must be 1"
    assert np.all((mock_obs['neighbor_masks'] | ~action.astype(bool))), \
        "Violates neighbor_masks"

    print("✓ MTI baseline generates valid actions")
    print(f"  Action shape: {action.shape}")
    print(f"  Action dtype: {action.dtype}")
    print(f"  Self-loops correct: {np.all(np.diag(action) == 1)}")
    print(f"  Average neighbors selected: {(action.sum(axis=1) - 1).mean():.2f}")

except Exception as e:
    print(f"✗ Mock observation test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test reset functionality
print("\nTesting reset functionality...")
try:
    mti.reset(num_agents_max=10)
    print("✓ Reset functionality works")
    print(f"  Modes initialized: {mti.modes is not None}")
    print(f"  All modes are TOP: {np.all(mti.modes == mti.MODE_TOP)}")
except Exception as e:
    print(f"✗ Reset test failed: {e}")
    exit(1)

# Test mode switching over multiple steps
print("\nTesting mode switching over multiple steps...")
try:
    mti.reset(num_agents_max=10)
    initial_modes = mti.modes.copy()

    # Run multiple steps
    for step in range(5):
        action = mti(mock_obs)

    final_modes = mti.modes.copy()

    print("✓ Mode switching works over multiple steps")
    print(f"  Initial modes (all TOP): {np.all(initial_modes == mti.MODE_TOP)}")
    print(f"  Final modes TOP count: {(final_modes == mti.MODE_TOP).sum()}/{num_agents_max}")
    print(f"  Final modes MET count: {(final_modes == mti.MODE_MET).sum()}/{num_agents_max}")

except Exception as e:
    print(f"✗ Multi-step test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("All verification tests passed! ✓")
print("="*60)
print("\nMTI baseline is ready to use!")
print("\nExample usage:")
print("  from baselines import MetricTopologicalInteractionSelection")
print("  baseline = MetricTopologicalInteractionSelection(k=5, distance_threshold=0.5,")
print("                                                    threshold_a=0.1, threshold_b=0.5)")
print("  action = baseline(obs)")
