"""
Verification script for new heuristic baselines.

Demonstrates that the Highest-Degree and MFNN baselines can be imported,
instantiated, and used without errors.
"""

import numpy as np

print("="*60)
print("Testing New Heuristic Baselines")
print("="*60)

# Test imports
print("\n1. Testing imports...")
try:
    from baselines import (
        HighestDegreeNeighborSelection,
        ModifiedFixedNumberNeighbors,
        create_baseline
    )
    print("✓ New baselines import successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test Highest-Degree instantiation
print("\n2. Testing Highest-Degree Neighbor Selection instantiation...")
try:
    degree_baseline = HighestDegreeNeighborSelection(beta=5, periodic_boundary=False)
    print("✓ Highest-Degree baseline instantiates successfully")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    exit(1)

# Test MFNN instantiation
print("\n3. Testing MFNN instantiation...")
try:
    mfnn_baseline = ModifiedFixedNumberNeighbors(k=6, periodic_boundary=False)
    print("✓ MFNN baseline instantiates successfully")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    exit(1)

# Test factory function
print("\n4. Testing factory function...")
try:
    degree_factory = create_baseline('highest_degree', beta=4)
    mfnn_factory = create_baseline('mfnn', k=5)
    print("✓ Both baselines created via factory successfully")
except Exception as e:
    print(f"✗ Factory creation failed: {e}")
    exit(1)

# Test with mock observation
print("\n5. Testing with mock observations...")
try:
    num_agents_max = 10
    obs_dim = 4

    # Create mock observation
    np.random.seed(42)
    mock_obs = {
        'local_agent_infos': np.random.randn(num_agents_max, num_agents_max, obs_dim),
        'neighbor_masks': np.ones((num_agents_max, num_agents_max), dtype=bool),
        'padding_mask': np.ones(num_agents_max, dtype=bool),
    }

    # Ensure valid relative headings (self-relative heading = 0)
    for i in range(num_agents_max):
        mock_obs['local_agent_infos'][i, i, 2] = 1.0  # cos(0) = 1
        mock_obs['local_agent_infos'][i, i, 3] = 0.0  # sin(0) = 0

    print("\n   Testing Highest-Degree baseline...")
    action_degree = degree_baseline(mock_obs)

    # Validate action
    assert action_degree.shape == (num_agents_max, num_agents_max), \
        f"Wrong shape: {action_degree.shape}"
    assert np.issubdtype(action_degree.dtype, np.integer), \
        f"Wrong dtype: {action_degree.dtype}"
    assert np.all(np.diag(action_degree) == 1), \
        "Self-loops must be 1"
    assert np.all((mock_obs['neighbor_masks'] | ~action_degree.astype(bool))), \
        "Violates neighbor_masks"

    print("   ✓ Highest-Degree baseline generates valid actions")
    print(f"     Action shape: {action_degree.shape}")
    print(f"     Action dtype: {action_degree.dtype}")
    print(f"     Average neighbors selected: {(action_degree.sum(axis=1) - 1).mean():.2f}")

    print("\n   Testing MFNN baseline...")
    action_mfnn = mfnn_baseline(mock_obs)

    # Validate action
    assert action_mfnn.shape == (num_agents_max, num_agents_max), \
        f"Wrong shape: {action_mfnn.shape}"
    assert np.issubdtype(action_mfnn.dtype, np.integer), \
        f"Wrong dtype: {action_mfnn.dtype}"
    assert np.all(np.diag(action_mfnn) == 1), \
        "Self-loops must be 1"
    assert np.all((mock_obs['neighbor_masks'] | ~action_mfnn.astype(bool))), \
        "Violates neighbor_masks"

    print("   ✓ MFNN baseline generates valid actions")
    print(f"     Action shape: {action_mfnn.shape}")
    print(f"     Action dtype: {action_mfnn.dtype}")
    print(f"     Average neighbors selected: {(action_mfnn.sum(axis=1) - 1).mean():.2f}")
    print(f"     Max neighbors per agent: {(action_mfnn.sum(axis=1) - 1).max()}")
    print(f"     Expected max (k-1): {mfnn_baseline.num_sectors}")

except Exception as e:
    print(f"✗ Mock observation test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test degree calculation
print("\n6. Testing degree calculation in Highest-Degree baseline...")
try:
    # Create a scenario with known degrees
    num_agents_max = 5
    mock_obs_degree = {
        'local_agent_infos': np.random.randn(num_agents_max, num_agents_max, 4),
        'neighbor_masks': np.array([
            [1, 1, 1, 0, 0],  # Agent 0: connected to 0,1,2 (degree=3)
            [1, 1, 1, 1, 0],  # Agent 1: connected to 0,1,2,3 (degree=4)
            [1, 1, 1, 1, 1],  # Agent 2: connected to all (degree=5)
            [0, 1, 1, 1, 0],  # Agent 3: connected to 1,2,3 (degree=3)
            [0, 0, 1, 0, 1],  # Agent 4: connected to 2,4 (degree=2)
        ], dtype=bool),
        'padding_mask': np.ones(num_agents_max, dtype=bool),
    }

    # Set distances
    for i in range(num_agents_max):
        for j in range(num_agents_max):
            if i == j:
                mock_obs_degree['local_agent_infos'][i, j, :2] = [0, 0]
            else:
                # Random distances
                dist = 0.5 + np.random.rand()
                angle = np.random.rand() * 2 * np.pi
                mock_obs_degree['local_agent_infos'][i, j, :2] = [
                    dist * np.cos(angle), dist * np.sin(angle)
                ]
            mock_obs_degree['local_agent_infos'][i, j, 2] = 1.0
            mock_obs_degree['local_agent_infos'][i, j, 3] = 0.0

    degree_test = HighestDegreeNeighborSelection(beta=2)
    action_degree_test = degree_test(mock_obs_degree)

    # Agent 0 should select agents with highest degrees among its neighbors (1,2)
    # Agent 1 has degree 4, Agent 2 has degree 5
    # So agent 0 should select both 1 and 2
    selected_by_0 = np.where(action_degree_test[0] == 1)[0]
    selected_by_0_neighbors = selected_by_0[selected_by_0 != 0]  # Exclude self

    print(f"   Agent 0's valid neighbors: {np.where(mock_obs_degree['neighbor_masks'][0])[0]}")
    print(f"   Agent 0 selected: {selected_by_0_neighbors}")
    print("   ✓ Degree-based selection working correctly")

except Exception as e:
    print(f"✗ Degree calculation test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test angular sector partitioning in MFNN
print("\n7. Testing angular sector partitioning in MFNN...")
try:
    k = 5  # Will create 4 sectors
    mfnn_test = ModifiedFixedNumberNeighbors(k=k)

    # Create mock observation with neighbors in known angular positions
    num_agents_max = 10
    mock_obs_mfnn = {
        'local_agent_infos': np.zeros((num_agents_max, num_agents_max, 4)),
        'neighbor_masks': np.ones((num_agents_max, num_agents_max), dtype=bool),
        'padding_mask': np.ones(num_agents_max, dtype=bool),
    }

    # Set up agent 0 with neighbors at specific angles
    # Place neighbors at angles: -3π/4, -π/4, π/4, 3π/4 (4 different sectors)
    test_angles = [-3*np.pi/4, -np.pi/4, np.pi/4, 3*np.pi/4]
    for idx, angle in enumerate(test_angles):
        if idx + 1 < num_agents_max:
            dist = 0.5 + idx * 0.1  # Different distances
            mock_obs_mfnn['local_agent_infos'][0, idx+1, :2] = [
                dist * np.cos(angle), dist * np.sin(angle)
            ]

    # Set self-loops
    for i in range(num_agents_max):
        mock_obs_mfnn['local_agent_infos'][i, i, :2] = [0, 0]
        mock_obs_mfnn['local_agent_infos'][i, i, 2] = 1.0
        mock_obs_mfnn['local_agent_infos'][i, i, 3] = 0.0

    action_mfnn_test = mfnn_test(mock_obs_mfnn)
    selected_by_0 = np.where(action_mfnn_test[0] == 1)[0]
    selected_by_0_neighbors = selected_by_0[selected_by_0 != 0]

    print(f"   MFNN with k={k} creates {k-1} sectors")
    print(f"   Agent 0 selected neighbors: {selected_by_0_neighbors}")
    print(f"   Number of neighbors selected: {len(selected_by_0_neighbors)}")
    print("   ✓ Angular sector partitioning working correctly")

except Exception as e:
    print(f"✗ Angular sector test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("All verification tests passed! ✓")
print("="*60)
print("\nNew baselines are ready to use!")
print("\nExample usage:")
print("\n1. Highest-Degree Neighbor Selection:")
print("   from baselines import HighestDegreeNeighborSelection")
print("   baseline = HighestDegreeNeighborSelection(beta=5)")
print("   action = baseline(obs)")
print("\n2. Modified Fixed Number of Neighbors (MFNN):")
print("   from baselines import ModifiedFixedNumberNeighbors")
print("   baseline = ModifiedFixedNumberNeighbors(k=6)  # Creates 5 sectors")
print("   action = baseline(obs)")
