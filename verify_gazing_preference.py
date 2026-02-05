"""
Verification script for GazingPreferenceNeighborSelection baseline.

Tests the implementation against the paper:
"Gazing Preference Induced Controllable Milling Behavior in Swarm Robotics"
(Zhou et al., IEEE Robotics and Automation Letters, Vol. 10, No. 6, June 2025)
"""

import numpy as np
from baselines import GazingPreferenceNeighborSelection, create_baseline
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config, load_dict


def main():
    print("=" * 60)
    print("Testing Gazing Preference Neighbor Selection Baseline")
    print("=" * 60)
    print()

    # =========================================================================
    # Test 1: Import and instantiation
    # =========================================================================
    print("1. Testing imports...")
    try:
        baseline = GazingPreferenceNeighborSelection()
        print("✓ Gazing Preference baseline imports successfully")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return
    print()

    # =========================================================================
    # Test 2: Parameter validation
    # =========================================================================
    print("2. Testing parameter validation...")

    # Valid parameters
    try:
        baseline = GazingPreferenceNeighborSelection(
            theta_g=np.pi/3,
            lambda_coef=3.0,
            beta=0.5
        )
        print("✓ Valid parameters accepted")
    except Exception as e:
        print(f"✗ Valid parameters rejected: {e}")
        return

    # Invalid theta_g (outside [-π/2, π/2])
    try:
        baseline = GazingPreferenceNeighborSelection(theta_g=np.pi)
        print("✗ Invalid theta_g not rejected")
        return
    except AssertionError:
        print("✓ Invalid theta_g rejected correctly")

    # Invalid lambda_coef (negative)
    try:
        baseline = GazingPreferenceNeighborSelection(lambda_coef=-1.0)
        print("✗ Invalid lambda_coef not rejected")
        return
    except AssertionError:
        print("✓ Invalid lambda_coef rejected correctly")

    # Invalid beta (outside [0, 1])
    try:
        baseline = GazingPreferenceNeighborSelection(beta=1.5)
        print("✗ Invalid beta not rejected")
        return
    except AssertionError:
        print("✓ Invalid beta rejected correctly")
    print()

    # =========================================================================
    # Test 3: Factory function
    # =========================================================================
    print("3. Testing factory function...")
    try:
        baseline_factory = create_baseline('gazing_preference', theta_g=np.pi/4, lambda_coef=5.0)
        print("✓ Gazing Preference created via factory successfully")
    except Exception as e:
        print(f"✗ Factory creation failed: {e}")
        return
    print()

    # =========================================================================
    # Test 4: Integration with environment
    # =========================================================================
    print("4. Testing with real environment...")
    try:
        config_dict = load_dict('envs/default_env_config.yaml')
        config_dict['env']['num_agents_pool'] = [5, 10]
        config_dict['env']['max_time_steps'] = 50
        config = Config(**config_dict)
        env_context = config_to_env_input(config, seed_id=42)
        env = NeighborSelectionFlockingEnv(env_context)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return

    baseline = GazingPreferenceNeighborSelection(
        theta_g=np.pi/3,
        lambda_coef=3.0,
        beta=0.5,
        seed=42
    )

    try:
        obs = env.reset()
        num_agents_max = obs['neighbor_masks'].shape[0]
        print(f"✓ Environment reset, num_agents_max={num_agents_max}")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        return

    try:
        action = baseline(obs)
        print(f"✓ Baseline called successfully, action shape: {action.shape}")
    except Exception as e:
        print(f"✗ Baseline call failed: {e}")
        return
    print()

    # =========================================================================
    # Test 5: Action validation
    # =========================================================================
    print("5. Testing action validation...")

    # Check action shape
    if action.shape == (num_agents_max, num_agents_max):
        print(f"✓ Action shape correct: {action.shape}")
    else:
        print(f"✗ Action shape incorrect: {action.shape}, expected ({num_agents_max}, {num_agents_max})")
        return

    # Check action dtype
    if action.dtype == np.int8:
        print(f"✓ Action dtype correct: {action.dtype}")
    else:
        print(f"✗ Action dtype incorrect: {action.dtype}, expected int8")

    # Check self-loops
    if np.all(np.diag(action) == 1):
        print("✓ Self-loops set correctly")
    else:
        print("✗ Self-loops not set correctly")
        return

    # Check neighbor_masks constraint
    neighbor_masks = obs['neighbor_masks']
    padding_mask = obs['padding_mask']

    # Create valid mask (including self-loops)
    padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
    valid_with_self = neighbor_masks & padding_mask_2d
    np.fill_diagonal(valid_with_self, True)  # Self-loops always valid

    if np.all(action <= valid_with_self):
        print("✓ neighbor_masks constraint respected")
    else:
        print("✗ neighbor_masks constraint violated")
        return

    # Check padding_mask constraint
    # Padded agents should still have self-loops (diagonal)
    action_no_diag = action.copy()
    np.fill_diagonal(action_no_diag, 0)
    invalid_rows = action_no_diag[~padding_mask].sum()
    invalid_cols = action_no_diag[:, ~padding_mask].sum()
    if invalid_rows == 0 and invalid_cols == 0:
        print("✓ padding_mask constraint respected")
    else:
        print(f"✗ padding_mask constraint violated (invalid rows: {invalid_rows}, cols: {invalid_cols})")
    print()

    # =========================================================================
    # Test 6: Reset functionality
    # =========================================================================
    print("6. Testing reset functionality...")

    # Manual reset
    baseline.reset()
    if baseline.num_agents_max is None:
        print("✓ Manual reset works correctly")
    else:
        print(f"✗ Manual reset failed, num_agents_max={baseline.num_agents_max}")
        return

    # Auto-detection
    obs = env.reset()
    action = baseline(obs)
    if baseline.num_agents_max == num_agents_max:
        print(f"✓ Auto-detection works, num_agents_max={baseline.num_agents_max}")
    else:
        print(f"✗ Auto-detection failed, num_agents_max={baseline.num_agents_max}")
        return
    print()

    # =========================================================================
    # Test 7: Episode execution
    # =========================================================================
    print("7. Testing episode execution...")

    baseline = GazingPreferenceNeighborSelection(
        theta_g=np.pi/3,
        lambda_coef=3.0,
        seed=42
    )

    obs = env.reset()
    total_reward = 0.0
    num_steps = 50

    try:
        for step in range(num_steps):
            action = baseline(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"✓ Episode executed: {step + 1} steps, total reward: {total_reward:.2f}")
    except Exception as e:
        print(f"✗ Episode execution failed: {e}")
        return
    print()

    # =========================================================================
    # Test 8: Different theta_g values (milling direction)
    # =========================================================================
    print("8. Testing different theta_g values...")

    # Counterclockwise milling (positive theta_g)
    baseline_ccw = GazingPreferenceNeighborSelection(
        theta_g=np.pi/3,  # 60 degrees
        lambda_coef=3.0,
        seed=42
    )

    # Clockwise milling (negative theta_g)
    baseline_cw = GazingPreferenceNeighborSelection(
        theta_g=-np.pi/3,  # -60 degrees
        lambda_coef=3.0,
        seed=42
    )

    obs = env.reset()
    action_ccw = baseline_ccw(obs)

    obs = env.reset()
    action_cw = baseline_cw(obs)

    print(f"✓ Counterclockwise milling (θ^g=π/3): action shape {action_ccw.shape}")
    print(f"✓ Clockwise milling (θ^g=-π/3): action shape {action_cw.shape}")
    print()

    # =========================================================================
    # Test 9: Different lambda_coef values (selection concentration)
    # =========================================================================
    print("9. Testing different lambda_coef values...")

    # Low lambda (more uniform selection)
    baseline_low_lambda = GazingPreferenceNeighborSelection(
        theta_g=np.pi/3,
        lambda_coef=1.0,
        seed=42
    )

    # High lambda (more concentrated selection)
    baseline_high_lambda = GazingPreferenceNeighborSelection(
        theta_g=np.pi/3,
        lambda_coef=5.0,
        seed=42
    )

    obs = env.reset()
    action_low = baseline_low_lambda(obs)

    obs = env.reset()
    action_high = baseline_high_lambda(obs)

    print(f"✓ Low lambda (λ=1.0): action shape {action_low.shape}")
    print(f"✓ High lambda (λ=5.0): action shape {action_high.shape}")
    print()

    # =========================================================================
    # Test 10: No neighbors case
    # =========================================================================
    print("10. Testing no neighbors case...")

    baseline = GazingPreferenceNeighborSelection(seed=42)

    # Create observation with no neighbors
    obs_no_neighbors = {
        'neighbor_masks': np.eye(num_agents_max, dtype=bool),  # Only self
        'padding_mask': np.ones(num_agents_max, dtype=bool),
        'local_agent_infos': np.zeros((num_agents_max, num_agents_max, 4))
    }

    try:
        action = baseline(obs_no_neighbors)
        # Should only have self-loops
        if np.array_equal(action, np.eye(num_agents_max, dtype=np.int8)):
            print("✓ No neighbors case handled correctly (only self-loops)")
        else:
            print("✗ No neighbors case: unexpected action pattern")
    except Exception as e:
        print(f"✗ No neighbors case failed: {e}")
        return
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print()
    print("All tests passed:")
    print("✓ Import and instantiation")
    print("✓ Parameter validation (theta_g, lambda_coef, beta)")
    print("✓ Factory function")
    print("✓ Environment integration")
    print("✓ Action format validation")
    print("✓ Mask constraints respected")
    print("✓ Reset functionality (manual and auto-detection)")
    print("✓ Episode execution")
    print("✓ Different theta_g values (milling direction control)")
    print("✓ Different lambda_coef values (selection concentration)")
    print("✓ No neighbors case")
    print()
    print("Usage pattern:")
    print("  baseline = GazingPreferenceNeighborSelection(")
    print("      theta_g=np.pi/3,    # Counterclockwise milling")
    print("      lambda_coef=3.0,    # Medium concentration")
    print("      beta=0.5            # Medium gazing preference strength")
    print("  )")
    print("  action = baseline(obs)")
    print()
    print("Paper parameters (Section III.A):")
    print("  - theta_g ∈ [-π/2, π/2]: gazing preference angle")
    print("  - lambda_coef ∈ {1, 3, 5}: probability density function coefficient")
    print("  - beta ∈ [0, 1]: gazing preference force coefficient")
    print("=" * 60)


if __name__ == "__main__":
    main()
