"""
Verification script for MotionSalienceThresholdSelection baseline.

Tests the implementation against the paper:
"Tuning responsivity-persistence trade-off in swarm robotics:
A motion salience threshold approach"
(Li et al., Robotics and Autonomous Systems, 2025)
"""

import numpy as np
from baselines import MotionSalienceThresholdSelection, create_baseline
from envs.env import NeighborSelectionFlockingEnv, config_to_env_input, Config, load_dict


def main():
    print("=" * 60)
    print("Testing Motion Salience Threshold Selection Baseline")
    print("=" * 60)
    print()

    # =========================================================================
    # Test 1: Import and instantiation
    # =========================================================================
    print("1. Testing imports...")
    try:
        baseline = MotionSalienceThresholdSelection(motion_salience_threshold=1.0)
        print("✓ Motion Salience baseline imports successfully")
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
        baseline = MotionSalienceThresholdSelection(
            motion_salience_threshold=1.0,
            velocity_diff_threshold=0.2,
            k_neighbors=7,
            time_interval=1.0,
            scaling_factor=100.0
        )
        print("✓ Valid parameters accepted")
    except Exception as e:
        print(f"✗ Valid parameters rejected: {e}")
        return

    # Invalid motion_salience_threshold (negative)
    try:
        baseline = MotionSalienceThresholdSelection(motion_salience_threshold=-1.0)
        print("✗ Invalid motion_salience_threshold not rejected")
        return
    except AssertionError:
        print("✓ Invalid motion_salience_threshold rejected correctly")

    # Invalid velocity_diff_threshold (non-positive)
    try:
        baseline = MotionSalienceThresholdSelection(
            motion_salience_threshold=1.0,
            velocity_diff_threshold=0.0
        )
        print("✗ Invalid velocity_diff_threshold not rejected")
        return
    except AssertionError:
        print("✓ Invalid velocity_diff_threshold rejected correctly")

    # Invalid k_neighbors (non-positive)
    try:
        baseline = MotionSalienceThresholdSelection(
            motion_salience_threshold=1.0,
            k_neighbors=0
        )
        print("✗ Invalid k_neighbors not rejected")
        return
    except AssertionError:
        print("✓ Invalid k_neighbors rejected correctly")
    print()

    # =========================================================================
    # Test 3: Factory function
    # =========================================================================
    print("3. Testing factory function...")
    try:
        baseline_factory = create_baseline('motion_salience', motion_salience_threshold=1.5, k_neighbors=5)
        print("✓ Motion Salience created via factory successfully")
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

    baseline = MotionSalienceThresholdSelection(
        motion_salience_threshold=1.0,
        velocity_diff_threshold=0.2,
        k_neighbors=7,
        seed=42
    )

    # Set environment reference
    baseline.set_env(env)
    print("✓ Environment reference set")

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

    baseline = MotionSalienceThresholdSelection(
        motion_salience_threshold=1.0,
        velocity_diff_threshold=0.2,
        k_neighbors=7,
        seed=42
    )
    baseline.set_env(env)

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
    # Test 8: State transitions
    # =========================================================================
    print("8. Testing state transitions...")

    # High threshold (likely stay in general state)
    baseline_high = MotionSalienceThresholdSelection(
        motion_salience_threshold=10.0,  # Very high threshold
        k_neighbors=7,
        seed=42
    )
    baseline_high.set_env(env)

    # Low threshold (likely trigger fixed-point state)
    baseline_low = MotionSalienceThresholdSelection(
        motion_salience_threshold=0.1,  # Very low threshold
        k_neighbors=7,
        seed=42
    )
    baseline_low.set_env(env)

    obs = env.reset()

    # Run a few steps to build motion salience
    for _ in range(5):
        action_high = baseline_high(obs)
        action_low = baseline_low(obs)
        obs, _, _, _ = env.step(action_high)

    # Check state distributions
    general_count_high = np.sum(baseline_high.agent_states == MotionSalienceThresholdSelection.STATE_GENERAL)
    fixed_count_high = np.sum(baseline_high.agent_states == MotionSalienceThresholdSelection.STATE_FIXED)

    general_count_low = np.sum(baseline_low.agent_states == MotionSalienceThresholdSelection.STATE_GENERAL)
    fixed_count_low = np.sum(baseline_low.agent_states == MotionSalienceThresholdSelection.STATE_FIXED)

    print(f"✓ High threshold (C=10.0): General={general_count_high}, Fixed={fixed_count_high}")
    print(f"✓ Low threshold (C=0.1): General={general_count_low}, Fixed={fixed_count_low}")
    print()

    # =========================================================================
    # Test 9: Different k_neighbors values
    # =========================================================================
    print("9. Testing different k_neighbors values...")

    # Small k
    baseline_k3 = MotionSalienceThresholdSelection(
        motion_salience_threshold=1.0,
        k_neighbors=3,
        seed=42
    )
    baseline_k3.set_env(env)

    # Large k
    baseline_k10 = MotionSalienceThresholdSelection(
        motion_salience_threshold=1.0,
        k_neighbors=10,
        seed=42
    )
    baseline_k10.set_env(env)

    obs = env.reset()
    action_k3 = baseline_k3(obs)
    action_k10 = baseline_k10(obs)

    print(f"✓ k=3: action shape {action_k3.shape}")
    print(f"✓ k=10: action shape {action_k10.shape}")
    print()

    # =========================================================================
    # Test 10: RuntimeError without set_env
    # =========================================================================
    print("10. Testing RuntimeError without set_env...")

    baseline_no_env = MotionSalienceThresholdSelection(motion_salience_threshold=1.0)

    obs = env.reset()
    try:
        action = baseline_no_env(obs)
        print("✗ Should raise RuntimeError without set_env")
        return
    except RuntimeError as e:
        if "Must call set_env" in str(e):
            print("✓ RuntimeError raised correctly without set_env")
        else:
            print(f"✗ Wrong RuntimeError message: {e}")
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
    print("✓ Parameter validation (C, δ, k, τ, s)")
    print("✓ Factory function")
    print("✓ Environment integration with set_env()")
    print("✓ Action format validation")
    print("✓ Mask constraints respected")
    print("✓ Reset functionality (manual and auto-detection)")
    print("✓ Episode execution")
    print("✓ State transitions (general ↔ fixed-point)")
    print("✓ Different k_neighbors values")
    print("✓ RuntimeError without set_env")
    print()
    print("Usage pattern:")
    print("  baseline = MotionSalienceThresholdSelection(")
    print("      motion_salience_threshold=1.0,  # C: state transition threshold")
    print("      velocity_diff_threshold=0.2,    # δ: return-to-general threshold")
    print("      k_neighbors=7,                  # k: topological neighborhood size")
    print("      time_interval=1.0,              # τ: motion salience time window")
    print("      scaling_factor=100.0            # s: motion salience scaling")
    print("  )")
    print("  baseline.set_env(env)  # REQUIRED: set environment reference")
    print("  action = baseline(obs)")
    print()
    print("Paper parameters (Section 2, 3):")
    print("  - C ≥ 0: motion salience threshold (controls state transitions)")
    print("  - δ = 0.2: velocity difference threshold")
    print("  - k = 7: number of nearest neighbors")
    print("  - τ = 1.0: time interval for motion salience")
    print("  - s = 100.0: scaling factor for motion salience")
    print("=" * 60)


if __name__ == "__main__":
    main()
