"""
Monte Carlo evaluation comparing trained RL policy vs Pure ACS (no neighbor selection).

Compares:
1. Trained Centralized RL Policy (from checkpoint)
2. Pure ACS (Fully Connected Network - no neighbor selection)

Metrics:
- Spatial Entropy (lower is better - more cohesion)
- Velocity Entropy (lower is better - more alignment)  
- Control Cost (lower is better - less effort)
- Episode Return
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from envs.env import NeighborSelectionFlockingEnv, load_config
from models.ppo_centralized import NeighborSelectionPPORLlibCentralized


def find_latest_checkpoint(exp_dir: str) -> str:
    """Find the latest checkpoint directory in the experiment folder."""
    trial_dirs = [d for d in os.listdir(exp_dir) if d.startswith("PPO_")]
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found in {exp_dir}")
    
    trial_dir = os.path.join(exp_dir, trial_dirs[0])
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(trial_dir) if d.startswith("checkpoint_")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {trial_dir}")
    
    # Get the one with highest number
    checkpoint_nums = [(d, int(d.split("_")[1])) for d in checkpoint_dirs]
    latest = max(checkpoint_nums, key=lambda x: x[1])
    
    return os.path.join(trial_dir, latest[0])


def create_env(config_overrides: dict = None):
    """Create environment with optional config overrides."""
    default_config_path = "./envs/default_env_config.yaml"
    my_config = load_config(default_config_path)
    
    # Default settings for evaluation
    my_config.env.observation_type = "centralized"
    my_config.env.num_agents_pool = [20]
    my_config.env.max_time_steps = 1000
    my_config.env.use_fixed_episode_length = True
    my_config.env.is_training = False
    my_config.env.get_state_hist = False
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(my_config.env, key):
                setattr(my_config.env, key, value)
    
    env_config = {"seed_id": None, "config": my_config.dict()}
    return NeighborSelectionFlockingEnv(env_config)


class RLPolicy:
    """Wrapper for RL policy loaded from checkpoint."""
    
    def __init__(self, checkpoint_path: str, env):
        """Load model weights from checkpoint."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get model config from checkpoint params
        params_path = os.path.join(os.path.dirname(checkpoint_path), "params.json")
        import json
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        model_config = params.get("model", {}).get("custom_model_config", {})
        
        # Create model with same config
        obs_space = env.observation_space
        action_space = env.action_space
        
        # Calculate num_outputs based on action space
        if hasattr(action_space, 'n'):
            num_outputs = action_space.n
        elif hasattr(action_space, 'nvec'):
            # MultiDiscrete: N*N binary decisions, each with 2 options
            num_outputs = int(np.sum(action_space.nvec))
        else:
            # Assume Box action space - get shape
            num_outputs = int(np.prod(action_space.shape) * 2)
        
        self.model = NeighborSelectionPPORLlibCentralized(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config={"custom_model_config": model_config},
            name="centralized_policy"
        )
        self.model.to(self.device)
        
        # Load weights from checkpoint
        policy_path = os.path.join(checkpoint_path, "policies", "default_policy")
        self._load_weights(policy_path)
        
        self.model.eval()
    
    def _load_weights(self, policy_path: str):
        """Load weights from RLlib checkpoint."""
        import pickle
        
        policy_state_path = os.path.join(policy_path, "policy_state.pkl")
        
        if not os.path.exists(policy_state_path):
            raise FileNotFoundError(f"Could not find policy_state.pkl in {policy_path}")
        
        with open(policy_state_path, 'rb') as f:
            state = pickle.load(f)
        
        weights = state['weights']
        
        # Convert numpy arrays to torch tensors
        torch_state = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                torch_state[k] = torch.from_numpy(v)
            else:
                torch_state[k] = v
        
        self.model.load_state_dict(torch_state, strict=False)
        print(f"Loaded {len(torch_state)} weight tensors from checkpoint")
    
    def __call__(self, obs: dict) -> np.ndarray:
        """Compute action from observation."""
        with torch.no_grad():
            # Convert obs to tensors with batch dimension
            input_dict = {
                "obs": {
                    "local_agent_infos": torch.from_numpy(
                        obs["local_agent_infos"][np.newaxis]
                    ).float().to(self.device),
                    "neighbor_masks": torch.from_numpy(
                        obs["neighbor_masks"][np.newaxis]
                    ).float().to(self.device),
                    "padding_mask": torch.from_numpy(
                        obs["padding_mask"][np.newaxis]
                    ).float().to(self.device),
                }
            }
            
            # Forward pass
            logits, _ = self.model.forward(input_dict, state=[], seq_lens=None)
            logits = logits.cpu().numpy()[0]  # Remove batch dim
            
            # Convert logits to action
            # logits shape: (N*N*2,) -> reshape to (N, N, 2)
            num_agents_max = obs["padding_mask"].shape[0]
            logits_reshaped = logits.reshape(num_agents_max, num_agents_max, 2)
            
            # Sample action: argmax over last dim (0=not select, 1=select)
            action = np.argmax(logits_reshaped, axis=-1).astype(np.int8)
            
            return action


class PureACSPolicy:
    """
    Pure ACS baseline - no neighbor selection.
    Uses fully connected network (all agents communicate with all).
    """
    
    def __call__(self, obs: dict) -> np.ndarray:
        """Return fully connected action (all 1s with self-loops)."""
        padding_mask = obs['padding_mask']
        num_agents_max = len(padding_mask)
        
        # Fully connected: all agents connected to all
        action = np.ones((num_agents_max, num_agents_max), dtype=np.int8)
        
        # Zero out connections to/from padding agents
        padding_mask_2d = padding_mask[:, np.newaxis] & padding_mask[np.newaxis, :]
        action = action * padding_mask_2d.astype(np.int8)
        
        # Ensure self-loops for active agents
        action[np.arange(num_agents_max), np.arange(num_agents_max)] = padding_mask.astype(np.int8)
        
        return action


def run_episode(env, policy, collect_trajectory=False):
    """
    Run a single episode and collect metrics.
    
    Args:
        env: Environment
        policy: Policy to use
        collect_trajectory: If True, collect agent positions at each step
    
    Returns:
        dict: Episode metrics (and trajectory if collect_trajectory=True)
    """
    obs = env.reset()
    done = False
    
    episode_return = 0.0
    spatial_entropies = []
    velocity_entropies = []
    control_costs = []
    steps = 0
    
    # Collect trajectory if requested
    trajectory = [] if collect_trajectory else None
    if collect_trajectory:
        # Store initial positions
        positions = env.state["agent_states"][:, :2].copy()  # (N, 2)
        padding_mask = env.state["padding_mask"].copy()
        trajectory.append((positions, padding_mask))
    
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        
        episode_return += reward
        steps += 1
        
        # Collect trajectory
        if collect_trajectory:
            positions = env.state["agent_states"][:, :2].copy()
            padding_mask = env.state["padding_mask"].copy()
            trajectory.append((positions, padding_mask))
        
        # Collect metrics
        if info.get('spatial_entropy') is not None:
            spatial_entropies.append(info['spatial_entropy'])
        if info.get('velocity_entropy') is not None:
            velocity_entropies.append(info['velocity_entropy'])
        if 'original_reward' in info:
            control_costs.append(-info['original_reward'])  # Reward is negative cost
    
    result = {
        'episode_return': episode_return,
        'episode_length': steps,
        'spatial_entropy_mean': np.mean(spatial_entropies) if spatial_entropies else np.nan,
        'spatial_entropy_final': spatial_entropies[-1] if spatial_entropies else np.nan,
        'velocity_entropy_mean': np.mean(velocity_entropies) if velocity_entropies else np.nan,
        'velocity_entropy_final': velocity_entropies[-1] if velocity_entropies else np.nan,
        'control_cost_mean': np.mean(control_costs) if control_costs else np.nan,
        'control_cost_total': np.sum(control_costs) if control_costs else np.nan,
    }
    
    if collect_trajectory:
        result['trajectory'] = trajectory
    
    return result


def plot_trajectory_comparison(acs_trajectory, rl_trajectory, episode_idx, save_dir, 
                                acs_return, rl_return):
    """
    Plot side-by-side trajectory comparison.
    
    Args:
        acs_trajectory: List of (positions, padding_mask) tuples for ACS
        rl_trajectory: List of (positions, padding_mask) tuples for RL
        episode_idx: Episode index for filename
        save_dir: Directory to save the plot
        acs_return: Episode return for ACS
        rl_return: Episode return for RL
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different agents
    num_agents = np.sum(acs_trajectory[0][1])  # Count active agents from padding mask
    colors = plt.cm.tab20(np.linspace(0, 1, num_agents))
    
    for ax, trajectory, title, ep_return in [
        (axes[0], acs_trajectory, "Pure ACS (Fully Connected)", acs_return),
        (axes[1], rl_trajectory, "RL Policy", rl_return)
    ]:
        # Extract positions for each agent over time
        padding_mask = trajectory[0][1]
        active_indices = np.where(padding_mask)[0]
        
        for i, agent_idx in enumerate(active_indices):
            # Get trajectory for this agent
            agent_positions = np.array([t[0][agent_idx] for t in trajectory])
            
            # Plot trajectory line
            ax.plot(agent_positions[:, 0], agent_positions[:, 1], 
                    color=colors[i], alpha=0.6, linewidth=0.8)
            
            # Mark start position (circle)
            ax.scatter(agent_positions[0, 0], agent_positions[0, 1], 
                       color=colors[i], marker='o', s=50, edgecolors='black', linewidths=0.5)
            
            # Mark end position (triangle)
            ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1], 
                       color=colors[i], marker='^', s=80, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'{title}\nReturn: {ep_return:.2f}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Episode {episode_idx + 1}: Agent Trajectories\n(○ = start, △ = end)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'trajectory_episode_{episode_idx + 1:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def monte_carlo_evaluation(env, policy, num_episodes: int, desc="Evaluating", 
                           collect_trajectory=False):
    """
    Run Monte Carlo evaluation over multiple episodes.
    
    Args:
        env: Environment
        policy: Policy to use
        num_episodes: Number of episodes to run
        desc: Description for progress bar
        collect_trajectory: If True, collect and return trajectories
    
    Returns:
        dict: Aggregated statistics
        list: List of trajectories (if collect_trajectory=True)
    """
    results = defaultdict(list)
    trajectories = [] if collect_trajectory else None
    
    for _ in tqdm(range(num_episodes), desc=desc):
        episode_result = run_episode(env, policy, collect_trajectory=collect_trajectory)
        
        if collect_trajectory and 'trajectory' in episode_result:
            trajectories.append(episode_result.pop('trajectory'))
        
        for key, value in episode_result.items():
            results[key].append(value)
    
    # Compute statistics
    stats = {}
    for key, values in results.items():
        values = np.array(values)
        stats[f'{key}_mean'] = np.mean(values)
        stats[f'{key}_std'] = np.std(values)
        stats[f'{key}_min'] = np.min(values)
        stats[f'{key}_max'] = np.max(values)
    
    # Also store raw episode returns for trajectory plotting
    stats['_episode_returns'] = results['episode_return']
    
    if collect_trajectory:
        return stats, trajectories
    return stats


def print_comparison(rl_stats: dict, acs_stats: dict):
    """Print comparison table."""
    print("\n" + "="*80)
    print("MONTE CARLO EVALUATION RESULTS")
    print("="*80)
    
    metrics = [
        ('Episode Return', 'episode_return'),
        ('Episode Length', 'episode_length'),
        ('Spatial Entropy (Final)', 'spatial_entropy_final'),
        ('Velocity Entropy (Final)', 'velocity_entropy_final'),
        ('Spatial Entropy (Mean)', 'spatial_entropy_mean'),
        ('Velocity Entropy (Mean)', 'velocity_entropy_mean'),
        ('Control Cost (Total)', 'control_cost_total'),
    ]
    
    print(f"\n{'Metric':<30} {'RL Policy':<25} {'Pure ACS':<25} {'Diff':<15}")
    print("-"*95)
    
    for name, key in metrics:
        rl_mean = rl_stats.get(f'{key}_mean', np.nan)
        rl_std = rl_stats.get(f'{key}_std', np.nan)
        acs_mean = acs_stats.get(f'{key}_mean', np.nan)
        acs_std = acs_stats.get(f'{key}_std', np.nan)
        
        diff = rl_mean - acs_mean
        diff_pct = (diff / abs(acs_mean) * 100) if acs_mean != 0 else 0
        
        rl_str = f"{rl_mean:.4f} ± {rl_std:.4f}"
        acs_str = f"{acs_mean:.4f} ± {acs_std:.4f}"
        diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
        
        print(f"{name:<30} {rl_str:<25} {acs_str:<25} {diff_str:<15}")
    
    print("="*80)
    print("\nNote: For entropy metrics, LOWER is better (more cohesion/alignment)")
    print("      For episode return, HIGHER is better")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of RL vs Pure ACS")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If not provided, uses latest from ray_results")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes for Monte Carlo evaluation")
    parser.add_argument("--num_agents", type=int, default=20,
                        help="Number of agents")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_trajectory", action="store_true",
                        help="Save trajectory plots to evaluate_results directory")
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Find checkpoint
    if args.checkpoint is None:
        exp_dir = "./ray_results/centralized_legacy_style"
        checkpoint_path = find_latest_checkpoint(exp_dir)
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
    
    # Create output directory for trajectory plots
    save_dir = None
    if args.save_trajectory:
        save_dir = "./evaluate_results"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Trajectory plots will be saved to: {save_dir}")
    
    # Create environment
    config_overrides = {
        'num_agents_pool': [args.num_agents],
        'max_time_steps': args.max_steps,
    }
    env = create_env(config_overrides)
    
    print(f"\n{'='*60}")
    print("EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Num Episodes: {args.num_episodes}")
    print(f"  Num Agents: {args.num_agents}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Save Trajectory: {args.save_trajectory}")
    print(f"{'='*60}\n")
    
    # Load RL policy
    print("Loading RL policy from checkpoint...")
    try:
        rl_policy = RLPolicy(checkpoint_path, env)
    except Exception as e:
        print(f"Error loading RL policy: {e}")
        print("Falling back to Pure ACS only evaluation...")
        rl_policy = None
    
    # Create baseline policy
    acs_policy = PureACSPolicy()
    
    # For trajectory comparison, we need to run episodes with same initial conditions
    if args.save_trajectory and rl_policy is not None:
        print("\n[1/2] Running paired evaluation with trajectory collection...")
        
        rl_results = defaultdict(list)
        acs_results = defaultdict(list)
        
        for ep_idx in tqdm(range(args.num_episodes), desc="Paired Episodes"):
            # Set seed for reproducible initial conditions
            episode_seed = args.seed + ep_idx
            np.random.seed(episode_seed)
            
            # Run ACS episode
            env.seed(episode_seed)
            acs_result = run_episode(env, acs_policy, collect_trajectory=True)
            acs_trajectory = acs_result.pop('trajectory')
            for k, v in acs_result.items():
                acs_results[k].append(v)
            
            # Run RL episode with same initial condition
            env.seed(episode_seed)
            rl_result = run_episode(env, rl_policy, collect_trajectory=True)
            rl_trajectory = rl_result.pop('trajectory')
            for k, v in rl_result.items():
                rl_results[k].append(v)
            
            # Plot trajectory comparison
            plot_trajectory_comparison(
                acs_trajectory, rl_trajectory, ep_idx, save_dir,
                acs_result['episode_return'], rl_result['episode_return']
            )
        
        # Compute statistics
        rl_stats = {}
        for key, values in rl_results.items():
            values = np.array(values)
            rl_stats[f'{key}_mean'] = np.mean(values)
            rl_stats[f'{key}_std'] = np.std(values)
            rl_stats[f'{key}_min'] = np.min(values)
            rl_stats[f'{key}_max'] = np.max(values)
        
        acs_stats = {}
        for key, values in acs_results.items():
            values = np.array(values)
            acs_stats[f'{key}_mean'] = np.mean(values)
            acs_stats[f'{key}_std'] = np.std(values)
            acs_stats[f'{key}_min'] = np.min(values)
            acs_stats[f'{key}_max'] = np.max(values)
        
        print(f"\nSaved {args.num_episodes} trajectory plots to {save_dir}")
        
    else:
        # Standard evaluation without trajectory
        if rl_policy is not None:
            print("\n[1/2] Evaluating RL Policy...")
            rl_stats = monte_carlo_evaluation(env, rl_policy, args.num_episodes, desc="RL Policy")
        else:
            rl_stats = {}
        
        print("\n[2/2] Evaluating Pure ACS (Fully Connected)...")
        acs_stats = monte_carlo_evaluation(env, acs_policy, args.num_episodes, desc="Pure ACS")
    
    # Print comparison
    if rl_policy is not None:
        print_comparison(rl_stats, acs_stats)
    else:
        print("\n" + "="*60)
        print("PURE ACS RESULTS (RL policy not loaded)")
        print("="*60)
        for key, value in acs_stats.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
