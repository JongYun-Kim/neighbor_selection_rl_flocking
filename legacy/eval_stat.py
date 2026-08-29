"""400-episode evaluation with Welch's t-test."""
import sys, os, json
import numpy as np
from scipy import stats as sp_stats
from legacy.evaluate_checkpoint import (
    create_env, RLPolicy, PureACSPolicy,
    monte_carlo_evaluation, print_comparison
)

def main():
    ckpt = sys.argv[1]
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    params_path = os.path.join(os.path.dirname(ckpt), "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "params.json")
    continuous_action = False
    obs_type = "ego_centric"
    if os.path.exists(params_path):
        p = json.load(open(params_path))
        ec = p.get("env_config", {}).get("config", {}).get("env", {})
        continuous_action = ec.get("continuous_action", False)
        obs_type = ec.get("observation_type", "ego_centric")

    env = create_env({"continuous_action": continuous_action}, observation_type=obs_type)
    rl_policy = RLPolicy(ckpt, env, observation_type=obs_type)
    acs_policy = PureACSPolicy(continuous=continuous_action)

    print(f"Evaluating {n_ep} episodes (deterministic)...")
    print(f"Checkpoint: {ckpt}")

    rl_stats = monte_carlo_evaluation(env, rl_policy, n_ep, desc="RL Policy")
    acs_stats = monte_carlo_evaluation(env, acs_policy, n_ep, desc="FC-ACS")

    print_comparison(rl_stats, acs_stats)

    rl_ret = np.array(rl_stats['_episode_returns'])
    fc_ret = np.array(acs_stats['_episode_returns'])

    t_stat, p_val = sp_stats.ttest_ind(rl_ret, fc_ret, equal_var=False)
    diff = rl_ret.mean() - fc_ret.mean()
    pct = 100 * diff / abs(fc_ret.mean())

    print("\n" + "="*60)
    print("STATISTICAL TEST (Welch's t-test, unpaired)")
    print("="*60)
    print(f"  RL mean:  {rl_ret.mean():.2f} ± {rl_ret.std():.2f}")
    print(f"  FC mean:  {fc_ret.mean():.2f} ± {fc_ret.std():.2f}")
    print(f"  Diff:     {diff:+.2f} ({pct:+.1f}%)")
    print(f"  t-stat:   {t_stat:.4f}")
    print(f"  p-value:  {p_val:.6f}")
    print(f"  Result:   {'SIGNIFICANT (p<0.05)' if p_val < 0.05 else 'NOT significant'}")
    if diff > 0:
        print(f"  --> RL beats FC by {pct:.1f}% (t={t_stat:.2f}, p={p_val:.4f})")
    else:
        print(f"  --> FC beats RL by {-pct:.1f}% (t={t_stat:.2f}, p={p_val:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()
