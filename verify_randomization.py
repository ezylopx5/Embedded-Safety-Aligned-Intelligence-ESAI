"""
Verify environment randomization is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from esaiv3.env_wrappers import make_env

def test_env_randomization(env_name, num_episodes=50):
    """Test if environment properly randomizes across episodes."""
    
    print(f"\n{'='*60}")
    print(f"Testing {env_name} randomization")
    print(f"{'='*60}\n")
    
    # Create environment
    config = {
        'grid_size': 8,
        'help_reward': 1.0,
        'steal_reward': 6.0,
        'victim_distress': 3.0,
        'max_episode_steps': 200,
        'num_agents': 16,
        'shock_interval': 50,
        'shock_magnitude': 3.0
    }
    
    env = make_env(env_name, config)
    
    # Collect initial observations
    initial_obs_list = []
    episode_rewards = []
    episode_lengths = []
    
    print(f"Running {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        # Reset with different seeds
        obs, info = env.reset(seed=ep * 1000)
        initial_obs_list.append(obs.copy())
        
        done = False
        episode_reward = 0
        steps = 0
        
        # Run random policy
        while not done and steps < 100:
            action = env.action_space.sample()
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
    
    # Analysis
    initial_obs_array = np.array(initial_obs_list)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Check observation variance
    obs_variance = np.var(initial_obs_array, axis=0)
    obs_mean_variance = np.mean(obs_variance)
    
    print(f"\n1. Initial Observation Variance:")
    print(f"   Mean variance across features: {obs_mean_variance:.6f}")
    print(f"   Min variance: {np.min(obs_variance):.6f}")
    print(f"   Max variance: {np.max(obs_variance):.6f}")
    
    if obs_mean_variance < 1e-6:
        print("   ⚠️  WARNING: Initial observations are nearly identical!")
        print("   → Environment is NOT randomizing properly!")
    else:
        print("   ✓ Initial observations vary across episodes")
    
    # Check reward variance
    reward_mean = np.mean(episode_rewards)
    reward_std = np.std(episode_rewards)
    reward_variance = np.var(episode_rewards)
    
    print(f"\n2. Episode Reward Statistics (Random Policy):")
    print(f"   Mean: {reward_mean:.2f}")
    print(f"   Std: {reward_std:.2f}")
    print(f"   Variance: {reward_variance:.2f}")
    print(f"   Range: [{np.min(episode_rewards):.2f}, {np.max(episode_rewards):.2f}]")
    
    if reward_std < 0.1:
        print("   ⚠️  WARNING: Rewards have almost no variance!")
        print("   → Environment may be deterministic or broken!")
    else:
        print("   ✓ Rewards vary across episodes")
    
    # Check length variance
    length_std = np.std(episode_lengths)
    
    print(f"\n3. Episode Length Statistics:")
    print(f"   Mean: {np.mean(episode_lengths):.1f}")
    print(f"   Std: {length_std:.1f}")
    print(f"   Range: [{np.min(episode_lengths)}, {np.max(episode_lengths)}]")
    
    # Check for identical episodes
    print(f"\n4. Checking for Duplicate Episodes:")
    unique_obs = np.unique(initial_obs_array, axis=0)
    num_unique = len(unique_obs)
    duplicate_rate = 1.0 - (num_unique / num_episodes)
    
    print(f"   Unique initial states: {num_unique}/{num_episodes}")
    print(f"   Duplicate rate: {duplicate_rate*100:.1f}%")
    
    if duplicate_rate > 0.5:
        print("   ⚠️  WARNING: More than 50% of episodes have identical starts!")
    else:
        print("   ✓ Most episodes have unique starting states")
    
    # Visualization
    print(f"\n5. Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Initial observation variance
    ax = axes[0, 0]
    ax.bar(range(len(obs_variance)), obs_variance)
    ax.set_xlabel('Observation Feature Index')
    ax.set_ylabel('Variance')
    ax.set_title('Variance of Initial Observations Across Episodes')
    ax.axhline(y=1e-6, color='r', linestyle='--', label='Near-zero threshold')
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 2: Episode rewards
    ax = axes[0, 1]
    ax.plot(episode_rewards, 'o-', alpha=0.6)
    ax.axhline(y=reward_mean, color='r', linestyle='--', label=f'Mean: {reward_mean:.2f}')
    ax.fill_between(range(num_episodes), 
                     reward_mean - reward_std, 
                     reward_mean + reward_std, 
                     alpha=0.3, label=f'±1 Std: {reward_std:.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards (Random Policy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Initial obs first 2 dimensions (scatter)
    ax = axes[1, 0]
    if initial_obs_array.shape[1] >= 2:
        ax.scatter(initial_obs_array[:, 0], initial_obs_array[:, 1], alpha=0.6)
        ax.set_xlabel('Observation Feature 0')
        ax.set_ylabel('Observation Feature 1')
        ax.set_title('Initial Observation Distribution (First 2 Features)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Reward histogram
    ax = axes[1, 1]
    ax.hist(episode_rewards, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(x=reward_mean, color='r', linestyle='--', label=f'Mean: {reward_mean:.2f}')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'randomization_check_{env_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   Saved plot to: {filename}")
    
    # Final verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    issues = []
    
    if obs_mean_variance < 1e-6:
        issues.append("Initial observations not randomizing")
    if reward_std < 0.1:
        issues.append("Rewards have no variance")
    if duplicate_rate > 0.5:
        issues.append("High duplicate episode rate")
    
    if issues:
        print("\n❌ PROBLEMS DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n→ Environment randomization is NOT working properly!")
        print("→ Training will likely fail or produce degenerate policies.")
    else:
        print("\n✅ Environment randomization is working correctly!")
        print("→ Episodes are diverse and stochastic.")
        print("→ Ready for training.")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'obs_mean_variance': obs_mean_variance,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'duplicate_rate': duplicate_rate,
        'issues': issues
    }


if __name__ == '__main__':
    import sys
    
    # Test moral_temptation
    results_mt = test_env_randomization('moral_temptation', num_episodes=50)
    
    # Test social_distress
    results_sd = test_env_randomization('social_distress', num_episodes=50)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_good = True
    
    for env_name, results in [('moral_temptation', results_mt), 
                               ('social_distress', results_sd)]:
        print(f"\n{env_name}:")
        if results['issues']:
            print(f"  ❌ {len(results['issues'])} issue(s) found")
            all_good = False
        else:
            print(f"  ✅ All checks passed")
    
    if all_good:
        print("\n✅ All environments are properly randomized!")
        print("→ You can proceed with training.")
        sys.exit(0)
    else:
        print("\n❌ Some environments have issues!")
        print("→ Fix randomization before training.")
        sys.exit(1)