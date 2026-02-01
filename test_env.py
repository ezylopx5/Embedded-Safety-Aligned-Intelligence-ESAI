"""Test if environments have proper randomization."""

import numpy as np
from esaiv3.env_wrappers import make_env

def test_env_variance(env_name, num_episodes=10):
    """Test if environment has variance."""
    env = make_env(env_name, {})
    
    print(f"\nTesting {env_name}...")
    
    # Collect initial states
    initial_obs = []
    for _ in range(num_episodes):
        obs = env.reset(seed=np.random.randint(0, 1000000))
        initial_obs.append(obs)
    
    # Check if all initial states are identical
    all_same = all(np.array_equal(initial_obs[0], obs) for obs in initial_obs)
    if all_same:
        print(f"  ❌ FAIL: All initial observations identical!")
    else:
        print(f"  ✅ PASS: Initial observations vary")
    
    # Test rewards
    total_rewards = []
    for _ in range(num_episodes):
        obs = env.reset(seed=np.random.randint(0, 1000000))
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"  Episode rewards: {mean_reward:.2f} ± {std_reward:.2f}")
    if std_reward < 0.01:
        print(f"  ❌ FAIL: No reward variance!")
    else:
        print(f"  ✅ PASS: Rewards have variance")
    
    return std_reward > 0.01

# Test both environments
test_env_variance("moral_temptation")
test_env_variance("social_distress")