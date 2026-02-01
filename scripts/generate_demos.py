"""
Generate demonstration data for counterfactual reference policy.
Creates pi_cf_*.pt files in data/demos/
Handles both old and new gymnasium APIs.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from esaiv3.env_wrappers import make_env
from esaiv3.logging_utils import ensure_dir
from esaiv3.utils import set_seed


class OptimalPolicy:
    """
    Optimal demonstration policy that minimizes harm for each environment.
    """
    
    def __init__(self, env_name: str, config: Dict[str, Any]):
        self.env_name = env_name
        self.config = config
        self.step_count = 0
        
    def reset(self):
        """Reset policy state."""
        self.step_count = 0
    
    def get_action(self, obs: np.ndarray, info: Dict[str, Any] = None) -> int:
        """
        Get optimal action based on environment and state.
        
        Args:
            obs: Current observation
            info: Additional info from environment
            
        Returns:
            Optimal action (minimizes harm)
        """
        self.step_count += 1
        
        if self.env_name == "moral_temptation":
            # Moral Temptation: Prioritize helping over stealing
            # obs[0:2] = relative position to victim
            # obs[2:4] = agent position
            # obs[4] = distance to victim
            
            # If close to victim, help
            if obs[4] < 0.3:  # Close to victim
                if np.random.random() < 0.9:  # 90% help rate
                    return 4  # help action
            
            # Otherwise move toward victim
            dx = obs[0]  # Relative x to victim
            dy = obs[1]  # Relative y to victim
            
            if abs(dx) > abs(dy):
                # Move horizontally
                return 0 if dx < 0 else 1  # up/down
            else:
                # Move vertically
                return 2 if dy < 0 else 3  # left/right
        
        elif self.env_name == "social_distress":
            # Social Distress: Help distressed agents
            # obs[2] = own distress level
            # obs[18] = global distress level
            
            # If global distress is high, assist
            if obs[18] > 0.3 or obs[2] < 0.2:
                if np.random.random() < 0.8:  # 80% assist rate
                    return 4  # assist action
            
            # Otherwise move to find distressed agents
            return np.random.randint(0, 4)  # random movement
        
        elif self.env_name == "mpe":
            # MPE: Cooperative navigation (avoid collisions)
            # Simple policy: move away from other agents
            return np.random.randint(0, 5)
        
        elif self.env_name == "overcooked":
            # Overcooked: Coordinate to deliver dishes
            # Simplified: alternate between stations
            if self.step_count % 20 < 10:
                return np.random.randint(0, 4)  # movement
            else:
                return 4  # interact
        
        elif self.env_name == "ssd":
            # SSD: Sustainable harvesting
            # Harvest conservatively
            if np.random.random() < 0.2:  # 20% harvest rate (sustainable)
                return 4  # harvest
            else:
                return np.random.randint(0, 4)  # movement
        
        else:
            # Unknown environment: random but prefer non-harmful
            return np.random.randint(0, 4)  # movement actions usually safe


def collect_demonstration(env_name: str, config: Dict[str, Any], 
                         policy: OptimalPolicy, seed: int = None) -> List[Dict]:
    """
    Collect a single demonstration episode.
    
    Args:
        env_name: Name of environment
        config: Environment configuration
        policy: Demonstration policy
        seed: Random seed for episode
        
    Returns:
        List of transitions in the episode
    """
    env = make_env(env_name, config)
    
    # Reset environment with seed
    if seed is not None:
        reset_result = env.reset(seed=seed)
    else:
        reset_result = env.reset()
    
    # Handle both old and new gymnasium API
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    policy.reset()
    episode_data = []
    terminated = truncated = False
    total_harm = 0.0
    
    while not (terminated or truncated):
        # Get optimal action
        action = policy.get_action(obs, info)
        
        # Step environment
        step_result = env.step(action)
        
        # Handle both old (4 returns) and new (5 returns) gymnasium API
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            next_obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
        
        # Extract harm from info
        harm = info.get('harm_t', 0.0)
        total_harm += harm
        
        # Store transition
        episode_data.append({
            'obs': obs.copy(),
            'action': action,
            'reward': reward,
            'harm': harm,
            'done': done,
            'pr_flag': info.get('PR_flag', None)
        })
        
        obs = next_obs
    
    # Add episode statistics
    for transition in episode_data:
        transition['episode_harm'] = total_harm
        transition['episode_length'] = len(episode_data)
    
    return episode_data


def generate_demos_for_env(env_name: str, env_config: Dict[str, Any], 
                          num_episodes: int = 100) -> Tuple[List[List[Dict]], Dict]:
    """
    Generate demonstrations for a specific environment.
    
    Args:
        env_name: Name of environment
        env_config: Configuration for environment
        num_episodes: Number of episodes to generate
        
    Returns:
        demonstrations: List of episodes
        stats: Statistics about demonstrations
    """
    # Create optimal policy
    policy = OptimalPolicy(env_name, env_config)
    
    demonstrations = []
    total_rewards = []
    total_harms = []
    prosocial_counts = []
    
    print(f"\n[demos] Generating {num_episodes} demonstrations for {env_name}...")
    
    for ep in tqdm(range(num_episodes), desc=f"Generating {env_name} demos"):
        # Use different seed for each episode
        seed = 42 + ep
        
        try:
            episode_data = collect_demonstration(env_name, env_config, policy, seed)
            
            # Compute episode statistics
            episode_reward = sum(t['reward'] for t in episode_data)
            episode_harm = sum(t['harm'] for t in episode_data)
            prosocial_actions = sum(1 for t in episode_data if t['pr_flag'] == 'help')
            
            demonstrations.append(episode_data)
            total_rewards.append(episode_reward)
            total_harms.append(episode_harm)
            prosocial_counts.append(prosocial_actions)
            
        except Exception as e:
            print(f"\n[demos] Warning: Episode {ep} failed: {e}")
            continue
    
    # Compute statistics
    stats = {
        'num_episodes': len(demonstrations),
        'avg_reward': np.mean(total_rewards) if total_rewards else 0,
        'std_reward': np.std(total_rewards) if total_rewards else 0,
        'avg_harm': np.mean(total_harms) if total_harms else 0,
        'std_harm': np.std(total_harms) if total_harms else 0,
        'avg_prosocial': np.mean(prosocial_counts) if prosocial_counts else 0,
        'total_transitions': sum(len(ep) for ep in demonstrations)
    }
    
    print(f"[demos] Generated {len(demonstrations)} episodes for {env_name}")
    print(f"[demos] Stats: Avg reward={stats['avg_reward']:.2f}, "
          f"Avg harm={stats['avg_harm']:.2f}, "
          f"Avg prosocial={stats['avg_prosocial']:.1f}")
    
    return demonstrations, stats


def save_demonstrations(env_name: str, demonstrations: List[List[Dict]], 
                       stats: Dict, config: Dict[str, Any]):
    """
    Save demonstrations to disk.
    
    Args:
        env_name: Name of environment
        demonstrations: List of episodes
        stats: Statistics about demonstrations
        config: Environment configuration
    """
    save_path = f'data/demos/pi_cf_{env_name}.pt'
    
    # Prepare data for saving
    save_data = {
        'demonstrations': demonstrations,
        'env_name': env_name,
        'config': config,
        'num_episodes': len(demonstrations),
        'stats': stats,
        'version': '2.0'  # Track demo format version
    }
    
    # Save to disk
    torch.save(save_data, save_path)
    print(f"[demos] Saved {len(demonstrations)} episodes to {save_path}")
    
    # Also save in numpy format for compatibility
    np_save_path = f'data/demos/pi_cf_{env_name}.npz'
    
    # Flatten all observations and actions
    all_obs = []
    all_actions = []
    all_rewards = []
    all_harms = []
    
    for episode in demonstrations:
        for transition in episode:
            all_obs.append(transition['obs'])
            all_actions.append(transition['action'])
            all_rewards.append(transition['reward'])
            all_harms.append(transition['harm'])
    
    np.savez_compressed(
        np_save_path,
        observations=np.array(all_obs),
        actions=np.array(all_actions),
        rewards=np.array(all_rewards),
        harms=np.array(all_harms),
        stats=stats
    )
    print(f"[demos] Also saved numpy format to {np_save_path}")


def main():
    """Generate demonstrations for all environments."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Ensure demo directory exists
    ensure_dir('data/demos')
    
    # Environment configurations with proper settings
    environments = {
        'moral_temptation': {
            'grid_size': 8, 
            'max_episode_steps': 200,
            'help_reward': 2.0,  # Make helping more attractive
            'steal_reward': 3.0,  # Reduce temptation
            'victim_distress': 1.0,
            'random_start_position': True,
            'random_victim_position': True,
            'random_resource_positions': True,
            'debug_mode': False
        },
        'social_distress': {
            'num_agents': 16, 
            'grid_size': 8,
            'max_episode_steps': 300,
            'shock_interval': [40, 60],
            'shock_magnitude': [2.0, 4.0],
            'help_reward': 2.0,
            'random_start_positions': True,
            'debug_mode': False
        },
        # Commented out environments that need additional dependencies
        # 'mpe': {
        #     'num_agents': 4, 
        #     'max_episode_steps': 100
        # },
        # 'overcooked': {
        #     'num_agents': 2, 
        #     'max_episode_steps': 400
        # },
        # 'ssd': {
        #     'num_agents': 5, 
        #     'max_episode_steps': 300
        # }
    }
    
    all_stats = {}
    
    for env_name, config in environments.items():
        config['env'] = env_name  # Add env name to config
        
        try:
            # Generate demonstrations
            demonstrations, stats = generate_demos_for_env(
                env_name, config, num_episodes=100
            )
            
            # Save demonstrations
            save_demonstrations(env_name, demonstrations, stats, config)
            
            all_stats[env_name] = stats
            
        except Exception as e:
            print(f"[demos] ERROR: Could not generate demos for {env_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create placeholder file
            print(f"[demos] Creating placeholder for {env_name}...")
            placeholder_data = {
                'demonstrations': [],
                'env_name': env_name,
                'config': config,
                'num_episodes': 0,
                'stats': {'error': str(e)},
                'placeholder': True
            }
            torch.save(placeholder_data, f'data/demos/pi_cf_{env_name}.pt')
    
    # Print summary
    print("\n" + "="*60)
    print("[demos] Demo generation complete!")
    print("="*60)
    
    for env_name, stats in all_stats.items():
        print(f"\n{env_name}:")
        print(f"  Episodes: {stats['num_episodes']}")
        print(f"  Avg reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Avg harm: {stats['avg_harm']:.2f} ± {stats['std_harm']:.2f}")
        print(f"  Avg prosocial: {stats['avg_prosocial']:.1f}")
        print(f"  Total transitions: {stats['total_transitions']}")
    
    print("\n[demos] All demonstrations saved to data/demos/")


if __name__ == '__main__':
    main()