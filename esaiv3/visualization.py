"""
Real-time Training Visualization for ESAI-v3

This module provides live visualization of:
1. Agent position and victim position in the grid
2. IAE vector heatmap
3. Action distribution
4. Training metrics (reward, prosocial ratio, AR)

Usage:
    from esaiv3.visualization import TrainingVisualizer
    
    viz = TrainingVisualizer(env, agent, update_freq=10)
    
    # In training loop:
    viz.update(obs, action, reward, info, metrics)
    
    # End of training:
    viz.close()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import deque
import time
import os
from typing import Dict, Optional, Any, List

# Try to use non-blocking backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system


class TrainingVisualizer:
    """
    Real-time visualization of ESAI-v3 training.
    
    Features:
    - Grid environment view with agent/victim positions
    - IAE embedding visualization (heatmap)
    - Action probability distribution
    - Rolling metrics plots (reward, PR, AR, entropy)
    - Episode highlights (moral decisions)
    """
    
    def __init__(self, 
                 grid_size: int = 8,
                 iae_dim: int = 32,
                 action_names: List[str] = None,
                 update_freq: int = 10,
                 history_len: int = 500,
                 save_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            grid_size: Size of the environment grid
            iae_dim: Dimension of IAE embedding
            action_names: Names for each action
            update_freq: Update display every N steps
            history_len: Length of rolling history for metrics
            save_dir: Directory to save snapshots (optional)
        """
        self.grid_size = grid_size
        self.iae_dim = iae_dim
        self.update_freq = update_freq
        self.history_len = history_len
        self.save_dir = save_dir
        
        if action_names is None:
            self.action_names = ['↑', '↓', '←', '→', 'HELP', 'STEAL']
        else:
            self.action_names = action_names
        
        # Metrics history
        self.rewards_history = deque(maxlen=history_len)
        self.ar_history = deque(maxlen=history_len)
        self.iae_norm_history = deque(maxlen=history_len)
        self.entropy_history = deque(maxlen=history_len)
        self.pr_history = deque(maxlen=history_len)
        self.help_count = 0
        self.steal_count = 0
        
        # Current state
        self.step_count = 0
        self.episode_count = 0
        self.last_update_time = time.time()
        
        # Setup figure
        self._setup_figure()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def _setup_figure(self):
        """Create the matplotlib figure and axes."""
        plt.ion()  # Interactive mode
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('ESAI-v3 Training Visualization', fontsize=14, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main grid view (large, top-left)
        self.ax_grid = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_grid.set_title('Environment')
        self.ax_grid.set_xlim(0, self.grid_size)
        self.ax_grid.set_ylim(0, self.grid_size)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.grid(True, alpha=0.3)
        
        # IAE heatmap (top-right)
        self.ax_iae = self.fig.add_subplot(gs[0, 2:4])
        self.ax_iae.set_title('IAE Embedding')
        self.iae_data = np.zeros((1, self.iae_dim))
        self.iae_im = self.ax_iae.imshow(self.iae_data, cmap='RdBu_r', 
                                          vmin=-5, vmax=5, aspect='auto')
        self.ax_iae.set_xlabel('Dimension')
        self.ax_iae.set_yticks([])
        plt.colorbar(self.iae_im, ax=self.ax_iae, orientation='horizontal', pad=0.2)
        
        # Action distribution (middle-right)
        self.ax_action = self.fig.add_subplot(gs[1, 2:4])
        self.ax_action.set_title('Action Distribution')
        self.action_bars = self.ax_action.bar(range(len(self.action_names)), 
                                               [0]*len(self.action_names),
                                               color=['gray']*4 + ['green', 'red'])
        self.ax_action.set_xticks(range(len(self.action_names)))
        self.ax_action.set_xticklabels(self.action_names)
        self.ax_action.set_ylim(0, 1)
        self.ax_action.set_ylabel('Probability')
        
        # Reward plot (bottom-left)
        self.ax_reward = self.fig.add_subplot(gs[2, 0])
        self.ax_reward.set_title('Episode Reward')
        self.reward_line, = self.ax_reward.plot([], [], 'b-', linewidth=1)
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        
        # Prosocial ratio plot (bottom-middle-left)
        self.ax_pr = self.fig.add_subplot(gs[2, 1])
        self.ax_pr.set_title('Prosocial Ratio')
        self.pr_line, = self.ax_pr.plot([], [], 'g-', linewidth=1)
        self.ax_pr.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        self.ax_pr.set_xlabel('Episode')
        self.ax_pr.set_ylabel('PR')
        self.ax_pr.set_ylim(0, 1)
        
        # AR plot (bottom-middle-right)
        self.ax_ar = self.fig.add_subplot(gs[2, 2])
        self.ax_ar.set_title('Alignment Regret')
        self.ar_line, = self.ax_ar.plot([], [], 'r-', linewidth=1)
        self.ax_ar.set_xlabel('Step')
        self.ax_ar.set_ylabel('AR')
        
        # IAE norm plot (bottom-right)
        self.ax_norm = self.fig.add_subplot(gs[2, 3])
        self.ax_norm.set_title('IAE Norm')
        self.norm_line, = self.ax_norm.plot([], [], 'm-', linewidth=1)
        self.ax_norm.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Max bound')
        self.ax_norm.set_xlabel('Step')
        self.ax_norm.set_ylabel('||E||')
        
        # Agent and victim markers (will be updated)
        self.agent_marker = None
        self.victim_marker = None
        self.interaction_circle = None
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update(self, 
               agent_pos: np.ndarray,
               victim_pos: np.ndarray,
               action: int,
               action_probs: np.ndarray,
               reward: float,
               info: Dict[str, Any],
               iae: np.ndarray,
               ar: float = 0.0,
               episode_done: bool = False,
               episode_reward: float = 0.0,
               metrics: Optional[Dict] = None):
        """
        Update the visualization.
        
        Args:
            agent_pos: Agent position [x, y]
            victim_pos: Victim position [x, y]
            action: Action taken (int)
            action_probs: Action probability distribution
            reward: Step reward
            info: Environment info dict
            iae: IAE embedding vector
            ar: Alignment regret value
            episode_done: Whether episode ended
            episode_reward: Total episode reward (if done)
            metrics: Additional metrics dict
        """
        self.step_count += 1
        
        # Record metrics
        self.ar_history.append(ar)
        self.iae_norm_history.append(np.linalg.norm(iae))
        
        # Track moral decisions
        pr_flag = info.get('pr_flag')
        if pr_flag == 'help':
            self.help_count += 1
        elif pr_flag == 'harm':
            self.steal_count += 1
        
        if episode_done:
            self.episode_count += 1
            self.rewards_history.append(episode_reward)
            
            # Compute running PR
            total_moral = self.help_count + self.steal_count
            if total_moral > 0:
                pr = self.help_count / total_moral
                self.pr_history.append(pr)
        
        # Only update display every N steps
        if self.step_count % self.update_freq != 0:
            return
        
        # Update grid view
        self._update_grid(agent_pos, victim_pos, info)
        
        # Update IAE heatmap
        self._update_iae(iae)
        
        # Update action distribution
        self._update_actions(action_probs, action)
        
        # Update metric plots
        self._update_metrics()
        
        # Render
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Rate limiting (max 30 FPS)
        elapsed = time.time() - self.last_update_time
        if elapsed < 0.033:  # ~30 FPS
            time.sleep(0.033 - elapsed)
        self.last_update_time = time.time()
    
    def _update_grid(self, agent_pos, victim_pos, info):
        """Update the environment grid view."""
        # Clear previous markers
        if self.agent_marker:
            self.agent_marker.remove()
        if self.victim_marker:
            self.victim_marker.remove()
        if self.interaction_circle:
            self.interaction_circle.remove()
        
        # Draw interaction radius circle around agent
        interaction_radius = info.get('interaction_radius', 1.5)
        self.interaction_circle = plt.Circle(
            agent_pos, interaction_radius,
            fill=False, color='blue', alpha=0.3, linestyle='--'
        )
        self.ax_grid.add_patch(self.interaction_circle)
        
        # Draw agent (blue square)
        self.agent_marker = self.ax_grid.scatter(
            agent_pos[0], agent_pos[1], 
            s=200, c='blue', marker='s', label='Agent', zorder=5
        )
        
        # Draw victim (red circle) - color intensity based on distress
        can_interact = info.get('can_interact', False)
        victim_color = 'darkred' if can_interact else 'lightcoral'
        self.victim_marker = self.ax_grid.scatter(
            victim_pos[0], victim_pos[1],
            s=150, c=victim_color, marker='o', label='Victim', zorder=5
        )
        
        # Update title with step info
        dist = np.linalg.norm(agent_pos - victim_pos)
        status = "IN RANGE" if can_interact else f"dist={dist:.1f}"
        self.ax_grid.set_title(f'Environment (Step {self.step_count}, {status})')
    
    def _update_iae(self, iae):
        """Update the IAE embedding heatmap."""
        self.iae_data = iae.reshape(1, -1)
        self.iae_im.set_data(self.iae_data)
        
        # Auto-scale colormap
        vmax = max(np.abs(iae).max(), 1.0)
        self.iae_im.set_clim(-vmax, vmax)
        
        norm = np.linalg.norm(iae)
        self.ax_iae.set_title(f'IAE Embedding (||E||={norm:.2f})')
    
    def _update_actions(self, probs, action_taken):
        """Update the action distribution bar chart."""
        for i, (bar, p) in enumerate(zip(self.action_bars, probs)):
            bar.set_height(p)
            # Highlight taken action
            if i == action_taken:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
            else:
                bar.set_edgecolor('none')
                bar.set_linewidth(0)
        
        self.ax_action.set_title(f'Action Distribution (took: {self.action_names[action_taken]})')
    
    def _update_metrics(self):
        """Update all metric plots."""
        # Reward history
        if len(self.rewards_history) > 0:
            x = list(range(len(self.rewards_history)))
            self.reward_line.set_data(x, list(self.rewards_history))
            self.ax_reward.relim()
            self.ax_reward.autoscale_view()
        
        # PR history
        if len(self.pr_history) > 0:
            x = list(range(len(self.pr_history)))
            self.pr_line.set_data(x, list(self.pr_history))
            self.ax_pr.relim()
            self.ax_pr.autoscale_view()
            self.ax_pr.set_ylim(0, 1)
        
        # AR history
        if len(self.ar_history) > 0:
            x = list(range(len(self.ar_history)))
            self.ar_line.set_data(x, list(self.ar_history))
            self.ax_ar.relim()
            self.ax_ar.autoscale_view()
        
        # IAE norm history
        if len(self.iae_norm_history) > 0:
            x = list(range(len(self.iae_norm_history)))
            self.norm_line.set_data(x, list(self.iae_norm_history))
            self.ax_norm.relim()
            self.ax_norm.autoscale_view()
    
    def save_snapshot(self, tag: str = ''):
        """Save current visualization to file."""
        if self.save_dir is None:
            return
        
        filename = f'viz_step{self.step_count}'
        if tag:
            filename += f'_{tag}'
        filename += '.png'
        
        filepath = os.path.join(self.save_dir, filename)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved snapshot to {filepath}")
    
    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)
    
    def reset_episode_stats(self):
        """Reset per-episode statistics."""
        self.help_count = 0
        self.steal_count = 0


class MetricsLogger:
    """
    Lightweight metrics logger for when full visualization is not needed.
    Prints formatted progress to console.
    """
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        self.step = 0
        self.episode = 0
        self.help_total = 0
        self.steal_total = 0
        self.rewards = []
    
    def log_step(self, reward: float, info: Dict, ar: float, iae_norm: float):
        """Log a training step."""
        self.step += 1
        
        pr_flag = info.get('pr_flag')
        if pr_flag == 'help':
            self.help_total += 1
        elif pr_flag == 'harm':
            self.steal_total += 1
        
        if self.step % self.log_freq == 0:
            total = self.help_total + self.steal_total
            pr = self.help_total / max(1, total)
            print(f"Step {self.step:6d} | "
                  f"AR={ar:6.3f} | ||E||={iae_norm:5.2f} | "
                  f"PR={pr:.3f} ({self.help_total}H/{self.steal_total}S)")
    
    def log_episode(self, episode_reward: float):
        """Log episode completion."""
        self.episode += 1
        self.rewards.append(episode_reward)
        
        avg_reward = np.mean(self.rewards[-100:])
        print(f"\n=== Episode {self.episode} ===")
        print(f"  Reward: {episode_reward:.2f} (avg100: {avg_reward:.2f})")
        total = self.help_total + self.steal_total
        if total > 0:
            print(f"  Prosocial: {self.help_total / total:.3f}")
        print()


def create_visualization(env, agent, 
                        enabled: bool = True,
                        update_freq: int = 10,
                        save_dir: Optional[str] = None) -> Optional[TrainingVisualizer]:
    """
    Factory function to create visualization.
    
    Args:
        env: Environment instance
        agent: Agent instance
        enabled: Whether to enable visualization
        update_freq: Update frequency
        save_dir: Directory for snapshots
    
    Returns:
        TrainingVisualizer if enabled, else None
    """
    if not enabled:
        return None
    
    try:
        viz = TrainingVisualizer(
            grid_size=getattr(env, 'grid_size', 8),
            iae_dim=getattr(agent, 'iae_dim', 32),
            update_freq=update_freq,
            save_dir=save_dir
        )
        return viz
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
        return None


if __name__ == '__main__':
    # Demo/test visualization
    print("Testing TrainingVisualizer...")
    
    viz = TrainingVisualizer(grid_size=8, iae_dim=32, update_freq=1)
    
    # Simulate training
    agent_pos = np.array([4.0, 4.0])
    victim_pos = np.array([5.5, 5.5])
    
    for step in range(200):
        # Random movement
        agent_pos += np.random.randn(2) * 0.3
        agent_pos = np.clip(agent_pos, 0.5, 7.5)
        
        # Victim follows
        direction = agent_pos - victim_pos
        victim_pos += direction * 0.1
        victim_pos = np.clip(victim_pos, 0.5, 7.5)
        
        # Random IAE
        iae = np.random.randn(32) * (step / 100)
        
        # Random action
        action = np.random.randint(6)
        action_probs = np.random.dirichlet([1]*6)
        
        dist = np.linalg.norm(agent_pos - victim_pos)
        info = {
            'can_interact': dist < 1.5,
            'pr_flag': np.random.choice([None, 'help', 'harm'], p=[0.7, 0.15, 0.15]),
            'interaction_radius': 1.5
        }
        
        episode_done = (step + 1) % 50 == 0
        
        viz.update(
            agent_pos=agent_pos,
            victim_pos=victim_pos,
            action=action,
            action_probs=action_probs,
            reward=np.random.randn(),
            info=info,
            iae=iae,
            ar=np.random.rand() * 2,
            episode_done=episode_done,
            episode_reward=np.random.randn() * 10 if episode_done else 0
        )
        
        time.sleep(0.05)
    
    print("Test complete. Close window to exit.")
    plt.show(block=True)
    viz.close()
