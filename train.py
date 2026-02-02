"""
Training script for ESAI-v3 with proper scheduling and safeguards.

Key fixes:
1. Lambda_reg warmup schedule (critical for stability)
2. Entropy coefficient annealing
3. IAE and AR norm clipping
4. Proper forecaster training loop with gradient isolation
5. Enhanced diagnostics
6. Fixed gradient graph issues (detach persistent state)
7. Conditional Hebbian integration (matches model architecture)
8. STOCHASTIC evaluation (matches paper protocol)
9. ACTION MASKING for invalid help/steal when out of range (paper-faithful)
"""

import os
import sys
import time
import json
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# GPU Optimization imports
from torch.amp import autocast

# Local imports
from esaiv3.model import ESAIv3Agent
from esaiv3.env_wrappers import make_env
from esaiv3.utils import set_seed, compute_gae, get_device
from esaiv3.logging_utils import ensure_dir, save_json

# Optional visualization (not required for training)
try:
    from esaiv3.visualization import TrainingVisualizer, create_visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    TrainingVisualizer = None
    create_visualization = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Action indices for moral actions
ACTION_HELP = 4
ACTION_STEAL = 5


# =============================================================================
# GPU OPTIMIZATION (A100)
# =============================================================================

def setup_gpu_optimizations(device):
    """
    Configure GPU optimizations for maximum A100 performance.
    These settings can provide 2-5x speedup on A100 GPUs.
    """
    if device.type == 'cuda':
        # Enable TF32 for Ampere GPUs (A100) - 8x faster matmul with minimal precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuning - finds optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Set memory allocator for better memory reuse
        if hasattr(torch.cuda, 'memory'):
            # Use expandable segments for better memory management
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        
        print(f"[GPU] CUDA optimizations enabled:")
        print(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return True
    return False


# =============================================================================
# SCHEDULING FUNCTIONS
# =============================================================================

def linear_schedule(t: int, start_val: float, end_val: float, 
                    total_steps: int, warmup_steps: int = 0) -> float:
    """Linear interpolation with optional warmup."""
    if t < warmup_steps:
        return start_val
    
    progress = min(1.0, (t - warmup_steps) / max(1, total_steps - warmup_steps))
    return start_val + (end_val - start_val) * progress


def exponential_decay(t: int, start_val: float, end_val: float, 
                      decay_steps: int) -> float:
    """Exponential decay from start_val to end_val."""
    if start_val <= 0 or end_val <= 0:
        return linear_schedule(t, start_val, end_val, decay_steps)
    decay_rate = (end_val / start_val) ** (1.0 / max(1, decay_steps))
    return max(end_val, start_val * (decay_rate ** t))


class ScheduleManager:
    """Centralized schedule management for all annealed hyperparameters."""
    
    def __init__(self, cfg: Dict, total_steps: int):
        self.cfg = cfg
        self.total_steps = total_steps
        
        # Lambda scheduling
        self.lambda_reg_final = cfg.get('lambda_reg', 0.05)
        self.lambda_warmup_steps = cfg.get('lambda_warmup_steps', 50000)
        self.lambda_ramp_steps = cfg.get('lambda_ramp_steps', 100000)
        
        # Entropy scheduling
        self.entropy_init = cfg.get('entropy_coef_init', cfg.get('entropy_coef', 0.02))
        self.entropy_final = cfg.get('entropy_coef_final', 0.005)
        self.entropy_decay_steps = cfg.get('entropy_decay_steps', 200000)
        
        # Temperature scheduling
        self.tau_init = cfg.get('tau_init', 1.0)
        self.tau_min = cfg.get('tau_min', 0.01)
        self.tau_decay_steps = cfg.get('tau_decay_steps', 300000)
        
        # Clipping bounds
        self.iae_max_norm = cfg.get('iae_max_norm', 10.0)
        self.ar_max = cfg.get('ar_max', 50.0)
    
    def state_dict(self) -> Dict:
        """Return state dict for checkpoint saving."""
        return {
            'lambda_reg_final': self.lambda_reg_final,
            'lambda_warmup_steps': self.lambda_warmup_steps,
            'lambda_ramp_steps': self.lambda_ramp_steps,
            'entropy_init': self.entropy_init,
            'entropy_final': self.entropy_final,
            'entropy_decay_steps': self.entropy_decay_steps,
            'tau_init': self.tau_init,
            'tau_min': self.tau_min,
            'tau_decay_steps': self.tau_decay_steps,
            'iae_max_norm': self.iae_max_norm,
            'ar_max': self.ar_max,
            'total_steps': self.total_steps,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_lambda_reg(self, step: int) -> float:
        return linear_schedule(
            step, 0.0, self.lambda_reg_final,
            self.lambda_warmup_steps + self.lambda_ramp_steps,
            self.lambda_warmup_steps
        )
    
    def get_entropy_coef(self, step: int) -> float:
        return exponential_decay(step, self.entropy_init, self.entropy_final, self.entropy_decay_steps)
    
    def get_temperature(self, step: int) -> float:
        return exponential_decay(step, self.tau_init, self.tau_min, self.tau_decay_steps)
    
    def clip_iae(self, E: torch.Tensor) -> torch.Tensor:
        norm = E.norm()
        if norm > self.iae_max_norm:
            return E * (self.iae_max_norm / norm)
        return E
    
    def clip_ar(self, ar: float) -> float:
        return min(ar, self.ar_max)
    
    def get_all(self, step: int) -> Dict[str, float]:
        return {
            'lambda_reg': self.get_lambda_reg(step),
            'entropy_coef': self.get_entropy_coef(step),
            'temperature': self.get_temperature(step),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(agent, optimizer, scheduler, global_step, episode, 
                    episode_rewards, metrics_history, cfg, log_dir, tag='',
                    forecaster_optimizer=None):
    """Save training checkpoint with all state for proper resumption."""
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'forecaster_optimizer_state_dict': forecaster_optimizer.state_dict() if forecaster_optimizer else None,
        'global_step': global_step,
        'episode': episode,
        'episode_rewards': episode_rewards[-1000:],
        'metrics_history': {k: v[-1000:] for k, v in metrics_history.items()},
        'config': cfg,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f'checkpoint_{tag}.pt' if tag else f'checkpoint_{global_step}.pt'
    path = os.path.join(log_dir, filename)
    torch.save(checkpoint, path)
    
    # Also save metrics as JSON for easy plotting
    metrics_json = {
        'global_step': global_step,
        'episode': episode,
        'reward': list(episode_rewards[-1000:]),  # Last 1000 episode rewards
        # Evaluation metrics (key for paper figures)
        'eval_mean_reward': list(metrics_history.get('eval_mean_reward', [])),
        'eval_prosocial_ratio': list(metrics_history.get('eval_prosocial_ratio', [])),
        'eval_alignment_regret': list(metrics_history.get('eval_alignment_regret', [])),
        'eval_iae_norm': list(metrics_history.get('eval_iae_norm', [])),
        'eval_total_help': list(metrics_history.get('eval_total_help', [])),
        'eval_total_steal': list(metrics_history.get('eval_total_steal', [])),
        # Training metrics
        'policy_loss': list(metrics_history.get('policy_loss', []))[-500:],
        'value_loss': list(metrics_history.get('value_loss', []))[-500:],
        'entropy': list(metrics_history.get('entropy', []))[-500:],
        'lambda_reg': list(metrics_history.get('lambda_reg', []))[-500:],
        'config': cfg,
    }
    metrics_path = os.path.join(log_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return path


def apply_action_mask(logits: torch.Tensor, can_interact: bool) -> torch.Tensor:
    """
    Mask invalid moral actions when agent cannot interact.
    
    Paper-faithful: The paper does not allow help/steal actions
    when agent is outside interaction range.
    
    Args:
        logits: Raw policy logits [batch, num_actions]
        can_interact: Whether agent is within interaction range
    
    Returns:
        Masked logits (invalid actions set to -inf)
    """
    if not can_interact:
        masked_logits = logits.clone()
        # Use -1e4 instead of -1e9 to avoid FP16 overflow (max ~65504)
        masked_logits[..., ACTION_HELP] = -1e4
        masked_logits[..., ACTION_STEAL] = -1e4
        return masked_logits
    return logits


# =============================================================================
# ROLLOUT COLLECTION (with action masking)
# =============================================================================

def collect_rollout(agent, env, scheduler, global_step, cfg, device, 
                    max_steps=2048, debug=False, visualizer=None):
    """
    Collect rollout with proper IAE clipping, AR computation, and action masking.
    FIXED: Now collects MULTIPLE episodes until max_steps is reached.
    
    Args:
        visualizer: Optional TrainingVisualizer for real-time display
    """
    obs, info = env.reset(seed=np.random.randint(0, 1000000))
    agent.reset_iae()
    
    rollout_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'values': [],
        'log_probs': [],
        'dones': [],
        'ar_penalties': [],
        'next_observations': [],
        'current_E': [],  # FIX BUG-016: Store E at time of each observation
        'next_E': [],
        'pr_flags': [],
        'can_interact': [],
        'action_masks': [],  # Store masks for PPO update
    }
    
    terminated = truncated = False
    episode_reward = 0
    total_reward = 0  # Accumulate across all episodes
    steps = 0
    episode_count = 0
    
    current_temp = scheduler.get_temperature(global_step)
    
    # CRITICAL FIX: Continue collecting steps until max_steps, even across multiple episodes
    while steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # ISSUE #3 FIX: Get can_interact ONLY from environment info, not hardcoded obs index
        can_interact = info.get('can_interact', True)
        
        with torch.no_grad():
            agent.temperature = current_temp
            
            # FIX BUG-009: Single consistent action sampling path
            # First get logits and apply mask BEFORE sampling
            if agent.use_attention:
                obs_attended = agent.apply_attention(obs_tensor)
            else:
                obs_attended = obs_tensor
            
            # Check for NaN in obs_attended
            if torch.isnan(obs_attended).any():
                obs_attended = obs_tensor  # Fallback to unattended
            
            policy_input = torch.cat([obs_attended, agent.E.unsqueeze(0)], dim=-1)
            logits = agent.policy_net(policy_input)
            
            # Check for NaN in logits - reset if found
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
            
            # Apply action mask BEFORE sampling
            masked_logits = apply_action_mask(logits, can_interact)
            
            # Sample from masked distribution (single path - no resampling)
            probs = F.softmax(masked_logits, dim=-1)
            
            # Check for NaN in probs - use uniform if found
            if torch.isnan(probs).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            
            # Now compute AR for the SAME action we sampled
            # FIX BUG-009: AR must be computed for the action we actually take
            AR_t = torch.tensor(0.0, device=obs_tensor.device)
            if agent.use_alignment_regret and hasattr(agent, 'forecast_net'):
                E_preds = []
                obs_flat = obs_tensor.flatten()
                
                hebbian_readout = None
                if agent.use_hebbian and agent.hebbian is not None:
                    hebbian_readout = agent.hebbian.read(agent.E)
                
                # FIX: Dynamically detect forecaster input dimension
                first_layer = None
                for module in agent.forecast_net_target.children():
                    if isinstance(module, nn.Linear):
                        first_layer = module
                        break
                expected_dim = first_layer.in_features if first_layer else None
                
                for a in range(agent.action_dim):
                    action_onehot = torch.zeros(agent.action_dim, device=obs_tensor.device)
                    action_onehot[a] = 1.0
                    
                    input_parts = [obs_flat, action_onehot, agent.E]
                    if hebbian_readout is not None:
                        input_parts.append(hebbian_readout.squeeze())
                    forecast_input = torch.cat(input_parts)
                    
                    # FIX: Pad input if model expects extra dimension (legacy reward placeholder)
                    if expected_dim is not None and forecast_input.shape[0] < expected_dim:
                        padding = torch.zeros(expected_dim - forecast_input.shape[0], device=obs_tensor.device)
                        forecast_input = torch.cat([forecast_input, padding])
                    
                    E_pred = agent.forecast_net_target(forecast_input)
                    E_preds.append(E_pred)
                
                E_preds = torch.stack(E_preds)
                agent.alignment_loss.temperature = agent.temperature
                AR_t = agent.alignment_loss(E_current=agent.E, E_preds=E_preds, E_neighbors=None)
            
            ar_penalty = AR_t.item() if isinstance(AR_t, torch.Tensor) else AR_t
            ar_penalty = scheduler.clip_ar(ar_penalty)
            
            # Get value estimate
            value = agent.compute_value(obs_tensor)
            
            # Store extra info
            extra = {
                'E': agent.E.detach().cpu().numpy(),
                'AR_t': ar_penalty,
                'E_norm': agent.E.norm().item()
            }
        
        action_np = action.item()
        step_result = env.step(action_np)
        
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
        else:
            next_obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # Update IAE using HARM signal (not reward) - IAE grows when agent causes harm
        action_tensor = torch.tensor([action_np], device=device)
        harm_t = info.get('harm_t', 0.0)  # harm_t > 0 when stealing
        harm_tensor = torch.tensor([harm_t], dtype=torch.float32, device=device)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # FIX BUG-020: Store current E BEFORE IAE update (the E used for this decision)
        current_E_for_step = agent.E.clone().detach().cpu()
        
        agent.update_iae(next_obs_tensor, action_tensor, harm_tensor)
        agent.E.data = scheduler.clip_iae(agent.E.data)
        
        # Create action mask tensor for this step
        action_mask = torch.ones(agent.action_dim, dtype=torch.float32)
        if not can_interact:
            action_mask[ACTION_HELP] = 0.0
            action_mask[ACTION_STEAL] = 0.0
        
        # Update visualization if enabled
        if visualizer is not None:
            # Get action probabilities for visualization
            with torch.no_grad():
                probs = F.softmax(masked_logits, dim=-1).cpu().numpy().flatten()
            
            # Add interaction_radius to info for visualization
            info['interaction_radius'] = getattr(env, 'interaction_radius', 1.5)
            
            visualizer.update(
                agent_pos=getattr(env, 'agent_pos', np.array([0, 0])),
                victim_pos=getattr(env, 'victim_pos', np.array([1, 1])),
                action=action_np,
                action_probs=probs,
                reward=reward,
                info=info,
                iae=agent.E.detach().cpu().numpy(),
                ar=ar_penalty,
                episode_done=(terminated or truncated),
                episode_reward=episode_reward if (terminated or truncated) else 0
            )
        
        # Store transition
        rollout_data['observations'].append(obs_tensor.squeeze(0).detach().cpu())
        rollout_data['actions'].append(action_np)
        rollout_data['rewards'].append(reward)
        rollout_data['values'].append(value.item())
        rollout_data['log_probs'].append(log_prob)
        rollout_data['dones'].append(terminated or truncated)
        rollout_data['ar_penalties'].append(ar_penalty)
        rollout_data['next_observations'].append(torch.tensor(next_obs, dtype=torch.float32).cpu())
        rollout_data['current_E'].append(current_E_for_step)  # E at decision time
        rollout_data['next_E'].append(agent.E.clone().detach().cpu())  # E after update
        rollout_data['pr_flags'].append(info.get('pr_flag', None))
        rollout_data['can_interact'].append(can_interact)
        rollout_data['action_masks'].append(action_mask)
        
        obs = next_obs
        episode_reward += reward
        steps += 1
        
        # CRITICAL FIX: Reset environment when episode ends, continue collecting
        if terminated or truncated:
            total_reward += episode_reward
            episode_count += 1
            episode_reward = 0
            
            # Reset for next episode
            obs, info = env.reset(seed=np.random.randint(0, 1000000))
            agent.reset_iae()
            terminated = truncated = False
    
    # Add final episode reward if not yet counted
    if episode_reward > 0:
        total_reward += episode_reward
        episode_count += 1
    
    # Return average episode reward (or total if only one episode)
    avg_episode_reward = total_reward / max(episode_count, 1)
    return rollout_data, avg_episode_reward, steps


# =============================================================================
# REWARD TRANSFORMATION
# =============================================================================

def compute_returns_and_advantages(rollout_data, scheduler, global_step, cfg):
    """Compute returns and advantages with SCHEDULED lambda_reg."""
    rewards = rollout_data['rewards']
    values = rollout_data['values']
    dones = rollout_data['dones']
    ar_penalties = rollout_data['ar_penalties']
    
    lambda_reg = scheduler.get_lambda_reg(global_step)
    transformed_rewards = [r - lambda_reg * ar for r, ar in zip(rewards, ar_penalties)]
    
    returns, advantages = compute_gae(
        transformed_rewards, values, dones,
        gamma=cfg.get('gamma', 0.99),
        lambda_=cfg.get('gae_lambda', 0.95)
    )
    
    return returns, advantages, transformed_rewards, lambda_reg


# =============================================================================
# FORECASTER TRAINING
# =============================================================================

def train_forecaster(agent, forecaster_optimizer, rollout_data, cfg, device):
    """
    Train IAE forecaster with COUNTERFACTUAL targets.
    
    FIX BUG-011: The forecaster must learn to predict E^(a) for ALL actions,
    not just the action taken. We compute counterfactual targets using the
    known IAE dynamics: E_{t+1} = γ_E * E_t + g_φ(obs, action, harm_t)
    
    FIX BUG-015: Use per-sample E values from rollout, not end-of-rollout E.
    
    For each sample, we generate targets for ALL 6 actions:
    - Actions 0-3 (movement): harm_t = 0
    - Action 4 (HELP): harm_t = 0  
    - Action 5 (STEAL): harm_t = victim_distress (from cfg)
    """
    if not agent.use_alignment_regret or not hasattr(agent, 'forecast_net'):
        return 0.0
    
    n_samples = len(rollout_data['observations'])
    if n_samples < 32:
        return 0.0
    
    obs_batch = torch.stack(rollout_data['observations']).to(device).detach()
    # FIX BUG-015: Use per-sample E values from rollout
    E_batch = torch.stack(rollout_data['current_E']).to(device).detach()
    
    # Get victim_distress for counterfactual STEAL targets
    victim_distress = cfg.get('victim_distress', 3.0)
    
    batch_size = min(64, n_samples)
    n_epochs = 3
    total_loss = 0.0
    n_updates = 0
    
    obs_dim = obs_batch.shape[1]
    expected_base_dim = obs_dim + agent.action_dim + agent.iae_dim
    
    first_layer = None
    if hasattr(agent.forecast_net, 'children'):
        for module in agent.forecast_net.children():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
    
    forecaster_input_dim = first_layer.in_features if first_layer is not None else expected_base_dim
    use_hebbian_in_forecaster = (forecaster_input_dim > expected_base_dim) and agent.use_hebbian
    
    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples, device=device)
        
        for start in range(0, n_samples - batch_size + 1, batch_size):
            idx = indices[start:start + batch_size]
            obs = obs_batch[idx]
            # FIX BUG-015: Get per-sample E for this batch
            E_samples = E_batch[idx]
            batch_len = len(idx)
            
            # Compute Hebbian readouts for each sample if needed
            if use_hebbian_in_forecaster:
                with torch.no_grad():
                    # Hebbian read for each sample's E
                    heb_reads = []
                    for i in range(batch_len):
                        heb_read_i = agent.hebbian.read(E_samples[i]).view(-1)
                        heb_reads.append(heb_read_i)
                    heb_batch = torch.stack(heb_reads)
            
            # FIX BUG-011: Train on ALL actions, not just the action taken
            # This teaches the forecaster proper counterfactuals
            all_losses = []
            
            for action_id in range(agent.action_dim):
                # Determine harm_t for this action (counterfactual)
                if action_id == 5:  # STEAL
                    harm_t = victim_distress
                else:  # Movement or HELP
                    harm_t = 0.0
                
                harm_tensor = torch.full((batch_len, 1), harm_t, device=device)
                
                # Compute counterfactual target E using IAE dynamics
                # E_{t+1} = γ_E * E_t + g_φ(obs, action, harm_t)
                action_oh = torch.zeros(batch_len, agent.action_dim, device=device)
                action_oh[:, action_id] = 1.0
                
                with torch.no_grad():
                    dynamics_input = torch.cat([obs, action_oh, harm_tensor], dim=-1)
                    delta_E = agent.iae_dynamics(dynamics_input)
                    # FIX BUG-015: Use per-sample E, not single E for all
                    target_E = agent.gamma_E * E_samples + delta_E
                
                # Forecaster input with per-sample E
                forecast_input = torch.cat([obs, action_oh, E_samples], dim=-1)
                
                if use_hebbian_in_forecaster:
                    if hasattr(agent, 'hebbian_gate'):
                        gate = agent.hebbian_gate(E_samples)
                    else:
                        gate = torch.ones(batch_len, 1, device=device)
                    
                    gated_heb = gate * heb_batch
                    forecast_input = torch.cat([forecast_input, gated_heb], dim=-1)
                
                # FIX BUG-021: Pad input if model expects extra dimension (legacy reward placeholder)
                # instead of skipping the sample
                if forecast_input.shape[1] < forecaster_input_dim:
                    padding_size = forecaster_input_dim - forecast_input.shape[1]
                    padding = torch.zeros(batch_len, padding_size, device=device)
                    forecast_input = torch.cat([forecast_input, padding], dim=-1)
                elif forecast_input.shape[1] > forecaster_input_dim:
                    # Trim if too large (shouldn't happen but safety check)
                    forecast_input = forecast_input[:, :forecaster_input_dim]
                
                pred_E = agent.forecast_net(forecast_input)
                loss = F.smooth_l1_loss(pred_E, target_E)
                all_losses.append(loss)
            
            if not all_losses:
                continue
            
            # Average loss over all actions
            total_batch_loss = torch.stack(all_losses).mean()
            
            forecaster_optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.forecast_net.parameters(), 1.0)
            
            if getattr(agent, 'hebbian_gate', None) is not None:
                torch.nn.utils.clip_grad_norm_(agent.hebbian_gate.parameters(), 1.0)
            
            forecaster_optimizer.step()
            
            total_loss += total_batch_loss.item()
            n_updates += 1
    
    if hasattr(agent, 'update_target_forecaster'):
        agent.update_target_forecaster(tau=cfg.get('ema_tau', 0.995))
    
    return total_loss / max(n_updates, 1)


# =============================================================================
# PPO UPDATE (with action masking)
# =============================================================================

def ppo_update(agent, optimizer, forecaster_optimizer, rollout_data, returns, advantages, 
               scheduler, global_step, cfg, device, scaler=None):
    """PPO update with action masking for invalid moral actions.
    
    FIX BUG-010: Added separate forecaster_optimizer parameter.
    
    Args:
        optimizer: Optimizer for policy/value networks
        forecaster_optimizer: Separate optimizer for forecaster network
        scaler: Optional GradScaler for mixed precision training (A100 optimization)
    """
    num_epochs = cfg.get('num_epochs', 10)
    batch_size = cfg.get('batch_size', 64)
    clip_epsilon = cfg.get('clip_epsilon', 0.2)
    value_coef = cfg.get('value_coef', 0.5)
    max_grad_norm = cfg.get('max_grad_norm', 1.0)
    use_amp = scaler is not None and device.type == 'cuda'
    
    entropy_coef = scheduler.get_entropy_coef(global_step)
    
    obs_batch = torch.stack(rollout_data['observations']).to(device, non_blocking=True).detach()
    actions_batch = torch.tensor(rollout_data['actions'], dtype=torch.long, device=device)
    old_log_probs = torch.tensor(rollout_data['log_probs'], dtype=torch.float32, device=device)
    returns_batch = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_batch = torch.tensor(advantages, dtype=torch.float32, device=device)
    action_masks_batch = torch.stack(rollout_data['action_masks']).to(device, non_blocking=True)
    # FIX BUG-016: Use per-step E values, not end-of-rollout E
    E_batch = torch.stack(rollout_data['current_E']).to(device, non_blocking=True).detach()
    
    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
    
    n_samples = len(obs_batch)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0
    grad_norm = 0.0
    
    for epoch in range(num_epochs):
        indices = torch.randperm(n_samples, device=device)
        
        for start in range(0, n_samples - batch_size + 1, batch_size):
            idx = indices[start:start + batch_size]
            
            obs = obs_batch[idx]
            actions = actions_batch[idx]
            old_lp = old_log_probs[idx]
            ret = returns_batch[idx]
            adv = advantages_batch[idx]
            action_masks = action_masks_batch[idx]
            # FIX BUG-016: Use per-sample E values from rollout collection
            E_samples = E_batch[idx]
            
            # Mixed precision forward pass
            with autocast(device_type='cuda', enabled=use_amp):
                # FIX BUG-018 & BUG-023: Apply attention with correct per-sample E (batched)
                if agent.use_attention:
                    # Batch attention: AttentionGating can handle batched inputs
                    # E_samples: (batch, iae_dim), obs: (batch, obs_dim)
                    # AttentionGating.forward expects (E, obs) both potentially batched
                    obs_att = agent.attention(E_samples, obs)
                else:
                    obs_att = obs
                
                # FIX BUG-016: Use per-sample E, not single E for all
                policy_input = torch.cat([obs_att, E_samples], dim=-1)
                
                logits = agent.policy_net(policy_input)
                
                # ISSUE #2 FIX: Use masked_fill for numerically stable masking
                # Use -1e4 instead of -1e9 to avoid FP16 overflow (max ~65504)
                masked_logits = logits.masked_fill(action_masks == 0, -1e4)
                
                log_probs = F.log_softmax(masked_logits, dim=-1)
                new_lp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Entropy over valid actions only (avoid NaN from -inf * 0)
                probs = F.softmax(masked_logits, dim=-1)
                # Clamp log_probs to avoid -inf * 0 = nan
                safe_log_probs = torch.clamp(log_probs, min=-20.0)
                entropy_raw = -(probs * safe_log_probs * action_masks).sum(dim=-1).mean()
                # Guard against NaN entropy - use detached fallback to avoid graph issues
                if torch.isfinite(entropy_raw):
                    entropy = entropy_raw
                else:
                    entropy = torch.zeros(1, device=entropy_raw.device, requires_grad=False).squeeze()
                
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                values = agent.value_net(policy_input).squeeze(-1)
                value_loss = F.mse_loss(values, ret)
                
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                
                if agent.use_hebbian and hasattr(agent, 'hebbian'):
                    with torch.no_grad():
                        heb_norm = agent.hebbian.get_norm()
                        if isinstance(heb_norm, torch.Tensor):
                            heb_norm = heb_norm.item()
                    loss = loss + cfg.get('lambda_H', 1e-3) * heb_norm
            
            # Backward pass with optional mixed precision scaling
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1
    
    # FIX BUG-010: Use separate forecaster_optimizer
    forecast_loss = train_forecaster(agent, forecaster_optimizer, rollout_data, cfg, device)
    
    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),
        'entropy_coef': entropy_coef,
        'forecast_loss': forecast_loss,
        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


# =============================================================================
# EVALUATION (Stochastic + Action Masking)
# =============================================================================

def evaluate(agent, env, scheduler, global_step, num_episodes, device):
    """
    Run evaluation with STOCHASTIC policy and action masking.
    Matches paper protocol exactly.
    """
    results = {
        'rewards': [],
        'prosocial_ratios': [],
        'alignment_regrets': [],
        'iae_norms': [],
        'moral_decisions': [],
        'help_counts': [],
        'steal_counts': [],
    }
    
    current_temp = scheduler.get_temperature(global_step)
    eval_temp = max(0.05, current_temp)
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        agent.reset_iae()
        
        ep_reward = 0
        ep_help = 0
        ep_steal = 0
        ep_ar = []
        ep_iae = []
        
        terminated = truncated = False
        
        while not (terminated or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            # ISSUE #3 FIX: Get can_interact ONLY from environment info
            can_interact = info.get('can_interact', True)
            
            with torch.no_grad():
                agent.temperature = eval_temp
                
                # Get logits and apply mask
                if agent.use_attention:
                    obs_attended = agent.apply_attention(obs_t)
                else:
                    obs_attended = obs_t
                
                policy_input = torch.cat([obs_attended, agent.E.unsqueeze(0)], dim=-1)
                logits = agent.policy_net(policy_input)
                
                # Apply action mask
                masked_logits = apply_action_mask(logits, can_interact)
                
                # Sample from masked distribution (STOCHASTIC)
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                _, extra = agent.act(obs_t, deterministic=False)
            
            action_np = action.item()
            step_result = env.step(action_np)
            
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
            else:
                next_obs, reward, done, info = step_result
                terminated = done
                truncated = False
            
            ep_reward += reward
            
            pr_flag = info.get('pr_flag')
            if pr_flag == 'help':
                ep_help += 1
            elif pr_flag == 'harm':
                ep_steal += 1
            
            ar = extra.get('AR_t', 0.0)
            if isinstance(ar, torch.Tensor):
                ar = ar.item()
            ep_ar.append(ar)
            ep_iae.append(agent.E.norm().item())
            
            # Update IAE using HARM signal (not reward) - IAE grows when agent causes harm
            action_tensor = torch.tensor([action_np], device=device)
            harm_t = info.get('harm_t', 0.0)  # harm_t > 0 when stealing
            harm_tensor = torch.tensor([harm_t], dtype=torch.float32, device=device)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            agent.update_iae(next_obs_t, action_tensor, harm_tensor)
            agent.E.data = scheduler.clip_iae(agent.E.data)
            
            obs = next_obs
        
        results['rewards'].append(ep_reward)
        results['help_counts'].append(ep_help)
        results['steal_counts'].append(ep_steal)
        results['moral_decisions'].append(ep_help + ep_steal)
        
        # Per-episode PR: Track ALL episodes, including zero-engagement ones
        # Zero-engagement episodes get PR=0.0 (not NaN) to avoid inflation
        # This is more conservative - treats no moral action as "not prosocial"
        if ep_help + ep_steal > 0:
            results['prosocial_ratios'].append(ep_help / (ep_help + ep_steal))
        else:
            # Zero-engagement: Record as 0.0 (conservative) not NaN (exclusion)
            # Rationale: Agent had opportunity but didn't engage prosocially
            results['prosocial_ratios'].append(0.0)
        
        if ep_ar:
            results['alignment_regrets'].append(np.mean(ep_ar))
        if ep_iae:
            results['iae_norms'].append(np.mean(ep_iae))
    
    total_help = sum(results['help_counts'])
    total_steal = sum(results['steal_counts'])
    total_moral = total_help + total_steal
    
    # Count episodes with moral engagement
    engaged_episodes = sum(1 for md in results['moral_decisions'] if md > 0)
    
    # Per-episode PR mean (includes zeros for non-engaged episodes)
    per_episode_pr_mean = np.mean(results['prosocial_ratios'])
    
    return {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        # Aggregate PR: total_help / total_moral (undefined -> 0 if no moral decisions)
        'prosocial_ratio': total_help / max(1, total_moral),
        # Per-episode mean (includes 0s for non-engaged episodes - more conservative)
        'prosocial_ratio_per_ep': per_episode_pr_mean,
        'alignment_regret': np.mean(results['alignment_regrets']) if results['alignment_regrets'] else 0.0,
        'iae_norm': np.mean(results['iae_norms']) if results['iae_norms'] else 0.0,
        'total_moral_decisions': total_moral,
        'total_help': total_help,
        'total_steal': total_steal,
        'mean_moral_per_ep': total_moral / max(1, num_episodes),
        'engaged_episodes': engaged_episodes,
        'engagement_rate': engaged_episodes / max(1, num_episodes),  # % of episodes with moral decisions
        'zero_engagement_episodes': num_episodes - engaged_episodes,
        'invalid_rate': 0.0,  # Now 0% because we mask invalid actions
    }


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_diagnostics(rollout_data, scheduler, global_step, episode_count, agent):
    """Print training diagnostics with engagement tracking."""
    
    if episode_count % 10 != 0:
        return
    
    sched = scheduler.get_all(global_step)
    
    rewards = rollout_data['rewards']
    ar = rollout_data['ar_penalties']
    pr_flags = rollout_data['pr_flags']
    can_interact = rollout_data.get('can_interact', [])
    
    ep_return = np.sum(rewards)
    r_mean = np.mean(rewards)
    ar_mean = np.mean(ar) if ar else 0
    can_interact_rate = np.mean(can_interact) if can_interact else 0.0
    
    lambda_reg = sched['lambda_reg']
    r_trans = [r - lambda_reg * a for r, a in zip(rewards, ar)]
    r_trans_mean = np.mean(r_trans)
    
    n_help = sum(1 for p in pr_flags if p == 'help')
    n_harm = sum(1 for p in pr_flags if p == 'harm')
    n_moral = n_help + n_harm
    
    actions = rollout_data['actions']
    action_counts = np.bincount(actions, minlength=agent.action_dim)
    
    # With action masking, invalid rate should be ~0
    n_action_4 = action_counts[4] if len(action_counts) > 4 else 0
    n_action_5 = action_counts[5] if len(action_counts) > 5 else 0
    n_invalid = max(0, (n_action_4 - n_help) + (n_action_5 - n_harm))
    invalid_rate = n_invalid / max(1, n_action_4 + n_action_5)
    
    print(f"\n{'='*60}")
    print(f"[DIAG] Step {global_step:,}, Episode {episode_count}")
    print(f"{'='*60}")
    print(f"  Schedule: λ={lambda_reg:.4f}, H={sched['entropy_coef']:.4f}, τ={sched['temperature']:.4f}")
    print(f"  Episode:  Return={ep_return:.2f}, Steps={len(rewards)}")
    print(f"  Rewards:  r_ext={r_mean:.3f}, AR={ar_mean:.3f}, r'={r_trans_mean:.3f}")
    print(f"  CanInteract: {can_interact_rate:.2f} (agent in dilemma region)")
    print(f"  Moral:    {n_moral} decisions ({n_help} help / {n_harm} harm)")
    
    if n_moral > 0:
        pr = n_help / n_moral
        print(f"  PR:       {pr:.3f} (conditional on engagement)")
    else:
        print(f"  PR:       N/A (no moral engagement)")
    
    print(f"  Invalid:  {n_invalid} ({invalid_rate:.1%}) [should be ~0% with masking]")
    print(f"  Actions:  {action_counts.tolist()}")
    print(f"  ||E||:    {agent.E.norm().item():.3f}")
    
    if n_moral == 0 and can_interact_rate > 0.3:
        print(f"  ⚠️  Low moral engagement despite opportunities")
    if invalid_rate > 0.1:
        print(f"  ⚠️  Invalid actions detected - check masking logic")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ESAI-v3')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--env-config', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total-steps', type=int, default=300000)
    parser.add_argument('--eval-interval', type=int, default=25000)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--save-interval', type=int, default=50000)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--debug', action='store_true')
    # A100 GPU optimization flags
    parser.add_argument('--compile', action='store_true', 
                        help='Use torch.compile() for faster training (PyTorch 2.0+)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (use 512-1024 for A100)')
    parser.add_argument('--rollout-length', type=int, default=None,
                        help='Override rollout length (use 4096-8192 for A100)')
    # Visualization flags
    parser.add_argument('--visualize', action='store_true',
                        help='Enable real-time training visualization')
    parser.add_argument('--viz-freq', type=int, default=10,
                        help='Visualization update frequency (steps)')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualization snapshots')
    args = parser.parse_args()
    
    # Load configs
    model_cfg = load_yaml(args.config)
    env_cfg = load_yaml(args.env_config)
    cfg = {**model_cfg, **env_cfg}
    
    # Device
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print(f"[train] Device: {device}")
    
    # Setup GPU optimizations for A100
    use_cuda = setup_gpu_optimizations(device)
    
    # Mixed precision scaler (only for CUDA)
    # Use torch.cuda.amp.GradScaler for maximum compatibility
    if use_cuda:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        scaler = CudaGradScaler()
        print(f"[GPU] Mixed precision (AMP): Enabled")
    else:
        scaler = None
    
    # Override config with command-line args for A100 optimization
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
        print(f"[GPU] Batch size override: {args.batch_size}")
    if args.rollout_length is not None:
        cfg['rollout_length'] = args.rollout_length
        print(f"[GPU] Rollout length override: {args.rollout_length}")
    
    # Seed
    set_seed(args.seed)
    
    # Logging
    env_name = cfg.get('env', 'moral_temptation')
    log_dir = f"results/{env_name}/{args.exp_name}/seed_{args.seed}"
    ensure_dir(log_dir)
    save_json(cfg, os.path.join(log_dir, 'config.json'))
    
    # Environment
    env = make_env(env_name, cfg)
    print(f"[train] Environment: {env_name}")
    print(f"[train] Obs: {env.observation_space.shape}, Act: {env.action_space.n}")
    
    # Agent
    agent = ESAIv3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        iae_dim=cfg.get('iae_dim', 32),
        hidden_dim=cfg.get('hidden_dim', 128),
        gamma_E=cfg.get('gamma_E', 0.9),
        alpha_diffusion=cfg.get('alpha_diffusion', 0.05),
        num_agents=cfg.get('num_agents', 1),
        use_attention=cfg.get('attention', True),
        use_hebbian=cfg.get('heb_read_to_forecast', True),
        use_diffusion=cfg.get('diffusion', False),
        use_alignment_regret=cfg.get('use_alignment_regret', True),
        lambda_bias=cfg.get('lambda_bias', 0.0),
    ).to(device)
    
    # FIX: Set kappa from config (default in model is 0.5, config may specify 0.05)
    if hasattr(agent, 'alignment_loss') and agent.alignment_loss is not None:
        agent.alignment_loss.kappa = cfg.get('kappa', 0.05)
        print(f"  Kappa (neighbor harm): {agent.alignment_loss.kappa}")
    
    # Compile model for faster execution (PyTorch 2.0+)
    if args.compile and use_cuda:
        try:
            agent.policy_net = torch.compile(agent.policy_net, mode='reduce-overhead')
            agent.value_net = torch.compile(agent.value_net, mode='reduce-overhead')
            if hasattr(agent, 'forecast_net') and agent.forecast_net is not None:
                agent.forecast_net = torch.compile(agent.forecast_net, mode='reduce-overhead')
            print(f"[GPU] torch.compile(): Enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"[GPU] torch.compile() failed: {e}")
    
    print(f"\n[train] Agent:")
    print(f"  IAE dim: {agent.iae_dim}")
    print(f"  Hidden: {agent.hidden_dim}")
    print(f"  Attention: {agent.use_attention}")
    print(f"  Hebbian: {agent.use_hebbian}")
    print(f"  Alignment regret: {agent.use_alignment_regret}")
    
    if hasattr(agent, 'forecast_net'):
        first_layer = None
        for module in agent.forecast_net.children():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer:
            print(f"  Forecaster input dim: {first_layer.in_features}")
    
    # Initialize visualization if enabled
    visualizer = None
    if args.visualize and VISUALIZATION_AVAILABLE:
        viz_save_dir = os.path.join(log_dir, 'viz') if args.save_viz else None
        visualizer = create_visualization(
            env, agent,
            enabled=True,
            update_freq=args.viz_freq,
            save_dir=viz_save_dir
        )
        if visualizer:
            print(f"\n[train] Visualization: ENABLED (update every {args.viz_freq} steps)")
            if viz_save_dir:
                print(f"  Saving snapshots to: {viz_save_dir}")
        else:
            print(f"\n[train] Visualization: FAILED to initialize")
    elif args.visualize and not VISUALIZATION_AVAILABLE:
        print(f"\n[train] Visualization: NOT AVAILABLE (module not installed)")
    
    # FIX BUG-010: Separate optimizers for policy and forecaster
    # Policy/Value optimizer
    policy_params = []
    forecaster_params = []
    seen_ids = set()
    
    for name, p in agent.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        
        # Separate forecaster params from policy/value params
        if 'forecast_net' in name or 'forecast_net_target' in name:
            forecaster_params.append(p)
        else:
            policy_params.append(p)
    
    optimizer = torch.optim.Adam(policy_params, lr=cfg.get('lr', 3e-4))
    
    # FIX BUG-010: Separate forecaster optimizer with potentially different LR
    forecaster_lr = cfg.get('forecaster_lr', cfg.get('lr', 3e-4))
    forecaster_optimizer = torch.optim.Adam(forecaster_params, lr=forecaster_lr) if forecaster_params else None
    
    # Schedule manager
    scheduler = ScheduleManager(cfg, args.total_steps)
    
    print(f"\n[train] Schedule configuration:")
    print(f"  lambda_reg: 0 → {scheduler.lambda_reg_final} over {scheduler.lambda_warmup_steps + scheduler.lambda_ramp_steps} steps")
    print(f"  entropy:    {scheduler.entropy_init} → {scheduler.entropy_final} over {scheduler.entropy_decay_steps} steps")
    print(f"  tau:        {scheduler.tau_init} → {scheduler.tau_min} over {scheduler.tau_decay_steps} steps")
    print(f"\n[train] Features:")
    print(f"  Evaluation: STOCHASTIC (paper-faithful)")
    print(f"  Action masking: ENABLED (no invalid help/steal)")
    print(f"  Separate forecaster optimizer: {'ENABLED' if forecaster_optimizer else 'N/A'}")
    
    # Training state
    global_step = 0
    episode_count = 0
    episode_rewards = []
    metrics_history = defaultdict(list)
    best_eval_reward = float('-inf')
    
    pbar = tqdm(total=args.total_steps, desc="Training")
    start_time = time.time()
    
    while global_step < args.total_steps:
        rollout_data, ep_reward, steps = collect_rollout(
            agent, env, scheduler, global_step, cfg, device,
            max_steps=cfg.get('rollout_length', 2048),
            debug=args.debug,
            visualizer=visualizer
        )
        
        global_step += steps
        episode_count += 1
        episode_rewards.append(ep_reward)
        pbar.update(steps)
        
        print_diagnostics(rollout_data, scheduler, global_step, episode_count, agent)
        
        if len(rollout_data['rewards']) < 32:
            continue
        
        returns, advantages, trans_rewards, lambda_reg = compute_returns_and_advantages(
            rollout_data, scheduler, global_step, cfg
        )
        
        # FIX BUG-010: Pass separate forecaster_optimizer
        update_metrics = ppo_update(
            agent, optimizer, forecaster_optimizer, rollout_data, returns, advantages,
            scheduler, global_step, cfg, device, scaler=scaler
        )
        
        sched_vals = scheduler.get_all(global_step)
        pbar.set_postfix({
            'ep': episode_count,
            'r': f'{ep_reward:.2f}',
            'λ': f'{sched_vals["lambda_reg"]:.3f}',
            'H': f'{update_metrics["entropy"]:.2f}',
            '||E||': f'{agent.E.norm().item():.2f}'
        })
        
        metrics_history['reward'].append(ep_reward)
        metrics_history['policy_loss'].append(update_metrics['policy_loss'])
        metrics_history['value_loss'].append(update_metrics['value_loss'])
        metrics_history['entropy'].append(update_metrics['entropy'])
        metrics_history['lambda_reg'].append(lambda_reg)
        metrics_history['forecast_loss'].append(update_metrics['forecast_loss'])
        
        if global_step % args.eval_interval == 0:
            agent.eval()
            eval_metrics = evaluate(agent, env, scheduler, global_step, 
                                    args.eval_episodes, device)
            agent.train()
            
            print(f"\n[eval] Step {global_step:,}")
            print(f"  Reward:     {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"  Prosocial:  {eval_metrics['prosocial_ratio']:.3f}")
            print(f"  Moral:      {eval_metrics['total_moral_decisions']} ({eval_metrics['total_help']} help / {eval_metrics['total_steal']} steal)")
            print(f"  AR:         {eval_metrics['alignment_regret']:.3f}")
            print(f"  ||E||:      {eval_metrics['iae_norm']:.3f}")
            print(f"  λ_reg:      {sched_vals['lambda_reg']:.4f}")
            print(f"  Invalid:    {eval_metrics['invalid_rate']:.1%}")
            
            for k, v in eval_metrics.items():
                metrics_history[f'eval_{k}'].append(v)
            
            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                save_checkpoint(agent, optimizer, scheduler, global_step,
                               episode_count, episode_rewards, metrics_history,
                               cfg, log_dir, tag='best',
                               forecaster_optimizer=forecaster_optimizer)
        
        if global_step % args.save_interval == 0:
            save_checkpoint(agent, optimizer, scheduler, global_step,
                           episode_count, episode_rewards, metrics_history,
                           cfg, log_dir,
                           forecaster_optimizer=forecaster_optimizer)
    
    pbar.close()
    
    # Clean up visualization
    if visualizer is not None:
        if args.save_viz:
            visualizer.save_snapshot('final')
        visualizer.close()
    
    save_checkpoint(agent, optimizer, scheduler, global_step,
                   episode_count, episode_rewards, metrics_history,
                   cfg, log_dir, tag='final',
                   forecaster_optimizer=forecaster_optimizer)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[train] Complete!")
    print(f"  Steps: {global_step:,}")
    print(f"  Episodes: {episode_count}")
    print(f"  Time: {elapsed/3600:.2f} hours")
    print(f"  Best eval reward: {best_eval_reward:.2f}")
    print(f"  Final λ_reg: {scheduler.get_lambda_reg(global_step):.4f}")
    print(f"  Saved to: {log_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()