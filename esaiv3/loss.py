"""
Loss functions for ESAI-v3.
Implements:
- Counterfactual alignment regret (Equations 6-9)
- PPO-Clip with entropy regularization
- Value function loss with optional clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """
    Differentiable alignment loss via counterfactual forecasting.
    
    Implements Equations 6-9 from the paper:
    - Eq. 6: R(a) = ||E^(a)||²  (harm scalarization)
    - Eq. 7: π_ref(a|s) ∝ exp(-R(a)/τ)  (softmin reference)
    - Eq. 8: E^ref = Σ_a π_ref(a) E^(a)  (reference embedding)
    - Eq. 9: AR_t = ||E_t - E^ref||² + κ·neighbor_harm
    """
    
    def __init__(self, kappa: float = 0.5, temperature: float = 1.0):
        """
        Args:
            kappa: Weight for neighbor harm penalty
            temperature: Softmin temperature (annealed during training)
        """
        super().__init__()
        self.kappa = kappa
        self.temperature = temperature
    
    def compute_harm_values(self, E_preds: torch.Tensor):
        """
        Compute harm scalarization R(a) for each action (Equation 6).
        
        Args:
            E_preds: Predicted IAEs for each action (num_actions, iae_dim)
        
        Returns:
            R_values: Harm values (num_actions,)
        """
        # R(a) = ||E^(a)||²
        R_values = torch.norm(E_preds, dim=1, p=2) ** 2
        
        return R_values
    
    def compute_softmin_reference(self, E_preds: torch.Tensor, return_weights=False):
        """
        Compute softmin reference embedding (Equations 7-8).
        
        Args:
            E_preds: Predicted IAEs for each action (num_actions, iae_dim)
            return_weights: If True, also return π_ref weights
        
        Returns:
            E_ref: Reference embedding (iae_dim,)
            weights: Softmin weights (optional, num_actions)
        """
        # Equation 6: Compute harm scalarization
        R_values = self.compute_harm_values(E_preds)
        
        # Equation 7: Softmin reference distribution
        # π_ref(a|s) = exp(-R(a)/τ) / Σ_a' exp(-R(a')/τ)
        log_weights = -R_values / (self.temperature + 1e-8)
        weights = F.softmax(log_weights, dim=0)
        
        # Equation 8: Expected reference embedding
        # E^ref = Σ_a π_ref(a) E^(a)
        E_ref = torch.sum(weights.unsqueeze(1) * E_preds, dim=0)
        
        if return_weights:
            return E_ref, weights
        else:
            return E_ref
    
    def forward(self, E_current: torch.Tensor, E_preds: torch.Tensor,
                E_neighbors: torch.Tensor = None, return_info=False):
        """
        Compute alignment regret (Equation 9).
        
        Args:
            E_current: Current IAE (iae_dim,)
            E_preds: Predicted IAEs for candidate actions (num_actions, iae_dim)
            E_neighbors: Neighbors' IAEs (num_neighbors, iae_dim) [optional]
            return_info: If True, return diagnostic information
        
        Returns:
            regret: Alignment regret scalar
            info: Diagnostic dict (optional)
        """
        # Compute reference embedding (Equations 7-8)
        E_ref, weights = self.compute_softmin_reference(E_preds, return_weights=True)
        
        # Primary regret term: ||E_t - E^ref||²
        self_regret = torch.sum((E_current - E_ref) ** 2)
        
        # Neighbor harm penalty (multi-agent term)
        neighbor_harm = torch.tensor(0.0, device=E_current.device)
        if E_neighbors is not None and len(E_neighbors) > 0:
            # Average L2 norm of neighbor IAEs
            neighbor_norms = torch.norm(E_neighbors, dim=1, p=2)
            neighbor_harm = torch.mean(neighbor_norms ** 2)
        
        # Total alignment regret (Equation 9)
        regret = self_regret + self.kappa * neighbor_harm
        
        if return_info:
            # Compute harm values for diagnostics
            R_values = self.compute_harm_values(E_preds)
            
            info = {
                'E_ref': E_ref.detach(),
                'weights': weights.detach(),
                'R_values': R_values.detach(),
                'self_regret': self_regret.detach().item(),
                'neighbor_harm': neighbor_harm.detach().item(),
                'total_regret': regret.detach().item()
            }
            return regret, info
        else:
            return regret


class PPOLoss(nn.Module):
    """
    PPO-Clip loss with value clipping and entropy regularization.
    
    Implements the clipped surrogate objective from Schulman et al. 2017.
    """
    
    def __init__(self, clip_epsilon: float = 0.2, entropy_coef: float = 0.01,
                 value_clip: bool = False, value_clip_epsilon: float = 0.2):
        """
        Args:
            clip_epsilon: PPO clipping parameter (ε)
            entropy_coef: Entropy regularization coefficient
            value_clip: Whether to clip value function loss
            value_clip_epsilon: Value clipping parameter
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.value_clip_epsilon = value_clip_epsilon
    
    def compute_entropy(self, log_probs, logits=None):
        """
        Compute policy entropy.
        
        Args:
            log_probs: Log probabilities of selected actions (batch,)
            logits: Raw logits (batch, action_dim) [optional, for better entropy]
        
        Returns:
            entropy: Scalar entropy value
        """
        if logits is not None:
            # More accurate: H = -Σ p(a) log p(a)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs_all, dim=-1).mean()
        else:
            # Approximate from selected actions only
            entropy = -log_probs.mean()
        
        return entropy
    
    def forward(self, log_probs, old_log_probs, advantages, values, returns,
                old_values=None, logits=None):
        """
        Compute PPO clipped loss.
        
        Args:
            log_probs: Current policy log probabilities (batch,)
            old_log_probs: Old policy log probabilities (batch,)
            advantages: Advantage estimates (batch,)
            values: Current value predictions (batch,)
            returns: Target returns (batch,)
            old_values: Old value predictions (batch,) [optional, for value clipping]
            logits: Current policy logits (batch, action_dim) [optional, for entropy]
        
        Returns:
            policy_loss: Clipped policy gradient loss
            value_loss: Value function MSE loss
            entropy: Policy entropy
        """
        # Policy loss with clipping
        # ratio = π_θ(a|s) / π_θ_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with optional clipping
        if self.value_clip and old_values is not None:
            # Clipped value loss (helps stability)
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.value_clip_epsilon,
                self.value_clip_epsilon
            )
            value_loss_unclipped = F.mse_loss(values, returns)
            value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        else:
            # Standard MSE value loss
            value_loss = F.mse_loss(values, returns)
        
        # Entropy regularization
        entropy = self.compute_entropy(log_probs, logits)
        
        return policy_loss, value_loss, entropy
    
    def compute_metrics(self, log_probs, old_log_probs, advantages, values, returns):
        """
        Compute diagnostic metrics (clipfrac, approx_kl, explained_var).
        
        Args:
            log_probs: Current policy log probabilities (batch,)
            old_log_probs: Old policy log probabilities (batch,)
            advantages: Advantage estimates (batch,)
            values: Current value predictions (batch,)
            returns: Target returns (batch,)
        
        Returns:
            metrics: Dictionary of diagnostic metrics
        """
        with torch.no_grad():
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clip fraction (fraction of samples clipped)
            clipped = torch.abs(ratio - 1.0) > self.clip_epsilon
            clipfrac = torch.mean(clipped.float()).item()
            
            # Approximate KL divergence
            approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio)).item()
            
            # Explained variance
            var_y = torch.var(returns)
            if var_y > 0:
                explained_var = 1.0 - torch.var(returns - values) / var_y
                explained_var = explained_var.item()
            else:
                explained_var = 1.0 if torch.var(values) == 0 else 0.0
            
            metrics = {
                'clipfrac': clipfrac,
                'approx_kl': approx_kl,
                'explained_variance': explained_var
            }
        
        return metrics


class ForecastLoss(nn.Module):
    """
    Supervised loss for training the counterfactual forecaster.
    
    Trains h_ψ to predict E_{t+1} given (s_t, a_t, r_t).
    """
    
    def __init__(self, loss_type='mse', huber_delta=1.0):
        """
        Args:
            loss_type: 'mse', 'mae', or 'huber'
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
    
    def forward(self, E_pred, E_target):
        """
        Compute forecast loss.
        
        Args:
            E_pred: Predicted next IAE (batch, iae_dim)
            E_target: True next IAE (batch, iae_dim)
        
        Returns:
            loss: Scalar loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(E_pred, E_target)
        elif self.loss_type == 'mae':
            loss = F.l1_loss(E_pred, E_target)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(E_pred, E_target, delta=self.huber_delta)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def compute_accuracy(self, E_pred, E_target, threshold=0.5):
        """
        Compute forecast accuracy (fraction within threshold).
        
        Args:
            E_pred: Predicted next IAE (batch, iae_dim)
            E_target: True next IAE (batch, iae_dim)
            threshold: Distance threshold for "correct" prediction
        
        Returns:
            accuracy: Fraction of predictions within threshold
        """
        with torch.no_grad():
            distances = torch.norm(E_pred - E_target, dim=1, p=2)
            correct = distances < threshold
            accuracy = torch.mean(correct.float()).item()
        
        return accuracy


class TotalLoss(nn.Module):
    """
    Combined loss for ESAI-v3 training.
    
    Combines PPO loss, alignment regret, and regularizers:
    L = L_policy + c_v L_value - c_h H + λ_reg AR_t + λ_H ||H||² + λ_bias L_bias
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration dict with loss coefficients
        """
        super().__init__()
        
        self.ppo_loss = PPOLoss(
            clip_epsilon=cfg.get('clip_epsilon', 0.2),
            entropy_coef=cfg.get('entropy_coef', 0.01),
            value_clip=cfg.get('value_clip', False)
        )
        
        self.alignment_loss = AlignmentLoss(
            kappa=cfg.get('kappa', 0.5),
            temperature=1.0  # Set externally during training
        )
        
        self.forecast_loss = ForecastLoss(
            loss_type=cfg.get('forecast_loss_type', 'mse')
        )
        
        # Loss coefficients
        self.value_coef = cfg.get('value_coef', 0.5)
        self.lambda_reg = cfg.get('lambda_reg', 0.2)
        self.lambda_H = cfg.get('lambda_H', 1e-3)
        self.lambda_bias = cfg.get('lambda_bias', 0.0)
    
    def forward(self, policy_data, agent):
        """
        Compute total loss.
        
        Args:
            policy_data: Dict with log_probs, old_log_probs, advantages, values, returns
            agent: ESAI-v3 agent
        
        Returns:
            total_loss: Combined loss
            losses_dict: Dictionary of individual loss components
        """
        # PPO loss
        policy_loss, value_loss, entropy = self.ppo_loss(
            policy_data['log_probs'],
            policy_data['old_log_probs'],
            policy_data['advantages'],
            policy_data['values'],
            policy_data['returns'],
            old_values=policy_data.get('old_values'),
            logits=policy_data.get('logits')
        )
        
        # Alignment regret (if enabled)
        if agent.use_alignment_regret and 'AR_t' in policy_data:
            alignment_regret = policy_data['AR_t']
        else:
            alignment_regret = torch.tensor(0.0, device=policy_loss.device)
        
        # Hebbian regularization
        if agent.use_hebbian:
            hebb_reg = agent.hebbian.get_norm() ** 2
        else:
            hebb_reg = torch.tensor(0.0, device=policy_loss.device)
        
        # Bias suppression
        if agent.use_diffusion and agent.num_agents > 1:
            bias_penalty = agent.diffusion.compute_bias_penalty()
        else:
            bias_penalty = torch.tensor(0.0, device=policy_loss.device)
        
        # Combined loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            entropy +  # Negative because we maximize entropy
            self.lambda_reg * alignment_regret +
            self.lambda_H * hebb_reg +
            self.lambda_bias * bias_penalty
        )
        
        # Package losses
        losses_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'alignment_regret': alignment_regret.item() if isinstance(alignment_regret, torch.Tensor) else alignment_regret,
            'hebb_reg': hebb_reg.item() if isinstance(hebb_reg, torch.Tensor) else hebb_reg,
            'bias_penalty': bias_penalty.item() if isinstance(bias_penalty, torch.Tensor) else bias_penalty,
            'total_loss': total_loss.item()
        }
        
        return total_loss, losses_dict


def compute_loss_diagnostics(losses_dict, global_step):
    """
    Format loss diagnostics for logging.
    
    Args:
        losses_dict: Dictionary of loss components
        global_step: Current training step
    
    Returns:
        diagnostics_str: Formatted string
    """
    diag = f"\n[LOSSES @ step {global_step}]\n"
    diag += f"  Policy:    {losses_dict['policy_loss']:.4f}\n"
    diag += f"  Value:     {losses_dict['value_loss']:.4f}\n"
    diag += f"  Entropy:   {losses_dict['entropy']:.4f}\n"
    diag += f"  AR:        {losses_dict['alignment_regret']:.4f}\n"
    diag += f"  Hebb Reg:  {losses_dict['hebb_reg']:.6f}\n"
    diag += f"  Bias Pen:  {losses_dict['bias_penalty']:.6f}\n"
    diag += f"  TOTAL:     {losses_dict['total_loss']:.4f}\n"
    
    return diag