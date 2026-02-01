"""
ESAI-v3 Model Implementation.
Core agent with IAE dynamics, attention, Hebbian memory, and graph diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class AttentionGating(nn.Module):
    """IAE-weighted attention mechanism."""
    
    def __init__(self, iae_dim, obs_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(iae_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim),
            nn.Sigmoid()
        )
    
    def forward(self, E, obs):
        """
        Apply attention gating.
        
        Args:
            E: IAE vector (batch_size, iae_dim)
            obs: Observation (batch_size, obs_dim)
        
        Returns:
            Attended observation (batch_size, obs_dim)
        """
        attention_weights = self.attention_net(E)
        return attention_weights * obs


class HebbianMemory(nn.Module):
    """Differentiable Hebbian memory for temporal credit assignment."""
    
    def __init__(self, iae_dim, memory_dim=32, eta=1e-3, delta=0.02):
        super().__init__()
        self.iae_dim = iae_dim
        self.memory_dim = memory_dim
        self.eta = eta  # Learning rate
        self.delta = delta  # Decay rate
        
        # Initialize memory matrix
        self.H = nn.Parameter(torch.zeros(iae_dim, memory_dim), requires_grad=False)
        
        # Read network
        self.read_net = nn.Linear(iae_dim * memory_dim, iae_dim)
    
    def update(self, E, z):
        """
        Update Hebbian trace: H_{t+1} = (1-δ)H_t + η(E ⊗ z)
        
        Args:
            E: IAE vector (iae_dim,)
            z: Percept vector (memory_dim,)
        """
        with torch.no_grad():
            # Ensure proper dimensions
            if E.dim() == 1:
                E = E.unsqueeze(1)  # (iae_dim, 1)
            if z.dim() == 1:
                z = z.unsqueeze(0)  # (1, memory_dim)
            
            # Outer product
            outer = torch.mm(E, z)  # (iae_dim, memory_dim)
            
            # Hebbian update with decay
            self.H.data = (1 - self.delta) * self.H.data + self.eta * outer
    
    def read(self, E):
        """
        Read from memory given current IAE.
        
        Args:
            E: IAE vector (batch_size, iae_dim) or (iae_dim,)
        
        Returns:
            Memory readout (batch_size, iae_dim) or (iae_dim,)
        """
        # Flatten memory for read network
        H_flat = self.H.flatten()
        
        # Handle both batched and single inputs
        if E.dim() == 1:
            H_expanded = H_flat.unsqueeze(0)
        else:
            batch_size = E.size(0)
            H_expanded = H_flat.unsqueeze(0).expand(batch_size, -1)
        
        return self.read_net(H_expanded)
    
    def get_norm(self):
        """Get Frobenius norm for regularization."""
        return torch.norm(self.H, p='fro')


class GraphDiffusion(nn.Module):
    """Graph diffusion with similarity-weighted edges."""
    
    def __init__(self, num_agents, iae_dim, identity_dim=8, alpha=0.05):
        super().__init__()
        self.num_agents = num_agents
        self.iae_dim = iae_dim
        self.alpha = alpha
        
        # Identity embeddings for similarity computation
        self.identity_embeddings = nn.Parameter(
            torch.randn(num_agents, identity_dim) * 0.1
        )
        
        # Bias suppression weight
        self.lambda_bias = 0.1
    
    def compute_similarity_matrix(self):
        """Compute pairwise cosine similarities."""
        # Normalize embeddings
        normed = F.normalize(self.identity_embeddings, dim=1)
        
        # Pairwise cosine similarity
        S = torch.mm(normed, normed.t())
        
        # Zero out diagonal (no self-connections)
        S = S - torch.diag(torch.diag(S))
        
        # Clamp to [0, 1]
        S = torch.clamp(S, 0, 1)
        
        return S
    
    def compute_laplacian(self, S):
        """Compute normalized graph Laplacian."""
        # Degree matrix
        D = torch.diag(S.sum(dim=1))
        
        # Normalized Laplacian: I - D^(-1/2) S D^(-1/2)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D) + 1e-8))
        L = torch.eye(self.num_agents, device=S.device) - D_inv_sqrt @ S @ D_inv_sqrt
        
        # Spectral normalization for stability
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvalsh(L)
            max_eigenvalue = eigenvalues.max().item()
            if max_eigenvalue > 2.0:
                L = L * (1.95 / max_eigenvalue)
        
        return L
    
    def diffuse(self, E_all):
        """
        Apply graph diffusion: E' = E - α L E
        
        Args:
            E_all: IAEs for all agents (num_agents, iae_dim)
        
        Returns:
            Diffused IAEs (num_agents, iae_dim)
        """
        S = self.compute_similarity_matrix()
        L = self.compute_laplacian(S)
        
        # Diffusion step
        E_diffused = E_all - self.alpha * torch.mm(L, E_all)
        
        return E_diffused
    
    def compute_bias_penalty(self):
        """Compute bias penalty for regularization."""
        S = self.compute_similarity_matrix()
        # Penalize high similarity (in-group favoritism)
        return self.lambda_bias * torch.sum(S ** 2)


class ESAIv3Agent(nn.Module):
    """
    ESAI-v3 Agent with internal alignment embedding.
    """
    
    def __init__(self, obs_dim, action_dim, iae_dim=32, hidden_dim=128,
                 gamma_E=0.9, alpha_diffusion=0.05, num_agents=1,
                 use_attention=True, use_hebbian=True, use_diffusion=True,
                 use_alignment_regret=True, lambda_bias=0.0):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.iae_dim = iae_dim
        self.hidden_dim = hidden_dim
        self.gamma_E = gamma_E
        self.num_agents = num_agents
        
        # Method flags
        self.use_attention = use_attention
        self.use_hebbian = use_hebbian
        self.use_diffusion = use_diffusion
        self.use_alignment_regret = use_alignment_regret
        
        # Initialize IAE
        self.E = torch.zeros(iae_dim)
        
        # IAE dynamics network g_φ
        self.iae_dynamics = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim),  # obs + action + reward
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, iae_dim)
        )
        
        # Policy network
        policy_input_dim = obs_dim + iae_dim
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional modules
        if use_attention:
            self.attention = AttentionGating(iae_dim, obs_dim)
            self.attention_weights = nn.Linear(iae_dim, obs_dim)
        
        if use_hebbian:
            self.hebbian = HebbianMemory(iae_dim, memory_dim=32)
            # Gated Hebbian contribution (Option 3)
            self.hebbian_gate = nn.Sequential(
                nn.Linear(iae_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.hebbian = None
            self.hebbian_gate = None
        
        if use_diffusion and num_agents > 1:
            self.diffusion = GraphDiffusion(
                num_agents, iae_dim, alpha=alpha_diffusion
            )
            self.diffusion.lambda_bias = lambda_bias
        
        # CRITICAL FIX: Add alignment regret components
        if use_alignment_regret:
            # Forecaster network predicts E^(a)_{t+1} for each action
            # Input: obs + action_onehot + reward + current_E [+ hebbian_readout if enabled]
            forecaster_input_dim = obs_dim + action_dim + 1 + iae_dim
            
            if use_hebbian:
                forecaster_input_dim += iae_dim  # Add Hebbian readout dimension
            
            self.forecast_net = nn.Sequential(
                nn.Linear(forecaster_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, iae_dim),
                nn.Tanh()  # Bounded output for stability
            )
            
            # EMA target network (critical for stability)
            self.forecast_net_target = copy.deepcopy(self.forecast_net)
            for p in self.forecast_net_target.parameters():
                p.requires_grad = False
            
            # Alignment loss module
            from esaiv3.loss import AlignmentLoss
            self.alignment_loss = AlignmentLoss(kappa=0.5, temperature=1.0)
            
            # Temperature for annealing
            self.temperature = 1.0
        
        # Agent identity (for multi-agent)
        self.agent_id = 0
    
    def reset_iae(self):
        """Reset IAE to initial state."""
        self.E = torch.zeros(self.iae_dim, device=self.E.device)
        if self.use_hebbian:
            self.hebbian.H.data.zero_()
    
    def to(self, device):
        """Override to method to handle E tensor."""
        super().to(device)
        self.E = self.E.to(device)
        return self
    
    def update_iae(self, obs, action, harm_t):
        """
        Update IAE dynamics: E_{t+1} = γ_E E_t + g_φ(z,a,h) - α L E_t
        
        CRITICAL: Uses harm signal (not reward) to grow IAE when agent causes harm.
        This aligns with paper theory - IAE should grow when harm is caused.
        
        Args:
            obs: Observation (batch_size, obs_dim) or (obs_dim,)
            action: Action taken (batch_size,) or scalar
            harm_t: Harm signal from environment (>0 when agent harms others)
        """
        # Ensure proper dimensions
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # One-hot encode action
        if isinstance(action, int):
            action_tensor = torch.zeros(self.action_dim, device=obs.device)
            action_tensor[action] = 1.0
        else:
            action_tensor = F.one_hot(action, num_classes=self.action_dim).float()
            if action_tensor.dim() == 2 and action_tensor.size(0) == 1:
                action_tensor = action_tensor.squeeze(0)
        
        # Prepare harm signal as input (IAE grows when harm > 0)
        if not isinstance(harm_t, torch.Tensor):
            harm_t = torch.tensor([harm_t], dtype=torch.float32, device=obs.device)
        if harm_t.dim() == 0:
            harm_t = harm_t.unsqueeze(0)
        
        # IAE dynamics input: obs + action + harm signal
        dynamics_input = torch.cat([
            obs.squeeze(0) if obs.size(0) == 1 else obs[0],
            action_tensor,
            harm_t
        ])
        
        # Compute dynamics
        delta_E = self.iae_dynamics(dynamics_input)
        
        # Update with persistence
        self.E = self.gamma_E * self.E + delta_E
        
        # Apply diffusion if multi-agent
        if self.use_diffusion and self.num_agents > 1:
            # Placeholder for multi-agent case
            # In practice, would need all agents' IAEs
            pass
        
        # Update Hebbian memory
        if self.use_hebbian:
            # Use first 32 dims of obs as memory input
            memory_input = obs.squeeze(0)[:min(32, obs.size(-1))]
            if memory_input.size(0) < 32:
                # Pad if needed
                padding = torch.zeros(32 - memory_input.size(0), device=obs.device)
                memory_input = torch.cat([memory_input, padding])
            self.hebbian.update(self.E, memory_input)
        
        # Bound IAE norm for stability
        with torch.no_grad():
            if self.E.norm() > 10.0:
                self.E = self.E * (10.0 / self.E.norm())
    
    def apply_attention(self, obs):
        """Apply IAE-weighted attention to observation."""
        if not self.use_attention:
            return obs
        
        # Compute attention weights
        alpha = torch.sigmoid(self.attention_weights(self.E))
        
        # Apply attention
        if obs.dim() == 1:
            return alpha * obs
        else:
            return alpha.unsqueeze(0) * obs
    
    def compute_value(self, obs):
        """Compute state value."""
        if self.use_attention:
            obs = self.apply_attention(obs)
        
        # Concatenate observation and IAE
        if obs.dim() == 1:
            value_input = torch.cat([obs, self.E])
        else:
            value_input = torch.cat([obs, self.E.unsqueeze(0)], dim=-1)
        
        return self.value_net(value_input).squeeze()
    
    def act(self, obs, deterministic=False):
        """
        Select action with alignment regret computation.
        
        Args:
            obs: Observation tensor
            deterministic: If True, select argmax action
        
        Returns:
            action: Selected action
            extra: Dictionary with diagnostics
        """
        # Apply attention if enabled
        if self.use_attention:
            obs_attended = self.apply_attention(obs)
        else:
            obs_attended = obs
        
        # Policy input
        if obs_attended.dim() == 1:
            policy_input = torch.cat([obs_attended, self.E])
        else:
            policy_input = torch.cat([obs_attended, self.E.unsqueeze(0)], dim=-1)
        
        # Get action logits
        logits = self.policy_net(policy_input)
        
        # Sample or take argmax
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            if probs.dim() == 1:
                action = torch.multinomial(probs, 1).squeeze()
            else:
                action = torch.multinomial(probs.squeeze(0), 1).squeeze()
        
        # CRITICAL FIX: Compute alignment regret
        AR_t = torch.tensor(0.0, device=obs.device)
        
        if self.use_alignment_regret and self.forecast_net is not None:
            # Forecast E^(a)_{t+1} for ALL actions
            E_preds = []
            
            # Ensure obs is 1D for concatenation
            obs_flat = obs.flatten()
            
            # Get Hebbian readout if enabled
            hebbian_readout = None
            if self.use_hebbian and self.hebbian is not None:
                hebbian_readout = self.hebbian.read(self.E)
            
            for a in range(self.action_dim):
                # One-hot encode action
                action_onehot = torch.zeros(self.action_dim, device=obs.device)
                action_onehot[a] = 1.0
                
                # Forecast input: obs + action + dummy_reward + current_E [+ hebbian_readout]
                input_parts = [
                    obs_flat,
                    action_onehot,
                    torch.zeros(1, device=obs.device),  # Placeholder reward
                    self.E
                ]
                if hebbian_readout is not None:
                    input_parts.append(hebbian_readout.squeeze())
                forecast_input = torch.cat(input_parts)
                
                # Predict using TARGET network for stability
                with torch.no_grad():
                    E_pred = self.forecast_net_target(forecast_input)
                E_preds.append(E_pred)
            
            E_preds = torch.stack(E_preds)  # [action_dim, iae_dim]
            
            # Update alignment loss temperature
            self.alignment_loss.temperature = self.temperature
            
            # Compute alignment regret (state-based AR)
            # AR = ||E_current - E_ref||² where E_ref is the softmin-weighted reference
            with torch.no_grad():
                AR_t = self.alignment_loss(
                    E_current=self.E,
                    E_preds=E_preds,
                    E_neighbors=None
                )
        
        # Prepare extra info
        extra = {
            'E': self.E.detach().cpu().numpy(),
            'AR_t': AR_t.item() if isinstance(AR_t, torch.Tensor) else AR_t,
            'pr_flag': None,  # Set by environment
            'E_norm': self.E.norm().item()
        }
        
        return action, extra
    
    def update_target_forecaster(self, tau=0.995):
        """Update EMA target network for forecaster."""
        if not self.use_alignment_regret:
            return
        
        with torch.no_grad():
            for target_param, online_param in zip(
                self.forecast_net_target.parameters(),
                self.forecast_net.parameters()
            ):
                target_param.data.copy_(
                    tau * target_param.data + (1 - tau) * online_param.data
                )
    
    def train_forecaster(self, obs, action, reward, next_E, optimizer):
        """
        Train the forecaster network to predict next IAE.
        
        Args:
            obs: Observation that led to action
            action: Action taken
            reward: Reward received
            next_E: Actual next IAE (target)
            optimizer: Optimizer for forecaster
        
        Returns:
            forecast_loss: MSE loss value
        """
        if not self.use_alignment_regret:
            return 0.0
        
        # Prepare input
        obs_flat = obs.flatten()
        
        # One-hot encode action
        if isinstance(action, int):
            action_onehot = torch.zeros(self.action_dim, device=obs.device)
            action_onehot[action] = 1.0
        else:
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            if action_onehot.dim() == 2:
                action_onehot = action_onehot.squeeze(0)
        
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32, device=obs.device)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        
        # Forecast input: obs + action + reward + current_E [+ hebbian_readout]
        input_parts = [
            obs_flat,
            action_onehot,
            reward,
            self.E
        ]
        if self.use_hebbian and self.hebbian is not None:
            hebbian_readout = self.hebbian.read(self.E)
            input_parts.append(hebbian_readout.squeeze())
        forecast_input = torch.cat(input_parts)
        
        # Predict
        E_pred = self.forecast_net(forecast_input)
        
        # Compute loss
        forecast_loss = F.mse_loss(E_pred, next_E.detach())
        
        # Backward
        optimizer.zero_grad()
        forecast_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forecast_net.parameters(), 1.0)
        optimizer.step()
        
        return forecast_loss.item()