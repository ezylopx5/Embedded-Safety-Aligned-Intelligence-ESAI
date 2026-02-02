"""
Utility functions for ESAI-v3.
Includes seeding, spectral normalization, GAE, and helper functions.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def spectral_norm_matrix(matrix: torch.Tensor, max_spectral_radius: float = 0.95,
                         use_power_iteration: bool = True, n_iterations: int = 10):
    """
    Apply spectral normalization to ensure bounded spectral radius.
    
    Args:
        matrix: Input matrix (e.g., Laplacian or weight matrix)
        max_spectral_radius: Maximum allowed spectral radius
        use_power_iteration: If True, use O(n) power iteration; else O(n^3) eigvals
        n_iterations: Number of power iteration steps
    
    Returns:
        Normalized matrix
    """
    with torch.no_grad():
        if use_power_iteration:
            # Power iteration: O(n) per iteration, much faster for large matrices
            v = torch.randn(matrix.size(0), device=matrix.device)
            v = v / v.norm()
            for _ in range(n_iterations):
                v = matrix @ v
                v = v / (v.norm() + 1e-8)
            spectral_radius = (v @ matrix @ v).abs().item()
        else:
            # Full eigenvalue decomposition: O(n^3), exact but slow
            if matrix.is_complex():
                eigenvalues = torch.linalg.eigvals(matrix)
            else:
                eigenvalues = torch.linalg.eigvals(matrix.float())
            spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        
        if spectral_radius > max_spectral_radius:
            matrix = matrix * (max_spectral_radius / spectral_radius)
    
    return matrix


def compute_laplacian(adjacency: torch.Tensor, normalized: bool = True):
    """
    Compute graph Laplacian from adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix (N x N)
        normalized: If True, compute normalized Laplacian
    
    Returns:
        Laplacian matrix
    """
    # Degree matrix
    degree = torch.sum(adjacency, dim=1)
    
    if normalized:
        # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        laplacian = torch.eye(adjacency.size(0), device=adjacency.device) - \
                    degree_inv_sqrt @ adjacency @ degree_inv_sqrt
    else:
        # Standard Laplacian: D - A
        laplacian = torch.diag(degree) - adjacency
    
    return laplacian


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        eps: Small constant for numerical stability
    
    Returns:
        Cosine similarity in [0, 1]
    """
    x_norm = torch.norm(x, p=2) + eps
    y_norm = torch.norm(y, p=2) + eps
    similarity = torch.dot(x, y) / (x_norm * y_norm)
    return torch.clamp(similarity, 0.0, 1.0)


def apply_gradient_clip(parameters, max_norm: float = 1.0):
    """
    Clip gradients by global norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
    
    Returns:
        Total norm of parameters (before clipping)
    """
    if max_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
    return 0.0


class SpectralNormLinear(nn.Linear):
    """Linear layer with spectral normalization."""
    
    def __init__(self, in_features, out_features, bias=True, spectral_radius=0.95):
        super().__init__(in_features, out_features, bias)
        self.spectral_radius = spectral_radius
    
    def forward(self, x):
        # Normalize weight matrix
        with torch.no_grad():
            self.weight.data = spectral_norm_matrix(
                self.weight.data,
                max_spectral_radius=self.spectral_radius
            )
        return super().forward(x)


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0):
    """
    Softmax with temperature scaling.
    
    Args:
        logits: Input logits
        temperature: Temperature parameter (lower = sharper)
    
    Returns:
        Softmax probabilities
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)


def anneal_parameter(step: int, init_val: float, final_val: float, total_steps: int):
    """
    Exponential annealing of a parameter.
    
    Args:
        step: Current step
        init_val: Initial value
        final_val: Final value
        total_steps: Total annealing steps
    
    Returns:
        Annealed value
    """
    if step >= total_steps:
        return final_val
    
    if init_val == final_val:
        return init_val
    
    alpha = step / total_steps
    # Avoid log(0) issues
    if init_val <= 0 or final_val <= 0:
        # Linear annealing for non-positive values
        return init_val + alpha * (final_val - init_val)
    
    return init_val * np.exp(alpha * np.log(final_val / init_val))


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], 
                gamma: float = 0.99, lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        lambda_: GAE parameter
    
    Returns:
        returns: numpy array of returns
        advantages: numpy array of advantages
    """
    if len(rewards) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    
    advantages = []
    returns = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
        
        # TD error
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE
        gae = delta + gamma * lambda_ * next_non_terminal * gae
        
        # Insert at beginning
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    # Convert to numpy arrays (breaks gradient graph)
    advantages = np.array(advantages, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)
    
    return returns, advantages


def normalize_advantages(advantages: Union[torch.Tensor, np.ndarray], 
                        eps: float = 1e-8) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantages to normalize
        eps: Small constant for numerical stability
    
    Returns:
        Normalized advantages
    """
    if isinstance(advantages, torch.Tensor):
        return (advantages - advantages.mean()) / (advantages.std() + eps)
    else:
        return (advantages - np.mean(advantages)) / (np.std(advantages) + eps)


def create_adjacency_matrix(num_agents: int, topology: str = 'full', 
                           device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create an adjacency matrix for multi-agent graph.
    
    Args:
        num_agents: Number of agents
        topology: Graph topology ('full', 'ring', 'grid', 'star')
        device: Torch device
    
    Returns:
        Adjacency matrix (num_agents x num_agents)
    """
    if device is None:
        device = torch.device('cpu')
    
    if topology == 'full':
        # Fully connected (complete graph)
        adj = torch.ones(num_agents, num_agents, device=device)
        adj = adj - torch.eye(num_agents, device=device)
    
    elif topology == 'ring':
        # Ring topology
        adj = torch.zeros(num_agents, num_agents, device=device)
        for i in range(num_agents):
            adj[i, (i + 1) % num_agents] = 1
            adj[i, (i - 1) % num_agents] = 1
    
    elif topology == 'grid':
        # 2D grid topology (assumes square number of agents)
        grid_size = int(np.sqrt(num_agents))
        if grid_size ** 2 != num_agents:
            raise ValueError(f"Grid topology requires square number of agents, got {num_agents}")
        
        adj = torch.zeros(num_agents, num_agents, device=device)
        for i in range(num_agents):
            row, col = i // grid_size, i % grid_size
            # Connect to neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                    j = new_row * grid_size + new_col
                    adj[i, j] = 1
    
    elif topology == 'star':
        # Star topology (agent 0 is hub)
        adj = torch.zeros(num_agents, num_agents, device=device)
        for i in range(1, num_agents):
            adj[0, i] = 1
            adj[i, 0] = 1
    
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    return adj


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy from logits.
    
    Args:
        logits: Action logits (batch_size, action_dim)
    
    Returns:
        Entropy (scalar or batch_size tensor)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute fraction of variance explained by prediction.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Explained variance ratio
    """
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0 if np.var(y_pred) == 0 else -1.0
    
    return 1 - np.var(y_true - y_pred) / var_true


def polyak_average(target_params, source_params, tau: float = 0.995):
    """
    Polyak (exponential moving) average update for target network.
    
    Args:
        target_params: Target network parameters
        source_params: Source network parameters
        tau: Interpolation factor (0 = copy source, 1 = keep target)
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            target_param.data.mul_(tau).add_(source_param.data, alpha=1 - tau)


def discount_cumsum(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Compute discounted cumulative sum of rewards.
    
    Args:
        rewards: Array of rewards
        gamma: Discount factor
    
    Returns:
        Discounted cumulative sum
    """
    discounted = np.zeros_like(rewards)
    running_sum = 0
    
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        discounted[t] = running_sum
    
    return discounted


class RunningMeanStd:
    """
    Running statistics tracker for normalization.
    """
    
    def __init__(self, shape=(), eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps
    
    def update(self, x):
        """Update statistics with new batch."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        """Normalize input using running statistics."""
        # Clamp variance to prevent NaN from numerical precision issues
        safe_var = np.maximum(self.var, 0.0)
        return (x - self.mean) / np.sqrt(safe_var + 1e-8)
    
    def denormalize(self, x):
        """Denormalize input."""
        return x * np.sqrt(self.var + 1e-8) + self.mean


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)