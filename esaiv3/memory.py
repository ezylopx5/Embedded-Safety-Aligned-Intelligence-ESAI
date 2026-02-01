"""
Hebbian memory implementation for ESAI-v3.
Implements differentiable affect-memory coupling.
"""

import torch
import torch.nn as nn


class HebbianMemory(nn.Module):
    """
    Hebbian memory trace: H_{t+1} = (1-δ)H_t + η(E_t ⊗ z_t)
    
    Supports:
    - Differentiable outer-product updates
    - Decay for bounded norms
    - Read operations for forecasting
    """
    
    def __init__(self, iae_dim: int, obs_dim: int, eta: float = 1e-3, delta: float = 0.02):
        """
        Args:
            iae_dim: Dimension of internal alignment embedding
            obs_dim: Dimension of observation
            eta: Learning rate for Hebbian update
            delta: Decay rate
        """
        super().__init__()
        
        self.iae_dim = iae_dim
        self.obs_dim = obs_dim
        self.eta = eta
        self.delta = delta
        
        # Memory matrix: iae_dim x obs_dim
        self.register_buffer('H', torch.zeros(iae_dim, obs_dim))
        
        # Read projection
        self.read_proj = nn.Linear(iae_dim * obs_dim, iae_dim)
    
    def update(self, E: torch.Tensor, z: torch.Tensor):
        """
        Update Hebbian trace.
        
        Args:
            E: IAE vector (iae_dim,)
            z: Observation vector (obs_dim,)
        """
        # Ensure proper dimensions
        if E.dim() == 1:
            E = E.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Outer product: E ⊗ z
        outer = torch.bmm(E.unsqueeze(2), z.unsqueeze(1)).squeeze(0)
        
        # Hebbian update with decay
        self.H = (1 - self.delta) * self.H + self.eta * outer
    
    def read(self):
        """
        Read from memory for forecasting.
        
        Returns:
            Memory read vector (iae_dim,)
        """
        # Flatten and project
        h_flat = self.H.flatten()
        return self.read_proj(h_flat)
    
    def reset(self):
        """Reset memory to zero."""
        self.H.zero_()
    
    def get_norm(self):
        """Get Frobenius norm of memory."""
        return torch.norm(self.H, p='fro')