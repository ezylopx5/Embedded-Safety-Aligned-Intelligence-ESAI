"""
Graph diffusion for ESAI-v3.
Implements similarity-weighted diffusion with bias controls.
"""

import torch
import torch.nn as nn
from .utils import compute_laplacian, cosine_similarity


class GraphDiffusion(nn.Module):
    """
    Graph diffusion: E_{t+1} ← E_t - α L E_t
    
    Features:
    - Similarity-weighted adjacency
    - Spectral normalization
    - Bias suppression regularizer
    """
    
    def __init__(self, num_agents: int, iae_dim: int, alpha: float = 0.1,
                 lambda_bias: float = 0.0, identity_dim: int = 8):
        """
        Args:
            num_agents: Number of agents
            iae_dim: IAE dimension
            alpha: Diffusion strength
            lambda_bias: Bias suppression regularizer weight
            identity_dim: Dimension of identity embeddings
        """
        super().__init__()
        
        self.num_agents = num_agents
        self.iae_dim = iae_dim
        self.alpha = alpha
        self.lambda_bias = lambda_bias
        
        # Learnable identity embeddings
        self.identity_embeddings = nn.Parameter(torch.randn(num_agents, identity_dim))
        
        # Adjacency (learned or fixed)
        self.register_buffer('base_adjacency', torch.ones(num_agents, num_agents))
        self.base_adjacency.fill_diagonal_(0)  # No self-loops
    
    def compute_similarity_matrix(self):
        """Compute pairwise cosine similarity matrix."""
        S = torch.zeros(self.num_agents, self.num_agents, device=self.identity_embeddings.device)
        
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    sim = cosine_similarity(
                        self.identity_embeddings[i],
                        self.identity_embeddings[j]
                    )
                    S[i, j] = sim
        
        return S
    
    def get_weighted_adjacency(self):
        """Get similarity-weighted adjacency matrix."""
        S = self.compute_similarity_matrix()
        A = self.base_adjacency * S
        
        # Row-normalize
        row_sums = torch.sum(A, dim=1, keepdim=True) + 1e-8
        A = A / row_sums
        
        return A, S
    
    def forward(self, E_all: torch.Tensor):
        """
        Apply diffusion to all agents' IAEs.
        
        Args:
            E_all: Stacked IAEs (num_agents, iae_dim)
        
        Returns:
            Diffused IAEs (num_agents, iae_dim)
        """
        A, S = self.get_weighted_adjacency()
        L = compute_laplacian(A, normalized=True)
        
        # Diffusion: E ← E - α L E
        E_diffused = E_all - self.alpha * (L @ E_all)
        
        return E_diffused
    
    def compute_bias_penalty(self):
        """
        Compute bias suppression penalty: λ ||A ⊙ S||_F^2
        
        Returns:
            Scalar penalty
        """
        if self.lambda_bias == 0:
            return torch.tensor(0.0, device=self.identity_embeddings.device)
        
        A, S = self.get_weighted_adjacency()
        element_wise = A * S
        penalty = self.lambda_bias * torch.norm(element_wise, p='fro') ** 2
        
        return penalty