"""
ESAI-v3: Learning Internal Alignment Embeddings for Multi-Agent Coordination

Core package for differentiable affective alignment in MARL.
"""

__version__ = "1.0.0"
__author__ = "Harsh Rathva, Pruthwik Mishra"

from .model import ESAIv3Agent, GraphDiffusion  # GraphDiffusion is in model.py now (BUG-006 fix)
from .loss import AlignmentLoss, PPOLoss
from .memory import HebbianMemory
from .logging_utils import EvalLogger, ensure_dir, save_json
from .utils import set_seed, compute_gae, anneal_parameter

__all__ = [
    "ESAIv3Agent",
    "AlignmentLoss",
    "PPOLoss",
    "HebbianMemory",
    "GraphDiffusion",
    "EvalLogger",
    "ensure_dir",
    "save_json",
    "set_seed",
    "compute_gae",
    "anneal_parameter",
]