#!/bin/bash
# =============================================================================
# Lightning.ai A100 Setup Script for ESAI-v3 Experiments
# =============================================================================

set -e

echo "=============================================="
echo "Setting up ESAI-v3 on Lightning.ai A100"
echo "=============================================="

cd ~/MARL

# Create virtual environment
echo -e "\n[1/4] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install dependencies
echo -e "\n[2/4] Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# Verify GPU
echo -e "\n[3/4] Verifying GPU..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Quick test
echo -e "\n[4/4] Running quick test..."
python -c "
from esaiv3.model import ESAIv3Agent
from esaiv3.env_wrappers import MoralTemptationEnv
import torch

env = MoralTemptationEnv()
agent = ESAIv3Agent(obs_dim=16, action_dim=6, iae_dim=32, use_attention=True, use_hebbian=True, use_alignment_regret=True)

if torch.cuda.is_available():
    agent = agent.cuda()
    print('Agent moved to GPU ✓')

obs, _ = env.reset()
obs_t = torch.tensor(obs, dtype=torch.float32)
if torch.cuda.is_available():
    obs_t = obs_t.cuda()
action, extra = agent.act(obs_t)
print(f'Test action: {action}, AR: {extra[\"AR_t\"]:.4f} ✓')
"

echo -e "\n=============================================="
echo "Setup complete! Ready for experiments."
echo "=============================================="
echo ""
echo "To run experiments:"
echo "  cd ~/MARL && source .venv/bin/activate"
echo "  ./scripts/run_paper_experiments.sh"
