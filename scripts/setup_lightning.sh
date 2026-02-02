#!/bin/bash
# =============================================================================
# Lightning.ai H100 Setup Script for ESAI-v3 Experiments
# =============================================================================

set -e

echo "=============================================="
echo "Setting up ESAI-v3 on Lightning.ai H100"
echo "=============================================="

# Clone from GitHub
echo -e "\n[1/5] Cloning repository..."
cd ~
if [ -d "MARL" ]; then
    echo "MARL directory exists, pulling latest..."
    cd MARL && git pull
else
    git clone https://github.com/ezylopx5/Embedded-Safety-Aligned-Intelligence-ESAI.git MARL
    cd MARL
fi

# Create virtual environment
echo -e "\n[2/5] Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install dependencies - optimized for H100 (CUDA 12.1 + cuDNN 8.9)
echo -e "\n[3/5] Installing dependencies (H100 optimized)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# Enable H100 optimizations
echo -e "\n[4/5] Configuring H100 optimizations..."
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 compute capability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Verify GPU
echo -e "\n[5/5] Verifying H100 GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    # Test TF32 (H100 feature)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f'TF32 enabled: ✓')
"

echo -e "\n=============================================="
echo "H100 Setup complete! Ready for experiments."
echo "=============================================="
echo ""
echo "To run experiments:"
echo "  cd ~/MARL && source .venv/bin/activate"
echo "  python train.py --config configs/model/esaiv3_lambda5.yaml --env-config configs/envs/moral_temptation.yaml --exp-name esaiv3_h100 --total-steps 500000 --seed 1"
