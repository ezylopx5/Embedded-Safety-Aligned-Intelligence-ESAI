#!/bin/bash
# =============================================================================
# ESAI-v3 Training Script for Lightning AI A100 GPU
# =============================================================================
# This script is optimized for NVIDIA A100 GPUs with:
# - TF32 precision (8x faster matmul)
# - Mixed precision training (AMP)
# - torch.compile() for graph optimization
# - Large batch sizes (512) and rollout lengths (8192)
# 
# Expected training time for 300k steps:
# - A100: ~15-30 minutes per run
# - Default (MPS/CPU): ~2-4 hours per run
# =============================================================================

set -e

# Configuration
LAMBDA_REG=${LAMBDA_REG:-2.0}
SEEDS=${SEEDS:-"1 2 3 4 5"}
TOTAL_STEPS=${TOTAL_STEPS:-300000}
EXP_NAME=${EXP_NAME:-"esaiv3_a100"}

echo "============================================================"
echo " ESAI-v3 A100 Training"
echo "============================================================"
echo " Lambda: $LAMBDA_REG"
echo " Seeds: $SEEDS"
echo " Steps: $TOTAL_STEPS"
echo " Experiment: $EXP_NAME"
echo "============================================================"

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Update lambda_reg in config
sed -i "s/lambda_reg: .*/lambda_reg: $LAMBDA_REG/" configs/model/esaiv3_a100.yaml

# Run training for each seed
for seed in $SEEDS; do
    echo ""
    echo ">>> Training seed $seed..."
    python train.py \
        --config configs/model/esaiv3_a100.yaml \
        --env-config configs/envs/moral_temptation.yaml \
        --exp-name "${EXP_NAME}_lambda${LAMBDA_REG}" \
        --seed $seed \
        --total-steps $TOTAL_STEPS \
        --compile \
        --device cuda
    
    echo ">>> Seed $seed complete!"
done

echo ""
echo "============================================================"
echo " All training runs complete!"
echo " Results saved to: results/moral_temptation/${EXP_NAME}_lambda${LAMBDA_REG}/"
echo "============================================================"

# Run PPO baseline for comparison
echo ""
echo ">>> Running PPO baseline for comparison..."
for seed in $SEEDS; do
    echo ">>> PPO seed $seed..."
    python train.py \
        --config configs/model/ppo_baseline.yaml \
        --env-config configs/envs/moral_temptation.yaml \
        --exp-name "ppo_a100_baseline" \
        --seed $seed \
        --total-steps $TOTAL_STEPS \
        --compile \
        --batch-size 512 \
        --rollout-length 8192 \
        --device cuda
done

echo ""
echo "============================================================"
echo " All experiments complete!"
echo "============================================================"
