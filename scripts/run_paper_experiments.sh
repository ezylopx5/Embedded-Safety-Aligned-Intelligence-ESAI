#!/bin/bash
# =============================================================================
# Priority Experiments for ALA @ AAMAS 2026 Paper
# =============================================================================
# 
# Required experiments:
# 1. PPO Baseline (5 seeds) - Control
# 2. ESAI-v3 λ=5 (5 seeds) - Main experiment
#
# Each run: 500k steps for proper convergence
# =============================================================================

set -e

# Configuration
TOTAL_STEPS=500000
EVAL_INTERVAL=25000
SAVE_INTERVAL=50000
SEEDS="1 2 3 4 5"

echo "=============================================="
echo "ESAI-v3 Priority Experiments"
echo "=============================================="
echo "Total steps per run: $TOTAL_STEPS"
echo "Seeds: $SEEDS"
echo ""

# Activate virtual environment
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Experiment 1: PPO Baseline
# -----------------------------------------------------------------------------
echo "=============================================="
echo "EXPERIMENT 1: PPO Baseline"
echo "=============================================="

for SEED in $SEEDS; do
    echo ""
    echo ">>> PPO Baseline - Seed $SEED"
    echo "----------------------------------------------"
    
    python train.py \
        --config configs/model/ppo_baseline.yaml \
        --env-config configs/envs/moral_temptation.yaml \
        --exp-name ppo_baseline_paper \
        --total-steps $TOTAL_STEPS \
        --eval-interval $EVAL_INTERVAL \
        --save-interval $SAVE_INTERVAL \
        --seed $SEED
        
    echo ">>> PPO Baseline Seed $SEED complete"
done

# -----------------------------------------------------------------------------
# Experiment 2: ESAI-v3 λ=5
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "EXPERIMENT 2: ESAI-v3 λ=5"
echo "=============================================="

for SEED in $SEEDS; do
    echo ""
    echo ">>> ESAI-v3 λ=5 - Seed $SEED"
    echo "----------------------------------------------"
    
    python train.py \
        --config configs/model/esaiv3_lambda5.yaml \
        --env-config configs/envs/moral_temptation.yaml \
        --exp-name esaiv3_lambda5_paper \
        --total-steps $TOTAL_STEPS \
        --eval-interval $EVAL_INTERVAL \
        --save-interval $SAVE_INTERVAL \
        --seed $SEED
        
    echo ">>> ESAI-v3 λ=5 Seed $SEED complete"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/moral_temptation/ppo_baseline_paper/"
echo "  - results/moral_temptation/esaiv3_lambda5_paper/"
echo ""
echo "To analyze results, run:"
echo "  python evaluate.py --checkpoint results/moral_temptation/ppo_baseline_paper/seed_1/checkpoint_final.pt"
echo "  python evaluate.py --checkpoint results/moral_temptation/esaiv3_lambda5_paper/seed_1/checkpoint_final.pt"
