#!/bin/bash
# =============================================================================
# ALA @ AAMAS 2026 Paper Experiments
# Run all experiments needed for publication figures and tables
# =============================================================================

set -e

echo "=============================================="
echo "ESAI-v3 Paper Experiments"
echo "ALA @ AAMAS 2026"
echo "=============================================="

# Configuration
TOTAL_STEPS=500000
EVAL_INTERVAL=25000
EVAL_EPISODES=20
NUM_SEEDS=5
ENV_CONFIG="configs/envs/moral_temptation.yaml"
RESULTS_DIR="results/moral_temptation"

mkdir -p $RESULTS_DIR

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU detected: $GPU_INFO"
else
    echo "No NVIDIA GPU detected, using CPU/MPS"
fi

echo ""
echo "Experiment Parameters:"
echo "  Total steps: $TOTAL_STEPS"
echo "  Seeds: $NUM_SEEDS"
echo "  Eval interval: $EVAL_INTERVAL"
echo ""

# =============================================================================
# EXPERIMENT 1: PPO Baseline (5 seeds)
# =============================================================================
echo "=============================================="
echo "Experiment 1: PPO Baseline"
echo "=============================================="

for SEED in $(seq 1 $NUM_SEEDS); do
    EXP_NAME="ppo_baseline_paper/seed_$SEED"
    echo ""
    echo ">>> PPO Baseline - Seed $SEED"
    
    if [ -f "$RESULTS_DIR/$EXP_NAME/checkpoint_final.pt" ]; then
        echo "Seed $SEED already complete, skipping..."
        continue
    fi
    
    python train.py \
        --config configs/model/ppo_baseline_paper.yaml \
        --env-config $ENV_CONFIG \
        --exp-name $EXP_NAME \
        --total-steps $TOTAL_STEPS \
        --eval-interval $EVAL_INTERVAL \
        --eval-episodes $EVAL_EPISODES \
        --seed $SEED \
        --log-dir $RESULTS_DIR
        
    echo ">>> PPO Baseline Seed $SEED complete"
done

# =============================================================================
# EXPERIMENT 2: ESAI-v3 λ=5 (5 seeds)
# =============================================================================
echo ""
echo "=============================================="
echo "Experiment 2: ESAI-v3 λ=5"
echo "=============================================="

for SEED in $(seq 1 $NUM_SEEDS); do
    EXP_NAME="esaiv3_lambda5_paper/seed_$SEED"
    echo ""
    echo ">>> ESAI-v3 λ=5 - Seed $SEED"
    
    if [ -f "$RESULTS_DIR/$EXP_NAME/checkpoint_final.pt" ]; then
        echo "Seed $SEED already complete, skipping..."
        continue
    fi
    
    python train.py \
        --config configs/model/esaiv3_lambda5_paper.yaml \
        --env-config $ENV_CONFIG \
        --exp-name $EXP_NAME \
        --total-steps $TOTAL_STEPS \
        --eval-interval $EVAL_INTERVAL \
        --eval-episodes $EVAL_EPISODES \
        --seed $SEED \
        --log-dir $RESULTS_DIR
        
    echo ">>> ESAI-v3 λ=5 Seed $SEED complete"
done

# =============================================================================
# Generate Figures
# =============================================================================
echo ""
echo "=============================================="
echo "Generating Paper Figures"
echo "=============================================="

python scripts/generate_paper_figures.py \
    --results-dir $RESULTS_DIR \
    --output-dir paper/figures

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Figures saved to: paper/figures/"
echo ""
echo "Generated files:"
echo "  - fig1_learning_curves.pdf"
echo "  - fig2_prosocial_ratio.pdf"  
echo "  - fig3_alignment_regret.pdf"
echo "  - fig4_bar_comparison.pdf"
echo "  - table_results.tex"
