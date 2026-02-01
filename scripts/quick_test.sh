#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "Quick Test Run (5 minutes)"
echo "========================================="
echo ""
echo "Testing basic functionality with short runs..."
echo ""

# Test on Moral Temptation with minimal steps
python train.py \
  --config configs/model/esaiv3_default.yaml \
  --env-config configs/envs/moral_temptation.yaml \
  --seed 0 \
  --exp-name quick_test \
  --total_steps 1000

# Test evaluation
python evaluate.py \
  --load-dir data/results/logs/moral_temptation/quick_test/seed_0/ \
  --env-config configs/envs/moral_temptation.yaml \
  --eval-episodes 5

# Test intervention
python evaluate.py \
  --load-dir data/results/logs/moral_temptation/quick_test/seed_0/ \
  --env-config configs/envs/moral_temptation.yaml \
  --eval-episodes 5 \
  --intervene low \
  --clamp_low 0.1 \
  --intervene_steps 10 \
  --exp-tag test_low

echo ""
echo "========================================="
echo "Quick test complete!"
echo "If no errors, the setup is working correctly."
echo "========================================="