#!/usr/bin/env bash
set -euo pipefail

ENVS=("moral_temptation" "social_distress" "mpe" "overcooked" "ssd")

for ENV in "${ENVS[@]}"; do
  for SEED in {0..9}; do
    python train.py \
      --config configs/model/esaiv3_default.yaml \
      --env-config configs/envs/${ENV}.yaml \
      --seed ${SEED} \
      --exp-name corr_${ENV}_bias0 \
      --lambda_bias 0.0 \
      --total_steps 1000000
    
    python evaluate.py \
      --load-dir data/results/logs/${ENV}/corr_${ENV}_bias0/seed_${SEED}/ \
      --env-config configs/envs/${ENV}.yaml \
      --eval-episodes 100 \
      --exp-tag baseline
  done
done

echo "[run_correlation] Complete. Run: python tools/analyze_correlation.py"