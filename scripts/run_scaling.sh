#!/usr/bin/env bash
set -euo pipefail

ENVS=("social_distress" "mpe" "ssd")

for ENV in "${ENVS[@]}"; do
  for SEED in {0..9}; do
    # Train at N=4
    python train.py \
      --config configs/model/esaiv3_default.yaml \
      --env-config configs/envs/${ENV}.yaml \
      --seed ${SEED} \
      --exp-name scale_${ENV}_N4 \
      --override num_agents=4 \
      --total_steps 1000000
    
    # Eval at N=4
    python evaluate.py \
      --load-dir data/results/logs/${ENV}/scale_${ENV}_N4/seed_${SEED}/ \
      --env-config configs/envs/${ENV}.yaml \
      --eval-episodes 100 \
      --exp-tag evalN4
    
    # Eval at N=16 (zero-shot)
    python evaluate.py \
      --load-dir data/results/logs/${ENV}/scale_${ENV}_N4/seed_${SEED}/ \
      --env-config configs/envs/${ENV}.yaml \
      --override num_agents=16 \
      --eval-episodes 100 \
      --exp-tag evalN16
  done
done

echo "[run_scaling] Complete. Run: python tools/analyze_scaling.py"