#!/usr/bin/env bash
set -euo pipefail

LAMBDAS=("0.0" "0.001" "0.01" "0.1")
ENVS=("social_distress" "mpe")

for ENV in "${ENVS[@]}"; do
  for L in "${LAMBDAS[@]}"; do
    for SEED in {0..9}; do
      python train.py \
        --config configs/model/esaiv3_default.yaml \
        --env-config configs/envs/${ENV}.yaml \
        --seed ${SEED} \
        --exp-name bias_${ENV}_L${L} \
        --lambda_bias ${L} \
        --total_steps 1000000
      
      python evaluate.py \
        --load-dir data/results/logs/${ENV}/bias_${ENV}_L${L}/seed_${SEED}/ \
        --env-config configs/envs/${ENV}.yaml \
        --eval-episodes 100
    done
  done
done

echo "[run_bias_sweep] Complete. Run: python tools/analyze_bias.py"