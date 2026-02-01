#!/usr/bin/env bash
set -euo pipefail

ENVS=("moral_temptation" "social_distress" "mpe")

for ENV in "${ENVS[@]}"; do
  for SEED in {0..9}; do
    BASE=data/results/logs/${ENV}/corr_${ENV}_bias0/seed_${SEED}/
    
    # Baseline (no intervention)
    python evaluate.py \
      --load-dir ${BASE} \
      --env-config configs/envs/${ENV}.yaml \
      --eval-episodes 100 \
      --intervene none \
      --exp-tag baseline
    
    # Low clamp (suppression)
    python evaluate.py \
      --load-dir ${BASE} \
      --env-config configs/envs/${ENV}.yaml \
      --eval-episodes 100 \
      --intervene low \
      --clamp_low 0.1 \
      --intervene_steps 50 \
      --exp-tag low
    
    # High clamp (amplification)
    python evaluate.py \
      --load-dir ${BASE} \
      --env-config configs/envs/${ENV}.yaml \
      --eval-episodes 100 \
      --intervene high \
      --clamp_high 2.0 \
      --intervene_steps 50 \
      --exp-tag high
  done
done

echo "[run_intervention] Complete. Run: python tools/analyze_intervention.py"