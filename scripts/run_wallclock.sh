#!/usr/bin/env bash
set -euo pipefail

ENVS=("moral_temptation" "overcooked" "mpe")

for ENV in "${ENVS[@]}"; do
  for SEED in {0..4}; do
    # ESAI-v3
    python train.py \
      --config configs/model/esaiv3_default.yaml \
      --env-config configs/envs/${ENV}.yaml \
      --seed ${SEED} \
      --exp-name wallclock_esai_v3 \
      --total_steps 300000
    
    # PPO baseline
    python train.py \
      --config configs/model/ppo_baseline.yaml \
      --env-config configs/envs/${ENV}.yaml \
      --seed ${SEED} \
      --exp-name wallclock_ppo \
      --total_steps 300000
  done
done

echo "[run_wallclock] Complete. Run: python tools/analyze_wallclock.py"