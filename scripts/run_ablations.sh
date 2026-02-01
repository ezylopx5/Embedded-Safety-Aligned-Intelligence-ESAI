#!/usr/bin/env bash
set -euo pipefail

declare -A FLAGS
FLAGS[full]="--use_alignment_regret true --attention true --heb_read_to_forecast true --diffusion true"
FLAGS[no_regret]="--use_alignment_regret false --attention true --heb_read_to_forecast true --diffusion true"
FLAGS[no_attention]="--use_alignment_regret true --attention false --heb_read_to_forecast true --diffusion true"
FLAGS[no_hebbian]="--use_alignment_regret true --attention true --heb_read_to_forecast false --diffusion true"
FLAGS[no_diffusion]="--use_alignment_regret true --attention true --heb_read_to_forecast true --diffusion false"

ENVS=("moral_temptation" "overcooked")

for ENV in "${ENVS[@]}"; do
  for VAR in full no_regret no_attention no_hebbian no_diffusion; do
    for SEED in {0..9}; do
      python train.py \
        --config configs/model/esaiv3_default.yaml \
        --env-config configs/envs/${ENV}.yaml \
        --seed ${SEED} \
        --exp-name ablate_${ENV}_${VAR} \
        ${FLAGS[$VAR]} \
        --total_steps 1000000
      
      python evaluate.py \
        --load-dir data/results/logs/${ENV}/ablate_${ENV}_${VAR}/seed_${SEED}/ \
        --env-config configs/envs/${ENV}.yaml \
        --eval-episodes 100
    done
  done
done

echo "[run_ablations] Complete. Run: python tools/analyze_ablations.py"