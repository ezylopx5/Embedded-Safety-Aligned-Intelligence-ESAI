#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "Running ESAI-v3 Full Experimental Suite"
echo "========================================="
echo ""
echo "WARNING: This will take 40-60 GPU-hours on T4"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

bash scripts/run_correlation.sh
bash scripts/run_intervention.sh
bash scripts/run_ablations.sh
bash scripts/run_scaling.sh
bash scripts/run_bias_sweep.sh
bash scripts/run_wallclock.sh

echo ""
echo "========================================="
echo "All experiments complete!"
echo "Run analysis scripts in tools/"
echo "========================================="