#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "Running ESAI-v3 Full Experimental Suite"
echo "========================================="
echo ""
echo "Estimated time: 40-60 GPU-hours on NVIDIA T4"
echo "Required storage: ~15-20 GB"
echo ""
echo "WARNING: This will run ALL experiments!"
echo "Press Ctrl+C within 10 seconds to cancel..."
echo ""

for i in {10..1}; do
    echo -ne "\rStarting in $i seconds... "
    sleep 1
done
echo ""

# Track start time
START_TIME=$(date +%s)

# Create results directory
mkdir -p data/results/logs

echo ""
echo "========================================="
echo "[1/6] Running Correlation Experiments"
echo "========================================="
bash scripts/run_correlation.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Correlation experiments failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "[2/6] Running Intervention Experiments"
echo "========================================="
bash scripts/run_intervention.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Intervention experiments failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "[3/6] Running Ablation Experiments"
echo "========================================="
bash scripts/run_ablations.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Ablation experiments failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "[4/6] Running Scaling Experiments"
echo "========================================="
bash scripts/run_scaling.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Scaling experiments failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "[5/6] Running Bias Sweep Experiments"
echo "========================================="
bash scripts/run_bias_sweep.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Bias sweep experiments failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "[6/6] Running Wall-Clock Experiments"
echo "========================================="
bash scripts/run_wallclock.sh
if [ $? -ne 0 ]; then
    echo "ERROR: Wall-clock experiments failed!"
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Now run analysis scripts:"
echo "  python tools/analyze_correlation.py"
echo "  python tools/analyze_intervention.py"
echo "  python tools/analyze_ablations.py"
echo "  python tools/analyze_scaling.py"
echo "  python tools/analyze_bias.py"
echo "  python tools/analyze_wallclock.py"
echo ""
echo "Or run all analyses at once:"
echo "  bash scripts/run_all_analysis.sh"
echo ""
echo "========================================="