# Install dependencies
pip install -r requirements.txt

# Generate demonstration data
python scripts/generate_demos.py

# Verify imports
python -c "from esaiv3 import ESAIv3Agent; print('✅ Import successful')"

# Quick sanity test (should complete in ~30 seconds)
python train.py \
  --config configs/model/esaiv3_default.yaml \
  --env-config configs/envs/moral_temptation.yaml \
  --seed 0 \
  --exp-name sanity_test \
  --total_steps 1000

# Verify checkpoint creation
ls -la data/results/logs/moral_temptation/sanity_test/seed_0/