# ESAI: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accepted: ALA@AAMAS 2026](https://img.shields.io/badge/Accepted-ALA%40AAMAS%202026-green.svg)]()

Official implementation of ESAI for Moral Temptation experiments.

## Quick Links

- Reproducibility guide: this README
- Equation-to-code mapping: [IMPLEMENTATION.md](IMPLEMENTATION.md)

## Setup

```bash
git clone https://github.com/ezylopx5/Embedded-Safety-Aligned-Intelligence-ESAI.git
cd Embedded-Safety-Aligned-Intelligence-ESAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train (Single Seed)

All runs use `configs/envs/moral_temptation.yaml`.

### ESAI

```bash
python3 train.py \
  --env-config configs/envs/moral_temptation.yaml \
  --config configs/model/esaiv3_lambda5_paper.yaml \
  --exp-name esaiv3_lambda5_paper \
  --seed 1 \
  --total-steps 500000
```

### PPO Baseline

```bash
python3 train.py \
  --env-config configs/envs/moral_temptation.yaml \
  --config configs/model/ppo_baseline_paper.yaml \
  --exp-name ppo_baseline_paper \
  --seed 1 \
  --total-steps 500000
```

### CPO Baseline

```bash
python3 train.py \
  --env-config configs/envs/moral_temptation.yaml \
  --config configs/model/cpo_baseline.yaml \
  --exp-name cpo_baseline \
  --seed 1 \
  --total-steps 500000
```

## Train (5 Seeds)

### ESAI (5 seeds)

```bash
for seed in 1 2 3 4 5; do
  python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/esaiv3_lambda5_paper.yaml \
    --exp-name esaiv3_lambda5_paper \
    --seed "$seed" \
    --total-steps 500000
done
```

### PPO (5 seeds)

```bash
for seed in 1 2 3 4 5; do
  python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/ppo_baseline_paper.yaml \
    --exp-name ppo_baseline_paper \
    --seed "$seed" \
    --total-steps 500000
done
```

### CPO (5 seeds)

```bash
for seed in 1 2 3 4 5; do
  python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/cpo_baseline.yaml \
    --exp-name cpo_baseline \
    --seed "$seed" \
    --total-steps 500000
done
```

## Evaluate Saved Model

```bash
python3 evaluate.py \
  --load-dir results/moral_temptation/esaiv3_lambda5_paper/seed_1 \
  --env-config configs/envs/moral_temptation.yaml \
  --eval-episodes 100 \
  --seed 1
```

## Generate Figures

```bash
python3 scripts/generate_paper_figures.py \
  --results-dir results/moral_temptation \
  --output-dir results/figures
```

## Research Artifact Gallery

The following figures are sourced from `research_artifacts 22-28-03-755` and mirrored under `assets/paper_gallery/` for repository rendering.

| | | |
|---|---|---|
| ![Figure 1 Main Comparison](assets/paper_gallery/figure1_main_comparison.png) | ![Figure 1 Main V2](assets/paper_gallery/figure1_main_v2.png) | ![Prosocial Rate Comparison](assets/paper_gallery/prosocial_rate_comparison.png) |
| ![Training Curves](assets/paper_gallery/training_curves.png) | ![Reward Curves By Lambda](assets/paper_gallery/reward_curves_by_lambda.png) | ![Estimated PR By Lambda](assets/paper_gallery/estimated_pr_by_lambda.png) |
| ![Lambda Sensitivity](assets/paper_gallery/lambda_sensitivity.png) | ![Lambda Vs Reward](assets/paper_gallery/lambda_vs_reward.png) | ![Architecture Impact](assets/paper_gallery/architecture_impact.png) |
| ![Conceptual Diagram](assets/paper_gallery/conceptual_diagram.png) | ![Harm Help Breakdown](assets/paper_gallery/harm_help_breakdown.png) | ![Key Findings Summary](assets/paper_gallery/key_findings_summary.png) |
| ![Entropy Evolution](assets/paper_gallery/entropy_evolution.png) | ![IAE Norm Evolution](assets/paper_gallery/iae_norm_evolution.png) |  |

## Output Layout

Training outputs are written to:

`results/moral_temptation/<exp-name>/seed_<seed>/`

Key artifacts per run:

- `checkpoint_final.pt`
- `metrics.json`
- `config.json`

## Minimal Layout

```text
esaiv3/        core ESAI implementation
configs/       environment and model YAML configs
scripts/       run and plotting scripts
tools/         analysis utilities
train.py       training entrypoint
evaluate.py    evaluation entrypoint
```

## License

MIT License. See [LICENSE](LICENSE).