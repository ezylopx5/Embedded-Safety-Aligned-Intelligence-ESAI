# ESAI: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accepted: ALA@AAMAS 2026](https://img.shields.io/badge/Accepted-ALA%40AAMAS%202026-green.svg)]()

Official implementation of ESAI for Moral Temptation experiments.

## Quick Links

- Reproducibility guide: this README

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

### Additional Gallery: `research_artifacts`

| | | |
|---|---|---|
| ![Learning Curve PR](assets/additional_gallery/research_artifacts/learning_curve_pr.png)<br><sub>Prosocial rate learning curve over training.</sub> | ![Learning Curve Reward](assets/additional_gallery/research_artifacts/learning_curve_reward.png)<br><sub>External reward learning curve over training.</sub> | ![Lambda Sensitivity PR](assets/additional_gallery/research_artifacts/lambda_sensitivity_pr.png)<br><sub>Prosocial performance sensitivity across lambda values.</sub> |
| ![Lambda Sensitivity Reward](assets/additional_gallery/research_artifacts/lambda_sensitivity_reward.png)<br><sub>Reward sensitivity across lambda values.</sub> | ![Pareto Frontier](assets/additional_gallery/research_artifacts/pareto_frontier.png)<br><sub>Trade-off frontier between alignment and reward objectives.</sub> |  |

### Additional Gallery: `Everythingyouneed`

| | | |
|---|---|---|
| ![Figure 3 Lambda Sweep](assets/additional_gallery/everythingyouneed/Figure3_LambdaSweep_Paper.png)<br><sub>Paper Figure 3: full lambda sweep summary.</sub> | ![Figure 4 Training Curves](assets/additional_gallery/everythingyouneed/Figure4_TrainingCurves_Paper.png)<br><sub>Paper Figure 4: training dynamics comparison.</sub> | ![Figure 5 Pareto Frontier](assets/additional_gallery/everythingyouneed/Figure5_ParetoFrontier_Paper.png)<br><sub>Paper Figure 5: Pareto frontier of safety vs utility.</sub> |
| ![Figure 6 Mechanism Comparison](assets/additional_gallery/everythingyouneed/Figure6_MechanismComparison_Paper.png)<br><sub>Paper Figure 6: mechanism-level comparison.</sub> | ![Complete Lambda Sweep Analysis](assets/additional_gallery/everythingyouneed/complete_lambda_sweep_analysis.png)<br><sub>Detailed analysis of complete lambda sweep behavior.</sub> | ![CPO Detailed Analysis](assets/additional_gallery/everythingyouneed/cpo_detailed_analysis.png)<br><sub>CPO baseline diagnostics and behavioral profile.</sub> |
| ![ESAI Analysis Plots](assets/additional_gallery/everythingyouneed/esai_analysis_plots.png)<br><sub>Composite ESAI diagnostics and trends.</sub> | ![Extracted Subsection E](assets/additional_gallery/everythingyouneed/extracted_subsection_E.png)<br><sub>Supporting subsection figure extracted for paper alignment.</sub> | ![Full ESAI Lambda Sweep](assets/additional_gallery/everythingyouneed/full_esai_lambda_sweep.png)<br><sub>Full ESAI-only lambda sweep results.</sub> |
| ![Lambda 0.85 Detailed](assets/additional_gallery/everythingyouneed/lambda_085_detailed.png)<br><sub>Fine-grained diagnostics at lambda = 0.85.</sub> |  |  |

### Randomization Quality Check

| |
|---|
| ![Randomization Check Moral Temptation](assets/additional_gallery/quality_checks/randomization_check_moral_temptation.png)<br><sub>Randomization sanity check for Moral Temptation environment setup.</sub> |

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

---

## Implementation Details (Merged)

**Paper**: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning  
**Authors**: Harsh Rathva, Pruthwik Mishra — SVNIT, Surat, India  
**Venue**: ALA @ AAMAS 2026

### Paper Equation -> Code Mapping

| Paper | Equation | Code Location | Function/Class |
|-------|----------|---------------|----------------|
| Eq. 1 | IAE dynamics: $E_{t+1} = \gamma_E E_t + g_\phi(z,a,h)$ | `esaiv3/model.py` L353-450 | `ESAIv3Agent.update_iae()` |
| Eq. 2 | Counterfactual forecast: $\hat{E}(a) = h_\psi(s_t, a, \text{read}(H))$ | `esaiv3/model.py` L452-520 | `ESAIv3Agent.forecast_counterfactual()` |
| Eq. 3 | Softmin reference: $\pi_{ref}(a) \propto \exp(-R(a)/\tau)$ | `esaiv3/loss.py` L50-77 | `AlignmentLoss.compute_softmin_reference()` |
| Eq. 4 | Attention gating: $\tilde{o} = \sigma(W_E E) \odot o$ | `esaiv3/model.py` L13-57 | `AttentionGating.forward()` |
| Eq. 5 | Hebbian update: $H_{t+1} = (1-\delta)H_t + \eta(E \otimes z)$ | `esaiv3/model.py` L76-95 | `HebbianMemory.update()` |
| Eq. 6 | Harm scalarization: $R(a) = \|\hat{E}(a)\|^2$ | `esaiv3/loss.py` L35-48 | `AlignmentLoss.compute_harm_values()` |
| Eq. 7-8 | Softmin reference distribution | `esaiv3/loss.py` L50-77 | `AlignmentLoss.compute_softmin_reference()` |
| Eq. 9 | Alignment regret: $AR_t = \|E_{t+1} - E^{ref}\|^2$ | `esaiv3/loss.py` L79-142 | `AlignmentLoss.forward()` |
| Eq. 10 | Similarity weights: $S_{ij} = \cos(E_i, E_j)$ | `esaiv3/model.py` L141-155 | `GraphDiffusion.compute_similarity_matrix()` |
| Eq. 11 | Bias regularizer | `esaiv3/model.py` L193-197 | `GraphDiffusion.compute_bias_penalty()` |
| Eq. 12 | Graph diffusion: $E' = E - \alpha L E$ | `esaiv3/model.py` L175-191 | `GraphDiffusion.diffuse()` |
| Eq. 13 | Reward transform: $r' = r^{ext} - \lambda \cdot AR$ | `train.py` L600-650 | `transform_rewards()` |
| Eq. 14 | PPO objective | `esaiv3/loss.py` L189-235 | `PPOLoss.forward()` |

### Architecture -> Class Mapping

```text
┌─────────────────────────────────────────────────┐
│                  ESAIv3Agent                     │
│                (esaiv3/model.py)                 │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ AttentionGating│  │ HebbianMemory            │ │
│  │ (Eq. 4)       │  │ (Eq. 5)                  │ │
│  └──────────────┘  └──────────────────────────┘ │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ GraphDiffusion│  │ Forecaster Network       │ │
│  │ (Eq. 10–12)  │  │ (Eq. 2)                  │ │
│  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌────────────────┐     ┌─────────────────┐
│  AlignmentLoss │     │    PPOLoss      │
│ (Eq. 3,6-9)    │     │   (Eq. 14)      │
│ esaiv3/loss.py │     │ esaiv3/loss.py  │
└────────────────┘     └─────────────────┘
```

### Hyperparameters (Paper Table 1)

| Parameter | Value | Config Key |
|-----------|-------|------------|
| IAE dimension $k$ | 32 | `iae_dim` |
| Hidden dimension | 128 | `hidden_dim` |
| Learning rate | 3e-4 | `lr` |
| $\gamma$ (discount) | 0.99 | `gamma` |
| $\gamma_E$ (IAE decay) | 0.9 | `gamma_E` |
| $\lambda_{GAE}$ | 0.95 | `gae_lambda` |
| Entropy coefficient | 0.01 -> 0.001 | `entropy_coef_start/end` |
| PPO clip $\epsilon$ | 0.2 | `clip_epsilon` |
| Lambda warmup | 10,000 steps | `lambda_warmup_steps` |
| Softmin temperature $\tau$ | 1.0 -> 0.1 | `temperature_start/end` |
| Diffusion rate $\alpha$ | 0.05 | `alpha_diffusion` |
| Hebbian learning rate $\eta$ | 1e-3 | `eta` |
| Hebbian decay $\delta$ | 0.02 | `delta` |
| Memory dimension | 32 | `memory_dim` |

### File Structure

```text
├── esaiv3/                  # Core ESAI package
│   ├── __init__.py          # Package exports
│   ├── model.py             # ESAIv3Agent, AttentionGating, HebbianMemory, GraphDiffusion
│   ├── loss.py              # AlignmentLoss, PPOLoss, ForecastLoss
│   ├── memory.py            # HebbianMemory (standalone)
│   ├── env_wrappers.py      # MoralTemptation environment wrapper
│   ├── utils.py             # GAE, seed, scheduling utilities
│   ├── logging_utils.py     # Eval logging
│   └── visualization.py     # Training visualization tools
├── configs/
│   ├── model/               # YAML configs (ESAI, PPO, CPO, ablations)
│   └── envs/                # Environment configs
├── scripts/                 # Experiment and plotting scripts
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
└── setup.py                 # Package installation
```

### Known Limitations

- Results from limited seeds per condition
- Single simplified environment (Moral Temptation)
- No comparison to simple reward shaping baseline
- No full systematic component ablation
- Multi-agent components (graph diffusion) not exercised in all evaluations