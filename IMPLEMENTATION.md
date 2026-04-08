# ESAI Implementation Details

Primary reproduction guide: [README.md](README.md)

**Paper**: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning  
**Authors**: Harsh Rathva, Pruthwik Mishra — SVNIT, Surat, India  
**Venue**: ALA @ AAMAS 2026

---

## Paper Equation → Code Mapping

| Paper | Equation | Code Location | Function/Class |
|-------|----------|---------------|----------------|
| Eq. 1 | IAE dynamics: $E_{t+1} = \gamma_E E_t + g_\phi(z,a,h)$ | `esaiv3/model.py` L353–450 | `ESAIv3Agent.update_iae()` |
| Eq. 2 | Counterfactual forecast: $\hat{E}(a) = h_\psi(s_t, a, \text{read}(H))$ | `esaiv3/model.py` L452–520 | `ESAIv3Agent.forecast_counterfactual()` |
| Eq. 3 | Softmin reference: $\pi_{ref}(a) \propto \exp(-R(a)/\tau)$ | `esaiv3/loss.py` L50–77 | `AlignmentLoss.compute_softmin_reference()` |
| Eq. 4 | Attention gating: $\tilde{o} = \sigma(W_E E) \odot o$ | `esaiv3/model.py` L13–57 | `AttentionGating.forward()` |
| Eq. 5 | Hebbian update: $H_{t+1} = (1-\delta)H_t + \eta(E \otimes z)$ | `esaiv3/model.py` L76–95 | `HebbianMemory.update()` |
| Eq. 6 | Harm scalarization: $R(a) = \|\hat{E}(a)\|^2$ | `esaiv3/loss.py` L35–48 | `AlignmentLoss.compute_harm_values()` |
| Eq. 7–8 | Softmin reference distribution | `esaiv3/loss.py` L50–77 | `AlignmentLoss.compute_softmin_reference()` |
| Eq. 9 | Alignment regret: $AR_t = \|E_{t+1} - E^{ref}\|^2$ | `esaiv3/loss.py` L79–142 | `AlignmentLoss.forward()` |
| Eq. 10 | Similarity weights: $S_{ij} = \cos(E_i, E_j)$ | `esaiv3/model.py` L141–155 | `GraphDiffusion.compute_similarity_matrix()` |
| Eq. 11 | Bias regularizer | `esaiv3/model.py` L193–197 | `GraphDiffusion.compute_bias_penalty()` |
| Eq. 12 | Graph diffusion: $E' = E - \alpha L E$ | `esaiv3/model.py` L175–191 | `GraphDiffusion.diffuse()` |
| Eq. 13 | Reward transform: $r' = r^{ext} - \lambda \cdot AR$ | `train.py` L600–650 | `transform_rewards()` |
| Eq. 14 | PPO objective | `esaiv3/loss.py` L189–235 | `PPOLoss.forward()` |

---

## Architecture → Class Mapping

```
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
│ (Eq. 3,6–9)   │     │   (Eq. 14)      │
│ esaiv3/loss.py │     │ esaiv3/loss.py  │
└────────────────┘     └─────────────────┘
```

---

## Reproducing Paper Results

### Prerequisites
```bash
git clone https://github.com/ezylopx5/Embedded-Safety-Aligned-Intelligence-ESAI.git
cd Embedded-Safety-Aligned-Intelligence-ESAI
pip install -r requirements.txt
```

### Table 2 — Complete Lambda Sweep
```bash
# PPO baseline (single seed)
python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/ppo_baseline_paper.yaml \
    --exp-name ppo_baseline_paper \
    --seed 1 \
    --total-steps 500000

# CPO baseline (single seed)
python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/cpo_baseline.yaml \
    --exp-name cpo_baseline \
    --seed 1 \
    --total-steps 500000

# ESAI (single seed)
python3 train.py \
    --env-config configs/envs/moral_temptation.yaml \
    --config configs/model/esaiv3_lambda5_paper.yaml \
    --exp-name esaiv3_lambda5_paper \
    --seed 1 \
    --total-steps 500000
```

### Full Paper Experiments (All-in-One)
```bash
bash scripts/run_paper_experiments.sh
```

### Generate Paper Figures
```bash
python3 scripts/generate_paper_figures.py \
    --results-dir results/moral_temptation \
    --output-dir results/figures
```

---

## Hyperparameters (Paper Table 1)

| Parameter | Value | Config Key |
|-----------|-------|------------|
| IAE dimension $k$ | 32 | `iae_dim` |
| Hidden dimension | 128 | `hidden_dim` |
| Learning rate | 3e-4 | `lr` |
| $\gamma$ (discount) | 0.99 | `gamma` |
| $\gamma_E$ (IAE decay) | 0.9 | `gamma_E` |
| $\lambda_{GAE}$ | 0.95 | `gae_lambda` |
| Entropy coefficient | 0.01 → 0.001 | `entropy_coef_start/end` |
| PPO clip $\epsilon$ | 0.2 | `clip_epsilon` |
| Lambda warmup | 10,000 steps | `lambda_warmup_steps` |
| Softmin temperature $\tau$ | 1.0 → 0.1 | `temperature_start/end` |
| Diffusion rate $\alpha$ | 0.05 | `alpha_diffusion` |
| Hebbian learning rate $\eta$ | 1e-3 | `eta` |
| Hebbian decay $\delta$ | 0.02 | `delta` |
| Memory dimension | 32 | `memory_dim` |

---

## File Structure

```
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
│   ├── model/               # 13 YAML configs (ESAI, PPO, CPO, ablations)
│   └── envs/                # 5 environment configs
├── scripts/                 # 18 experiment & plotting scripts
├── train.py                 # Main training script (1727 lines)
├── evaluate.py              # Evaluation script
├── requirements.txt         # Dependencies
├── setup.py                 # Package installation
└── Embedded_..._camera_ready/  # Camera-ready LaTeX source
```

---

## Known Limitations

As documented in the paper (Section 6.3):
- Results from 1–2 seeds per condition
- Single simplified environment (Moral Temptation)
- No comparison to simple reward shaping baseline
- No systematic component ablation (configs exist but not validated)
- Multi-agent components (graph diffusion) not exercised in current evaluation
