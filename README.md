# ESAI: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accepted: ALA@AAMAS 2026](https://img.shields.io/badge/Accepted-ALA%40AAMAS%202026-green.svg)]()

Official implementation of **ESAI** from the paper:

> **Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning**  
> Harsh Rathva, Pruthwik Mishra  
> Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat, India  
> ALA @ AAMAS 2026 — Paphos, Cyprus

---

## 🎯 Overview

ESAI introduces **Internal Alignment Embeddings (IAE)**: differentiable latent regulators that predict and attenuate externalized harm during multi-agent coordination. Unlike external reward shaping or post-hoc safety constraints, IAE are learned latent variables that modulate policy gradients toward harm reduction via attention gating and graph diffusion.

### Key Results (Proof-of-Concept)
- **100% prosocial rate** on Moral Temptation environment (vs 6.5% PPO, 0% CPO)
- **Phase transition** at λ* ≈ 0.08 — below: selfish behavior, above: prosocial alignment
- **Saturation** at λ ≥ 0.085 — no additional benefit from higher regularization
- **IAE learns harm representations**: AR(STEAL) = 3.63 vs AR(HELP) ≈ 0

### Core Mechanisms
- **Counterfactual supervision** via softmin reference distributions
- **Graph diffusion** with similarity-weighted propagation
- **IAE-weighted attention** for perceptual salience modulation
- **Hebbian affect-memory** for temporal credit assignment

---

## ⚠️ Current Status

**This is a theoretical framework with proof-of-concept demonstration.**

### ✅ What's Been Demonstrated
- Framework is mathematically well-defined and implementable
- Phase transition phenomenon observed at λ* ≈ 0.08
- Saturation beyond λ ≥ 0.085 (all values yield identical results)
- IAE learns distinct representations for harmful vs. prosocial actions
- Full IAE architecture outperforms CPO (penalty-only) in our simplified environment

### ❌ Known Limitations (Acknowledged in Paper)
- **No comparison to simple reward shaping** (`r' = r_ext - λ·h_t`)
- **No systematic ablation studies** (component necessity not validated)
- **Single-seed results** (1-2 seeds per condition, no error bars)
- **Single simplified environment** (binary choice, no temporal complexity)
- **Multi-agent components untested** (graph diffusion, bias mitigation)
- **Missing baselines**: inequity aversion, intrinsic motivation approaches

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ezylopx5/Embedded-Safety-Aligned-Intelligence-ESAI.git
cd Embedded-Safety-Aligned-Intelligence-ESAI
pip install -r requirements.txt
```

### Training

```bash
# Train ESAI on Moral Temptation environment
python train.py --env moral_temptation --lambda_reg 0.5

# Train baseline PPO
python train.py --env moral_temptation --lambda_reg 0 --method ppo
```

### Evaluation

```bash
python evaluate.py --checkpoint results/moral_temptation/esai_seed1/
```

---

## 📁 Project Structure

```
├── esaiv3/              # Core ESAI implementation
│   ├── agents/          # Agent architectures (ESAI, PPO, CPO)
│   ├── envs/            # Environment wrappers
│   └── utils/           # Utilities and helpers
├── configs/             # Training configurations
├── scripts/             # Experiment scripts
├── train.py             # Main training script
├── evaluate.py          # Evaluation script
└── paper/               # LaTeX paper source
```

---

## 📊 Complete Lambda Sweep Results

| Method | λ | Prosocial Rate | Entropy | Regime |
|--------|---|----------------|---------|--------|
| PPO | 0 | 6.5% | 0.091 | Baseline |
| CPO | 2.0 | 0.35% | 0.004 | Baseline |
| ESAI | 0.05 | 6.8% | 0.091 | Selfish |
| ESAI | 0.078 | 5.0% | 0.004 | Selfish |
| ESAI | 0.08 | 58.7% | **0.672** | Transition |
| **ESAI** | **0.085** | **100%** | 0.005 | **Aligned** |
| **ESAI** | **0.5** | **100%** | 0.050 | **Aligned** |
| **ESAI** | **1.0** | **100%** | 0.006 | **Aligned** |

> **Key insight**: CPO's failure (0.35%) despite using alignment regret suggests that the alignment penalty alone may be insufficient — but systematic ablation is needed to confirm this.

> **Saturation**: All λ ≥ 0.085 achieve identical results. Use the minimum sufficient value.

---

## 📄 Citation

```bibtex
@inproceedings{esai2026,
  title={Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning},
  author={Rathva, Harsh and Mishra, Pruthwik},
  booktitle={Adaptive and Learning Agents Workshop (ALA) at AAMAS},
  year={2026},
  address={Paphos, Cyprus}
}
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.