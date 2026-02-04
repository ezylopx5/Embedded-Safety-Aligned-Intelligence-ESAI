# ESAI: Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **ESAI** from the paper:

> **Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning**  
> ALA @ AAMAS 2026

---

## 🎯 Overview

ESAI introduces **Internal Alignment Embeddings (IAE)**: differentiable latent regulators that predict and attenuate externalized harm during multi-agent coordination.

### Key Results
- **100% prosocial rate** on Moral Temptation environment (vs 6.5% PPO, 0% CPO)
- **Phase transition** at λ* ≈ 0.08 — below: selfish behavior, above: prosocial alignment
- **IAE learns harm representations**: AR(STEAL) = 3.63 vs AR(HELP) ≈ 0

### Core Mechanisms
- **Counterfactual supervision** via softmin reference distributions
- **Graph diffusion** with similarity-weighted propagation
- **IAE-weighted attention** for perceptual salience modulation
- **Hebbian affect-memory** for temporal credit assignment

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

## 📊 Key Findings

| Method | λ | Prosocial Rate |
|--------|---|----------------|
| PPO | 0 | 6.5% |
| CPO | 2.0 | 0% |
| **ESAI** | 0.085 | **100%** |
| **ESAI** | 0.5 | **100%** |
| **ESAI** | 1.0 | **100%** |

> **Key insight**: Alignment regret alone (CPO) is insufficient. The full IAE architecture is necessary.

---

## 📄 Citation

```bibtex
@inproceedings{esai2026,
  title={Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning},
  author={Anonymous},
  booktitle={Adaptive and Learning Agents Workshop at AAMAS},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.