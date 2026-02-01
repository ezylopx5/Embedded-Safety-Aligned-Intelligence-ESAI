  ESAI-v3: Learning Internal Alignment Embeddings for Scalable Multi-Agent Coordination

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **ESAI-v3** from the paper:

> Learning Internal Alignment Embeddings for Scalable Multi-Agent Coordination
> Harsh Rathva  
> NeurIPS 2025 (Under Review)

---

 🎯 Overview

ESAI-v3 introduces **Internal Alignment Embeddings (IAE)**: differentiable latent regulators that predict and attenuate externalized harm during multi-agent coordination. Unlike reward shaping, IAE dynamics are:

- **Counterfactually supervised** via softmin reference distributions
- **Graph-diffused** with similarity-weighted propagation
- **Interpretable** via attention mechanisms and bias-mitigation controls

---

 🚀 Quick Start

 Installation

```bash
git clone https://github.com/HarshRathva/ESAI-v3.git
cd ESAI-v3
pip install -r requirements.txt