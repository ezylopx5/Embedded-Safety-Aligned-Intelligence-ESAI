# ESAI Quick-Start Guide: Priority Experiments
## Based on ALA @ AAMAS 2026 Workshop Paper

---

## 🎯 Core Objective
Validate ESAI's theoretical claims through minimal viable experiments to support workshop publication.

---

## 📊 Priority Ranking (From Workshop Paper)

### **Critical Priority 1: Stability & Bounded Dynamics**
**Paper Claim**: Proposition 1 guarantees bounded IAE under spectral condition.
**What to Validate**:
1. IAE norms remain bounded during training
2. Spectral condition ∥γ_E I - α L∥₂ + L_g < 1 prevents divergence
3. Hebbian traces converge (Proposition 2)

**Minimal Experiment** (1-2 days):
```bash
# Single environment, track IAE norm over time
python experiments/stability_test.py \
    --env moral_temptation \
    --agents 4 \
    --episodes 10000 \
    --monitor_iae_norm \
    --monitor_spectral_radius
```

**Success Criteria**:
- IAE norm plateaus < 10.0
- No exponential growth
- Spectral radius ρ(γ_E I - α L) < 0.95

---

### **Critical Priority 2: Component Necessity**
**Paper Claim**: ESAI integrates 4 mechanisms; alternative implementations may exist.
**What to Validate**:
- Which components are essential?
- Can simpler architectures work?

**Minimal Experiment** (2-3 days):
```python
# Ablation study (from workshop paper Section 5.3)
ablations = [
    'Full ESAI',
    'No Counterfactual',  # Remove Eq. 4-6
    'No Attention',       # Remove Eq. 7
    'No Hebbian',         # Remove Eq. 8
    'No Diffusion',       # Remove Eq. 3 diffusion term
    'Vanilla PPO'         # Baseline
]

# Run each for 5k episodes on Moral Temptation
for ablation in ablations:
    python experiments/run_ablation.py \
        --variant $ablation \
        --episodes 5000 \
        --seeds 3
```

**Success Criteria**:
- Full ESAI outperforms all ablations
- At least 2 components show >10% performance drop when removed

---

### **Critical Priority 3: Harm Observability Assumption**
**Paper Claim**: Assumption 1 - ESAI requires externally specified harm metrics.
**What to Validate**:
- Does IAE magnitude correlate with ground-truth harm?
- Pearson r > 0.5 would support the framework

**Minimal Experiment** (1 day):
```python
# Test IAE-harm correlation (Table 3 from full paper)
python experiments/correlation_test.py \
    --env moral_temptation \
    --harm_metric victim_distress \
    --episodes 500 \
    --checkpoint checkpoints/esai_trained.pt

# Expected output:
# Pearson r = 0.XX, p < 0.001
```

**Success Criteria**:
- Pearson correlation r > 0.5 with p < 0.01
- Positive correlation (higher IAE → more harm)

---

### **High Priority 4: Computational Overhead**
**Paper Claim**: Table 1 shows O(N|A|k² + N|A|kd) complexity.
**What to Validate**:
- Is overhead manageable for moderate-scale problems?
- Workshop claims ~4.5× overhead for typical parameters

**Minimal Experiment** (0.5 days):
```python
# Profile computational cost
python experiments/profile_overhead.py \
    --env moral_temptation \
    --agents [4, 8, 16] \
    --measure_time \
    --compare_to_ppo

# Measure:
# - Forward pass time
# - Memory usage
# - Wall-clock training time
```

**Success Criteria**:
- Overhead < 10× vanilla PPO
- Scales roughly linearly with N agents
- Training completes in < 12 hours on single GPU

---

### **Medium Priority 5: Bias Mitigation**
**Paper Claim**: Eq. 10 similarity-suppression regularizer enables fairness-performance tradeoffs.
**What to Validate**:
- Can λ_bias control in-group favoritism?
- Is there a tunable tradeoff?

**Minimal Experiment** (1-2 days):
```python
# Sweep bias regularization strength
python experiments/bias_sweep.py \
    --env social_distress \
    --lambda_bias [0.0, 0.01, 0.05, 0.1] \
    --episodes 5000 \
    --measure_help_gap

# Measure:
# - In-group vs out-group help probability
# - Task performance
```

**Success Criteria**:
- λ_bias = 0.01 reduces help gap by >20%
- Performance degrades <10% at effective bias reduction

---

### **Medium Priority 6: Comparison to Baselines**
**Paper Claim**: ESAI differs from reward shaping, CPO, multi-objective RL (Section 5.2).
**What to Validate**:
- Does ESAI offer advantages over simpler methods?

**Minimal Experiment** (3-4 days):
```bash
# Compare 4 methods on 1 environment
methods=(esai ppo reward_shaping cpo)
for method in "${methods[@]}"; do
    python experiments/run_baseline.py \
        --method $method \
        --env moral_temptation \
        --episodes 10000 \
        --seeds 3
done

# Statistical test
python experiments/compare_methods.py \
    --metrics prosocial_ratio task_reward \
    --test t_test
```

**Success Criteria**:
- ESAI achieves highest prosocial ratio
- Statistically significant improvement (p < 0.05)
- At least matches task performance of baselines

---

## 🚀 Recommended Execution Order

### **Week 1: Core Validation**
```
Day 1-2: Priority 1 (Stability)
Day 3-4: Priority 3 (Correlation)
Day 5:   Priority 4 (Overhead)
```

### **Week 2: Component Analysis**
```
Day 1-3: Priority 2 (Ablations)
Day 4-5: Priority 5 (Bias)
```

### **Week 3: Comparative Evaluation**
```
Day 1-4: Priority 6 (Baselines)
Day 5:   Results compilation
```

---

## 📋 Minimal Implementation Checklist

### **Must Implement**:
- [ ] IAE dynamics (Eq. 3)
- [ ] Counterfactual forecasting (Eq. 4-6)
- [ ] IAE-weighted attention (Eq. 7)
- [ ] Hebbian memory (Eq. 8)
- [ ] Graph diffusion (Eq. 9)
- [ ] Policy optimization (Eq. 11-13)
- [ ] Vanilla PPO baseline
- [ ] Moral Temptation environment

### **Nice to Have**:
- [ ] CPO baseline
- [ ] Reward shaping baseline
- [ ] Multiple environments
- [ ] Advanced visualizations

### **Can Skip for Initial Validation**:
- [ ] Multi-objective RL baseline
- [ ] Zero-shot scaling experiments
- [ ] High-dimensional environments
- [ ] Adversarial robustness tests

---

## 🎯 Minimal Success Criteria

To support workshop paper claims, you need:

1. **Bounded Dynamics**: ✓ IAE norms stay finite
2. **Component Necessity**: ✓ At least 2/4 components show necessity
3. **Harm Correlation**: ✓ r > 0.5 with ground-truth harm
4. **Manageable Overhead**: ✓ Training completes in reasonable time
5. **Competitive Performance**: ✓ Matches or beats 1 baseline

If you achieve these 5, your workshop paper's theoretical claims are empirically grounded.

---

## 🔧 Simplified Implementation Path

### **Start Here** (Absolute Minimum - 1 Week):

```python
# File: minimal_esai.py

class MinimalESAI:
    """Bare-bones ESAI for quick validation"""
    
    def __init__(self, obs_dim, action_dim, iae_dim=16):
        # Simplify: k=16 instead of k=32
        self.iae_dim = iae_dim
        
        # Only essential components
        self.iae_dynamics = IAEDynamics(iae_dim, obs_dim, action_dim)
        self.forecaster = CounterfactualForecaster(iae_dim, obs_dim, action_dim)
        
        # Skip initially:
        # - Attention (use raw observations)
        # - Hebbian memory (use simple recurrence)
        # - Graph diffusion (single agent or broadcast)
        
        self.policy = SimplePolicyNet(obs_dim, action_dim)
        self.value = SimpleValueNet(obs_dim)
        
        self.E = torch.zeros(1, iae_dim)
    
    def forward(self, obs, action=None, reward=None):
        # Direct observation (no attention initially)
        action_logits = self.policy(obs)
        value = self.value(obs)
        
        if action is not None:
            # Update IAE (without diffusion initially)
            self.E = self.iae_dynamics(self.E, obs, action, reward)
            
            # Compute alignment regret
            AR = self.forecaster.compute_alignment_regret(
                self.E, obs, action_logits, reward
            )
            
            return action_logits, value, AR
        
        return action_logits, value, torch.tensor(0.0)
```

### **Add Complexity Gradually**:

1. **Week 1**: Minimal ESAI (IAE + counterfactual only)
2. **Week 2**: Add attention mechanism
3. **Week 3**: Add Hebbian memory
4. **Week 4**: Add graph diffusion

This incremental approach lets you:
- Validate core concepts early
- Debug simpler systems first
- Understand contribution of each component
- Maintain working baseline at each step

---

## 📊 Expected Results (Realistic Estimates)

Based on similar MARL research:

### **Moral Temptation Grid (Δ=5)**:
| Method | Prosocial Ratio | Training Time | Success Likelihood |
|--------|----------------|---------------|-------------------|
| Minimal ESAI | 0.60-0.70 | 2-4 hours | High |
| Full ESAI | 0.70-0.85 | 4-8 hours | Medium |
| Vanilla PPO | 0.30-0.40 | 1-2 hours | Very High |
| Reward Shaping | 0.55-0.65 | 2-3 hours | High |

### **Computational Resources**:
- **Single environment validation**: 1 GPU × 3 days
- **Full ablation study**: 1 GPU × 1 week
- **Complete workshop validation**: 1 GPU × 2-3 weeks

### **Common Failure Modes**:
1. **IAE explodes**: Increase α (diffusion) or decrease γ_E
2. **No learning**: Decrease λ_reg (alignment penalty too strong)
3. **Unstable training**: Add gradient clipping, reduce learning rate
4. **Counterfactual collapse**: Use EMA target network
5. **Memory overflow**: Reduce batch size or iae_dim

---

## 🎓 Key Insights from Workshop Paper

### **What Paper Emphasizes**:
1. **Theoretical framework** (not fully validated algorithm)
2. **Conceptual contribution** to differentiable alignment
3. **Open questions** remain (convergence, dimensionality, scaling)
4. **Alternative implementations** may satisfy ESAI principles

### **What Paper Admits**:
1. "Empirical validation remains future work" (Abstract)
2. "Claims about advantages are conjectural" (Section 5.3)
3. "Whether simpler architectures achieve comparable alignment... is unknown" (Section 5.3)
4. "We hope this framework motivates the community to investigate" (Conclusion)

### **Your Experiment Goal**:
Not to "prove ESAI is best", but to show:
1. Framework is **implementable**
2. Components have **theoretical grounding** (bounded, stable)
3. Design choices are **reasonable** (harm correlation exists)
4. Approach is **competitive** (not worse than baselines)

This is sufficient for a workshop paper presenting a conceptual framework.

---

## 🚦 Go/No-Go Decision Points

### **After Week 1** (Stability + Correlation):
- ✅ **GO**: IAE bounded, r > 0.3 with harm
- 🛑 **PIVOT**: IAE explodes → simplify architecture, reduce k
- 🛑 **PIVOT**: No correlation → rethink harm metric or IAE supervision

### **After Week 2** (Ablations):
- ✅ **GO**: ≥2 components show benefit
- 🛑 **PIVOT**: All components unnecessary → present as "minimal ESAI"
- 🛑 **PIVOT**: Only 1 component helps → simplify to that component + baseline

### **After Week 3** (Baselines):
- ✅ **GO**: Competitive with ≥1 baseline, unique properties
- 🛑 **PIVOT**: Worse than all baselines → refocus on theoretical contribution
- ✅ **PIVOT**: Equal to baselines → emphasize interpretability, bias control

---

## 💡 Pro Tips

1. **Start with toy environment**: 4×4 grid, 2 agents before full 8×8
2. **Use tensorboard**: Monitor IAE norm, AR, PR in real-time
3. **Save checkpoints frequently**: Every 1000 episodes
4. **Test small first**: 1 seed, 1000 episodes for debugging
5. **Parallelize smartly**: Run different ablations on different GPUs
6. **Log everything**: You never know what will be useful
7. **Version your code**: Git commit after each working component

---

## 📁 Minimal File Structure

```
esai-minimal/
├── minimal_esai.py        # Core implementation (500 lines)
├── moral_temptation.py    # Simple environment (200 lines)
├── train.py               # Training loop (300 lines)
├── evaluate.py            # Evaluation metrics (100 lines)
├── configs/
│   └── default.yaml       # Hyperparameters
└── experiments/
    ├── stability_test.py
    ├── correlation_test.py
    ├── ablation_study.py
    └── compare_baselines.py
```

Total: ~1100 lines of core code (very manageable!)

---

## ⏱️ Time Budget Breakdown

### **Absolute Minimum (1 week intensive)**:
- Implementation: 2 days
- Stability + correlation: 1 day
- 1 ablation run: 2 days
- 1 baseline comparison: 1 day
- Results + figures: 1 day

### **Comfortable (3 weeks part-time)**:
- Implementation: 1 week
- Priority 1-3: 1 week
- Priority 4-6: 1 week

### **Comprehensive (6 weeks)**:
- Follow full workflow document
- All environments, baselines, analyses

---

## 🎯 Final Recommendation

**For workshop paper support, execute**:

```bash
# Week 1: Core validation
./run_priority_1_stability.sh    # 2 days
./run_priority_3_correlation.sh  # 1 day
./run_priority_4_overhead.sh     # 0.5 days

# Week 2: Components
./run_priority_2_ablations.sh    # 3 days

# Week 3: Comparison
./run_priority_6_baselines.sh    # 3 days
./compile_results.sh             # 1 day
```

This gives you:
- ✅ Empirical grounding for theoretical claims
- ✅ Evidence of component necessity
- ✅ Competitive baseline comparison
- ✅ Manageable 3-week timeline

**Total Effort**: ~80-100 hours over 3 weeks
**Total Compute**: ~150 GPU-hours

This is very achievable for validating a workshop paper! 🚀