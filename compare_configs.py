"""
Compare old vs new config and show impact on transformed rewards.
"""

import numpy as np

def compute_transformed_reward(r_ext, AR_t, lambda_reg):
    """Compute r' = r_ext - lambda_reg * AR_t"""
    return r_ext - lambda_reg * AR_t

# Simulate training trajectory
print("="*70)
print("REWARD TRANSFORMATION ANALYSIS")
print("="*70)

# Typical values from your training run
scenarios = [
    ("Early training (step 2k)", -0.007, 9.17),
    ("Mid training (step 25k)", 0.5, 5.20),
    ("Late training (step 75k)", -0.5, 0.56),
    ("Final (step 100k)", -0.2, 0.46),
]

old_lambda = 2.0
new_lambda = 0.5

print("\nOLD CONFIG (lambda_reg = 2.0):")
print("-" * 70)
print(f"{'Stage':<30} {'r_ext':>10} {'AR_t':>10} {'r_transformed':>15}")
print("-" * 70)

for stage, r_ext, ar_t in scenarios:
    r_trans = compute_transformed_reward(r_ext, ar_t, old_lambda)
    print(f"{stage:<30} {r_ext:>10.3f} {ar_t:>10.3f} {r_trans:>15.3f}")

print("\n" + "="*70)
print("\nNEW CONFIG (lambda_reg = 0.5):")
print("-" * 70)
print(f"{'Stage':<30} {'r_ext':>10} {'AR_t':>10} {'r_transformed':>15}")
print("-" * 70)

for stage, r_ext, ar_t in scenarios:
    r_trans = compute_transformed_reward(r_ext, ar_t, new_lambda)
    print(f"{stage:<30} {r_ext:>10.3f} {ar_t:>10.3f} {r_trans:>15.3f}")

print("\n" + "="*70)
print("IMPACT ANALYSIS:")
print("-" * 70)

for stage, r_ext, ar_t in scenarios:
    r_old = compute_transformed_reward(r_ext, ar_t, old_lambda)
    r_new = compute_transformed_reward(r_ext, ar_t, new_lambda)
    improvement = r_new - r_old
    
    print(f"\n{stage}:")
    print(f"  Old r': {r_old:>8.3f}")
    print(f"  New r': {r_new:>8.3f}")
    print(f"  Improvement: {improvement:>8.3f} ({improvement/abs(r_old)*100:.1f}% change)")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("-" * 70)
print("✓ Early training: Old config created massive negative rewards (-18.3)")
print("  → New config: Only -4.6, much more learnable")
print("✓ Mid training: Old config still highly negative (-9.9)")
print("  → New config: Close to zero (-2.1), allows policy to explore")
print("✓ Late training: Both converge to similar values once AR_t is low")
print("  → But old config caused premature policy collapse")
print("\n" + "="*70)

# Entropy analysis
print("\nENTROPY COEFFICIENT IMPACT:")
print("-" * 70)
print(f"Old entropy_coef: 0.01")
print(f"New entropy_coef: 0.02")
print("\nExpected effect:")
print("  - 2x stronger exploration bonus")
print("  - Slower policy convergence (good - prevents premature collapse)")
print("  - More diverse action distributions during training")
print("="*70)