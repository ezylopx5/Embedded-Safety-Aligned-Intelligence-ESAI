"""
Quick validation that alignment penalty is in correct regime.
Run this BEFORE full training to sanity-check hyperparameters.
"""

import yaml
import numpy as np

def validate_alignment_scaling(config_path: str, env_stats: dict = None):
    """
    Check if lambda_reg is in the stable regime.
    
    Args:
        config_path: Path to YAML config
        env_stats: Optional dict with 'mean_reward', 'mean_ar' from random rollout
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    lambda_reg = config.get('lambda_reg', 0.1)
    kappa = config.get('kappa', 0.1)
    
    # Default estimates if no env stats provided
    if env_stats is None:
        env_stats = {
            'mean_reward': 2.5,    # Typical for moral temptation
            'mean_ar': 8.0,        # Typical untrained AR_t
            'max_ar': 15.0,        # Worst-case AR_t
        }
    
    mean_penalty = lambda_reg * env_stats['mean_ar']
    max_penalty = lambda_reg * env_stats['max_ar']
    
    print("=" * 60)
    print("ALIGNMENT PENALTY VALIDATION")
    print("=" * 60)
    print(f"lambda_reg:     {lambda_reg}")
    print(f"kappa:          {kappa}")
    print(f"Mean AR_t:      {env_stats['mean_ar']:.2f}")
    print(f"Mean r_ext:     {env_stats['mean_reward']:.2f}")
    print("-" * 60)
    print(f"Mean penalty:   {mean_penalty:.2f}")
    print(f"Max penalty:    {max_penalty:.2f}")
    print(f"Mean r':        {env_stats['mean_reward'] - mean_penalty:.2f}")
    print("-" * 60)
    
    # Validation checks
    issues = []
    
    if mean_penalty > env_stats['mean_reward']:
        issues.append(
            f"⚠️  OVER-PENALIZATION: penalty ({mean_penalty:.2f}) > reward ({env_stats['mean_reward']:.2f})\n"
            f"    This will cause PPO collapse. Reduce lambda_reg to {env_stats['mean_reward'] / env_stats['mean_ar'] / 2:.4f}"
        )
    
    if mean_penalty < 0.1 * env_stats['mean_reward']:
        issues.append(
            f"⚠️  UNDER-PENALIZATION: penalty ({mean_penalty:.2f}) << reward ({env_stats['mean_reward']:.2f})\n"
            f"    Alignment signal will be ignored. Increase lambda_reg to {env_stats['mean_reward'] / env_stats['mean_ar'] / 2:.4f}"
        )
    
    if max_penalty > 3 * env_stats['mean_reward']:
        issues.append(
            f"⚠️  INSTABILITY RISK: max penalty ({max_penalty:.2f}) >> reward\n"
            f"    Consider adding AR clipping: ar_max: {2 * env_stats['mean_reward'] / lambda_reg:.1f}"
        )
    
    if lambda_reg > 0.2:
        issues.append(
            f"⚠️  lambda_reg = {lambda_reg} is unusually high.\n"
            f"    Typical stable range: [0.03, 0.10]"
        )
    
    if issues:
        print("\n🚨 ISSUES DETECTED:")
        for issue in issues:
            print(f"\n{issue}")
    else:
        print("\n✅ Configuration looks stable!")
        print(f"   Expected r' range: [{env_stats['mean_reward'] - max_penalty:.2f}, {env_stats['mean_reward']:.2f}]")
    
    print("=" * 60)
    
    return len(issues) == 0


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/model/esaiv3_default.yaml"
    validate_alignment_scaling(config_path)