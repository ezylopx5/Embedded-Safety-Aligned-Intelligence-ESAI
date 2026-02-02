#!/usr/bin/env python3
"""
Lambda sweep analysis for finding optimal λ that beats PPO
"""
import torch
import numpy as np
from pathlib import Path

def load_checkpoint_metrics(path):
    """Load and extract metrics from checkpoint"""
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    result = {
        "episode": ckpt.get("episode", 0),
        "prosocial_ratio": None,
        "mean_reward": None,
        "help_count": None,
        "steal_count": None,
    }
    
    if "metrics_history" in ckpt:
        mh = ckpt["metrics_history"]
        if isinstance(mh, dict):
            if "eval_prosocial_ratio" in mh and len(mh["eval_prosocial_ratio"]) > 0:
                result["prosocial_ratio"] = mh["eval_prosocial_ratio"][-1]
            if "eval_mean_reward" in mh and len(mh["eval_mean_reward"]) > 0:
                result["mean_reward"] = mh["eval_mean_reward"][-1]
            if "eval_total_help" in mh and len(mh["eval_total_help"]) > 0:
                result["help_count"] = mh["eval_total_help"][-1]
            if "eval_total_steal" in mh and len(mh["eval_total_steal"]) > 0:
                result["steal_count"] = mh["eval_total_steal"][-1]
    
    return result

def main():
    print("=" * 70)
    print("LAMBDA ANALYSIS: Finding λ that beats PPO")
    print("=" * 70)
    
    # 1. PPO Baseline
    print("\n📊 PPO BASELINE (λ=0)")
    print("-" * 50)
    ppo_path = Path("/Users/haxx_sh/Desktop/MARL/data/results/logs/moral_temptation/ppo_baseline_mt/seed_42/checkpoint_best.pt")
    ppo_metrics = load_checkpoint_metrics(ppo_path)
    if ppo_metrics:
        print(f"  Episode: {ppo_metrics['episode']}")
        print(f"  Prosocial Ratio: {ppo_metrics['prosocial_ratio']}")
        print(f"  Mean Reward: {ppo_metrics['mean_reward']}")
        print(f"  Help/Steal: {ppo_metrics['help_count']}/{ppo_metrics['steal_count']}")
    else:
        print("  No PPO baseline found!")
    
    # 2. ESAI with λ=2.0 (esaiv3_paper)
    print("\n📊 ESAI-v3 (λ=2.0) - 5 seeds")
    print("-" * 50)
    esai_dir = Path("/Users/haxx_sh/Desktop/MARL/results/moral_temptation/esaiv3_paper")
    
    esai_pr = []
    esai_reward = []
    
    for seed_dir in sorted(esai_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        ckpt_path = seed_dir / "checkpoint_final.pt"
        m = load_checkpoint_metrics(ckpt_path)
        if m and m["prosocial_ratio"] is not None:
            esai_pr.append(m["prosocial_ratio"])
            esai_reward.append(m["mean_reward"])
            print(f"  {seed_dir.name}: PR={m['prosocial_ratio']:.3f}, Reward={m['mean_reward']:.1f}")
    
    if esai_pr:
        print(f"\n  Mean PR: {np.mean(esai_pr):.3f} ± {np.std(esai_pr):.3f}")
        print(f"  Mean Reward: {np.mean(esai_reward):.1f} ± {np.std(esai_reward):.1f}")
    
    # 3. Other λ values
    print("\n📊 OTHER λ VALUES")
    print("-" * 50)
    
    other_experiments = [
        ("esaiv3_lamda1_fixed", 1.0),
        ("esaiv3_lamda1_c", 1.0),
        ("esaiv3_a100_lambda2", 2.0),
        ("esaiv3_action_specific", 1.0),
    ]
    
    for exp_name, lambda_val in other_experiments:
        exp_dir = Path(f"/Users/haxx_sh/Desktop/MARL/results/moral_temptation/{exp_name}")
        if exp_dir.exists():
            for seed_dir in exp_dir.iterdir():
                if seed_dir.is_dir():
                    ckpt_path = seed_dir / "checkpoint_final.pt"
                    m = load_checkpoint_metrics(ckpt_path)
                    if m:
                        pr = m['prosocial_ratio'] if m['prosocial_ratio'] else "N/A"
                        rw = m['mean_reward'] if m['mean_reward'] else "N/A"
                        print(f"  {exp_name} (λ={lambda_val}): PR={pr}, Reward={rw}, ep={m['episode']}")
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What λ do we need?")
    print("=" * 70)
    
    if ppo_metrics and ppo_metrics["prosocial_ratio"] is not None:
        ppo_pr = ppo_metrics["prosocial_ratio"]
        print(f"\n  PPO baseline PR: {ppo_pr:.3f}")
        print(f"  ESAI (λ=2) PR:   {np.mean(esai_pr):.3f}")
        
        if np.mean(esai_pr) > ppo_pr:
            improvement = (np.mean(esai_pr) - ppo_pr) / max(ppo_pr, 0.001) * 100
            print(f"\n  ✅ ESAI beats PPO by {improvement:.1f}%")
        else:
            print(f"\n  ⚠️  ESAI does NOT beat PPO yet")
            print(f"  Need to find better λ or train longer")
    else:
        print("\n  Need to run PPO baseline first!")
        print("  Command: python train.py --config configs/model/ppo_baseline.yaml --env-config configs/envs/moral_temptation.yaml")

if __name__ == "__main__":
    main()
