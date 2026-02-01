"""Analyze intervention effects."""
import pandas as pd
import numpy as np
import glob
import os


def load_metrics(env, seed, tag):
    path = f"data/results/logs/{env}/corr_{env}_bias0/seed_{seed}/eval_metrics_{tag}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if len(df) > 0 else None


def aggregate(env):
    results = {tag: [] for tag in ["baseline", "low", "high"]}
    
    for seed in range(10):
        for tag in ["baseline", "low", "high"]:
            metrics = load_metrics(env, seed, tag)
            if metrics:
                results[tag].append(metrics)
    
    if len(results["baseline"]) == 0:
        print(f"{env}: No data")
        return
    
    print(f"\n{env}:")
    for tag in ["baseline", "low", "high"]:
        if results[tag]:
            prs = [m['PR'] for m in results[tag]]
            ars = [m['AR'] for m in results[tag]]
            print(f"  {tag:10s}: PR={np.mean(prs):.3f}±{np.std(prs):.3f}  AR={np.mean(ars):.3f}±{np.std(ars):.3f}")
    
    # Compute delta
    if results["baseline"] and results["low"]:
        pr_base = np.mean([m['PR'] for m in results["baseline"]])
        pr_low = np.mean([m['PR'] for m in results["low"]])
        delta = (pr_low - pr_base) / pr_base * 100
        print(f"  ΔPR (low vs baseline): {delta:.1f}%")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Intervention Causal Analysis")
    print("="*60)
    
    envs = ["moral_temptation", "social_distress", "mpe"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)