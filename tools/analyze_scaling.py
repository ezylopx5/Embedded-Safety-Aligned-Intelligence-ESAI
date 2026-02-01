"""Analyze zero-shot scaling results."""
import pandas as pd
import numpy as np
import os


def load_scaling_metrics(env, seed, tag):
    path = f"data/results/logs/{env}/scale_{env}_N4/seed_{seed}/eval_metrics_{tag}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if len(df) > 0 else None


def aggregate(env):
    pr4_list = []
    pr16_list = []
    
    for seed in range(10):
        m4 = load_scaling_metrics(env, seed, "evalN4")
        m16 = load_scaling_metrics(env, seed, "evalN16")
        
        if m4 and m16:
            pr4_list.append(m4['PR'])
            pr16_list.append(m16['PR'])
    
    if len(pr4_list) == 0:
        print(f"{env}: No data")
        return
    
    pr4 = np.mean(pr4_list)
    pr16 = np.mean(pr16_list)
    retention = pr16 / pr4 if pr4 > 0 else 0
    
    print(f"{env:20s}  PR_4={pr4:.3f}  PR_16={pr16:.3f}  Retention={retention:.3f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Zero-Shot Scaling Analysis (4→16 agents)")
    print("="*60 + "\n")
    
    envs = ["social_distress", "mpe", "ssd"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)