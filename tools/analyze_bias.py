"""Analyze bias mitigation sweep."""
import pandas as pd
import numpy as np
import glob
import os
import json
import gzip
from pathlib import Path


def iter_jsonl(path):
    p = Path(path)
    if p.suffix == '.gz':
        f = gzip.open(p, 'rt')
    else:
        f = open(p, 'r')
    with f:
        for line in f:
            yield json.loads(line)


def compute_help_gap(run_dir):
    """Compute help gap across similarity bins."""
    bins = {i: {"help": 0, "total": 0} for i in range(5)}
    
    for f in Path(run_dir).glob("per_step_eval*.jsonl*"):
        for rec in iter_jsonl(f):
            sim_bin = rec.get("sim_bin")
            pr_flag = rec.get("PR_flag")
            
            if sim_bin is not None:
                bins[sim_bin]["total"] += 1
                if pr_flag in ("help", 1, True):
                    bins[sim_bin]["help"] += 1
    
    pr_by_bin = []
    for i in range(5):
        if bins[i]["total"] > 0:
            pr_by_bin.append(bins[i]["help"] / bins[i]["total"])
        else:
            pr_by_bin.append(0.0)
    
    if len(pr_by_bin) > 0:
        return max(pr_by_bin) - min(pr_by_bin)
    return 0.0


def aggregate(env):
    lambdas = ["0.0", "0.001", "0.01", "0.1"]
    
    print(f"\n{env}:")
    print(f"{'lambda_bias':<12} {'PR':>8} {'AR':>8} {'Help Gap':>10}")
    print("-" * 50)
    
    for L in lambdas:
        prs, ars, gaps = [], [], []
        
        for seed in range(10):
            run_dir = f"data/results/logs/{env}/bias_{env}_L{L}/seed_{seed}/"
            
            # Load metrics
            csv_path = os.path.join(run_dir, "eval_metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    prs.append(df.iloc[0]['PR'])
                    ars.append(df.iloc[0]['AR'])
            
            # Compute gap
            if os.path.exists(run_dir):
                gap = compute_help_gap(run_dir)
                gaps.append(gap)
        
        if prs:
            print(f"{L:<12} {np.mean(prs):>8.3f} {np.mean(ars):>8.3f} {np.mean(gaps):>10.3f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Bias Mitigation Sweep")
    print("="*60)
    
    envs = ["social_distress", "mpe"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)