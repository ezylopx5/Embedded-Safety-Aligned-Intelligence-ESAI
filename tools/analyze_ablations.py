"""Analyze ablation results."""
import pandas as pd
import numpy as np
import glob
import os


def load_variant(env, variant):
    metrics = []
    for seed in range(10):
        path = f"data/results/logs/{env}/ablate_{env}_{variant}/seed_{seed}/eval_metrics.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                metrics.append(df.iloc[0].to_dict())
    return metrics


def aggregate(env):
    variants = ["full", "no_regret", "no_attention", "no_hebbian", "no_diffusion"]
    
    print(f"\n{env}:")
    print(f"{'Variant':<20} {'PR':>8} {'AR':>8} {'ESI':>8} {'IPA':>8}")
    print("-" * 60)
    
    results = {}
    for var in variants:
        data = load_variant(env, var)
        if data:
            results[var] = {
                'PR': np.mean([m['PR'] for m in data]),
                'AR': np.mean([m['AR'] for m in data]),
                'ESI': np.mean([m['ESI'] for m in data]),
                'IPA': np.mean([m['IPA'] for m in data])
            }
            print(f"{var:<20} {results[var]['PR']:>8.3f} {results[var]['AR']:>8.3f} "
                  f"{results[var]['ESI']:>8.3f} {results[var]['IPA']:>8.3f}")
    
    # Compute degradation
    if 'full' in results:
        print("\nDegradation vs Full:")
        for var in variants:
            if var != 'full' and var in results:
                dpr = (results[var]['PR'] - results['full']['PR']) / results['full']['PR'] * 100
                desi = (results[var]['ESI'] - results['full']['ESI']) / results['full']['ESI'] * 100
                print(f"  {var:<20} ΔPR: {dpr:>6.1f}%  ΔESI: {desi:>6.1f}%")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Ablation Analysis")
    print("="*60)
    
    envs = ["moral_temptation", "overcooked"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)