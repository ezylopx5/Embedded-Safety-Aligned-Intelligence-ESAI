"""Analyze wall-clock overhead."""
import json
import glob
import numpy as np


def load_wallclock(pattern):
    times = []
    for meta_path in glob.glob(pattern):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            times.append(meta.get('total_wallclock_s', 0))
    return times


def aggregate(env):
    ppo_pattern = f"data/results/logs/{env}/wallclock_ppo/seed_*/meta.json"
    esai_pattern = f"data/results/logs/{env}/wallclock_esai_v3/seed_*/meta.json"
    
    ppo_times = load_wallclock(ppo_pattern)
    esai_times = load_wallclock(esai_pattern)
    
    if not ppo_times or not esai_times:
        print(f"{env}: Incomplete data")
        return
    
    ppo_mean = np.mean(ppo_times)
    esai_mean = np.mean(esai_times)
    overhead = (esai_mean / ppo_mean - 1) * 100 if ppo_mean > 0 else 0
    
    print(f"{env:20s}  PPO: {ppo_mean:>6.1f}s  ESAI-v3: {esai_mean:>6.1f}s  Overhead: {overhead:>5.1f}%")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Wall-Clock Overhead Analysis")
    print("="*60 + "\n")
    
    envs = ["moral_temptation", "overcooked", "mpe"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)