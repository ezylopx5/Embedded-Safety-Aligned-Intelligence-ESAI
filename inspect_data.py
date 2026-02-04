import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Base path
BASE_PATH = "/Users/haxx_sh/Desktop/MARL"

# Experiment Mapping
EXPERIMENTS = {
    "ESAI (λ=0.1)": "MAR_results_moral_temptation/seed_1-lamb=0.1",
    "ESAI (λ=0.05)": "MAR_results_moral_temptation/seed_1-lamb=0.05",
    "ESAI (λ=0.07)": "MAR_results_moral_temptation/seed_1-lamb=0.07",
    "ESAI (λ=0.078)": "MAR_results_moral_temptation/seed_1-lamb=0.078",
    "ESAI (λ=0.06369)": "MAR_results_moral_temptation/seed_1-lamb=0.06369",
    "ESAI (λ=0.5)": "MAR_results_moral_temptation/esai_seed2_5_0.5",
    "ESAI (λ=0.080)": "MAR_results_moral_temptation/esai_seed2_5_0.080",
    "ESAI (λ=0.085)": "MAR_results_moral_temptation/esai_seed2_5_0.085",
    "ESAI (λ=5.0)": "MAR_results_moral_temptation/esai_seed2_5",
    "PPO Baseline (Cmp)": "MAR_results_moral_temptation/ppo_baseline_cmp",
    "PPO Baseline (Paper)": "moral_temptation 3/ppo_baseline_paper",
    "CPO Baseline": "moral_temptation 3/cpo_baseline/seed_1"
}

def find_metrics_file(path_suffix):
    full_path = os.path.join(BASE_PATH, path_suffix)
    # Check direct metrics.json
    if os.path.exists(os.path.join(full_path, "metrics.json")):
        return os.path.join(full_path, "metrics.json")
    
    # Check subdirectories (e.g., seed_42, seed_1)
    for root, dirs, files in os.walk(full_path):
        if "metrics.json" in files:
            return os.path.join(root, "metrics.json")
    return None

def inspect_data():
    print("Inspecting data availability...")
    found_data = {}
    
    for label, path in EXPERIMENTS.items():
        metrics_path = find_metrics_file(path)
        if metrics_path:
            print(f"[FOUND] {label}: {metrics_path}")
            found_data[label] = metrics_path
        else:
            print(f"[MISSING] {label}: {path}")
            
    return found_data

if __name__ == "__main__":
    inspect_data()
