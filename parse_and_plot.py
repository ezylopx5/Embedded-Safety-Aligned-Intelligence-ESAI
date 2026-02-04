import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
OUTPUT_DIR = "research_artifacts"
BASE_PATH = "/Users/haxx_sh/Desktop/MARL"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Data Containers
data_store = []

def find_metrics_files(root_dir):
    metrics_files = []
    for root, dirs, files in os.walk(root_dir):
        if "metrics.json" in files:
            metrics_files.append(os.path.join(root, "metrics.json"))
    return metrics_files

def extract_lambda_from_path(path):
    match = re.search(r"lamb=([0-9.]+)", path)
    if match:
        return float(match.group(1))
    match = re.search(r"lam=([0-9.]+)", path)
    if match:
        return float(match.group(1))
    match = re.search(r"esai_seed1_5_([0-9.]+)", path)
    if match:
        return float(match.group(1))
    return None

def parse_logs(log_path):
    print(f"Parsing logs from {log_path}...")
    current_lambda = None
    current_seed = "unknown"
    
    # Regex patterns
    # Schedule: λ=5.0000, ...
    # [train] Schedule configuration: lambda_reg: 0 -> 1.0 ...
    # [DIAG] Step 20,480, Episode 10
    # Episode: Return=236.30, Steps=2048
    # Rewards: r_ext=0.115, AR=0.441, r'=-2.091
    # PR: 0.929
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    current_run_data = {
        "steps": [],
        "reward": [],
        "ar": [],
        "pr": []
    }
    
    # Heuristic: reset run data when we see a new python command or schedule config
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect new run
        if "Schedule configuration:" in line and "lambda_reg:" in lines[i+1]:
            # Save previous run if it has data
            if current_lambda is not None and len(current_run_data["steps"]) > 0:
                print(f"  -> Extracted run for λ={current_lambda}, steps={len(current_run_data['steps'])}")
                df = pd.DataFrame(current_run_data)
                df["lambda"] = current_lambda
                df["algorithm"] = "ESAI"
                data_store.append(df)
            
            # Reset
            current_run_data = {"steps": [], "reward": [], "ar": [], "pr": []}
            
            # Extract lambda from next line
            # "  lambda_reg: 0 → 5.0 over 10000 steps"
            lambda_line = lines[i+1].strip()
            # Match both -> and unicode arrow →
            match = re.search(r"[->→]\s*([\d\.]+)", lambda_line)
            if match:
                current_lambda = float(match.group(1))
            else:
                current_lambda = "unknown"
                print(f"  [WARN] Could not parse lambda from: {lambda_line}")
                
        # Parse Metrics
        if "[DIAG] Step" in line:
            # [DIAG] Step 20,480, Episode 10
            match = re.search(r"Step ([\d,]+)", line)
            if match:
                step = int(match.group(1).replace(",", ""))
                current_run_data["steps"].append(step)
                
        if "Episode:  Return=" in line:
            # Episode:  Return=236.30, Steps=2048
            match = re.search(r"Return=([\d\.\-]+)", line)
            if match:
                # If we parsed a step recently (heuristic), append reward
                # We assume strict ordering: Step -> Episode -> Rewards -> PR
                if len(current_run_data["steps"]) > len(current_run_data["reward"]):
                    current_run_data["reward"].append(float(match.group(1)))
                    
        if "Rewards:  r_ext=" in line:
            # Rewards:  r_ext=0.115, AR=0.441, r'=-2.091
            match = re.search(r"AR=([\d\.\-]+)", line)
            if match:
                if len(current_run_data["steps"]) > len(current_run_data["ar"]):
                    current_run_data["ar"].append(float(match.group(1)))
                    
        if "PR: " in line:
            # PR:       0.929 (conditional on engagement)
            match = re.search(r"PR:\s+([\d\.]+)", line)
            if match:
                if len(current_run_data["steps"]) > len(current_run_data["pr"]):
                    current_run_data["pr"].append(float(match.group(1)))

    # Save last run
    if current_lambda is not None and len(current_run_data["steps"]) > 0:
        print(f"  -> Extracted run for λ={current_lambda}, steps={len(current_run_data['steps'])}")
        df = pd.DataFrame(current_run_data)
        df["lambda"] = current_lambda
        df["algorithm"] = "ESAI"
        data_store.append(df)

def parse_metrics_json(path, label, lambda_val=None):
    print(f"Parsing metrics.json from {path}...")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Structure check: is it array of dicts or dict of arrays?
        # User snippet: { "global_step": 501760, "episode": 245, "reward": [...] }
        # This looks like a final snapshot, not a time series log?
        # Wait, the snippet showed "reward": [69.14, 119.8, ...]
        # This implies "reward" is the history.
        # But "global_step" is a single int.
        # So "reward" is history, but do we have "step" history?
        # Usually steps are proportional to index * interval.
        # Let's assume log_interval = 2048 steps (episode length).
        
        rewards = data.get("reward", [])
        
        # Check for other metrics
        # The user file snippet didn't show 'pr' or 'ar', but we can try common names
        pr = data.get("prosocial_ratio", data.get("pr", []))
        ar = data.get("alignment_regret", data.get("ar", []))
        
        steps = [i * 2048 for i in range(len(rewards))] # Approximation
        
        # If lengths match
        min_len = len(rewards)
        if len(pr) > 0: min_len = min(min_len, len(pr))
        if len(ar) > 0: min_len = min(min_len, len(ar))
        
        df_data = {
            "steps": steps[:min_len],
            "reward": rewards[:min_len],
        }
        if len(pr) > 0: df_data["pr"] = pr[:min_len]
        if len(ar) > 0: df_data["ar"] = ar[:min_len]
        
        df = pd.DataFrame(df_data)
        df["algorithm"] = label
        if lambda_val is not None:
            df["lambda"] = lambda_val
        else:
            df["lambda"] = np.nan
            
        data_store.append(df)
        print(f"  -> Extracted {len(df)} records for {label}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to parse {path}: {e}")

def parse_marl_metrics(root_dir):
    metrics_files = find_metrics_files(root_dir)
    for path in metrics_files:
        lambda_val = extract_lambda_from_path(path)
        parse_metrics_json(path, "ESAI", lambda_val=lambda_val)

def parse_ppo_cpo_logs(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    current_algo = "PPO Baseline (Paper)"
    current_lambda = None
    last_train_reward = None
    current_entry = None

    for line in lines:
        line = line.strip()

        if "ppo_baseline_paper" in line:
            current_algo = "PPO Baseline (Paper)"
        if "cpo_baseline" in line:
            current_algo = "CPO Baseline"

        if line.startswith("Training:") and " r=" in line:
            match = re.search(r"r=([0-9.]+)", line)
            if match:
                last_train_reward = float(match.group(1))

        if "Schedule: λ=" in line:
            match = re.search(r"λ=([0-9.]+)", line)
            if match:
                current_lambda = float(match.group(1))

        if "[DIAG] Step" in line:
            match = re.search(r"Step ([\d,]+)", line)
            if match:
                current_entry = {
                    "steps": int(match.group(1).replace(",", "")),
                    "reward": last_train_reward,
                    "ar": None,
                    "pr": None
                }

        if "Episode:  Return=" in line and current_entry is not None and current_entry["reward"] is None:
            match = re.search(r"Return=([\d\.\-]+)", line)
            if match:
                current_entry["reward"] = float(match.group(1))

        if "Rewards:  r_ext=" in line and current_entry is not None:
            match = re.search(r"AR=([\d\.\-]+)", line)
            if match:
                current_entry["ar"] = float(match.group(1))

        if "PR:" in line and current_entry is not None:
            match = re.search(r"PR:\s+([\d\.]+)", line)
            if match:
                current_entry["pr"] = float(match.group(1))
                if current_entry["reward"] is not None:
                    df = pd.DataFrame([current_entry])
                    df["algorithm"] = current_algo
                    df["lambda"] = current_lambda if current_algo == "CPO Baseline" else 0.0
                    data_store.append(df)
            current_entry = None

# --- Execution ---

# 1. Parse Logs
parse_logs("/Users/haxx_sh/Desktop/MARL/LogsOfESAI.txt")

# 2. Parse Metrics JSON
# ESAI λ=0.05
parse_metrics_json(
    "/Users/haxx_sh/Desktop/MARL/MAR_results_moral_temptation/seed_1-lamb=0.05/seed_42/metrics.json", 
    "ESAI", lambda_val=0.05
)
# ESAI λ=0.078
parse_metrics_json(
    "/Users/haxx_sh/Desktop/MARL/MAR_results_moral_temptation/seed_1-lamb=0.078/seed_42/metrics.json", 
    "ESAI", lambda_val=0.078
)
# CPO Baseline
parse_metrics_json(
    "/Users/haxx_sh/Desktop/MARL/moral_temptation 3/cpo_baseline/seed_1/metrics.json", 
    "CPO Baseline"
)

# PPO Baseline (Paper) - Note: Only checkpoints/config exist, no metrics.json found in LS output.
# If metrics.json is missing, we can't plot it.
# However, user insisted. Let's try to find it recursively or skip if not found.
ppo_paper_path = find_metrics_file("moral_temptation 3/ppo_baseline_paper/seed_1")
if ppo_paper_path:
    parse_metrics_json(ppo_paper_path, "PPO Baseline (Paper)")
else:
    # Try seed 2 or others if seed 1 is missing
    # But based on LS, only checkpoint_final.pt and config.json exist.
    print("[WARN] PPO Baseline (Paper) metrics.json not found in seed_1. Checking other seeds...")
    for seed in range(2, 6):
        path = find_metrics_file(f"moral_temptation 3/ppo_baseline_paper/seed_{seed}")
        if path:
            parse_metrics_json(path, "PPO Baseline (Paper)")
            break


# 3. Consolidate
if not data_store:
    print("No data found!")
    exit()

full_df = pd.concat(data_store, ignore_index=True)

# Clean Data
# Convert lambda to numeric, coercing errors (unknown) to NaN
full_df["lambda"] = pd.to_numeric(full_df["lambda"], errors='coerce')
# Drop rows where lambda is NaN for ESAI (but keep baselines which might have NaN lambda or we assign them 0/None)
# Actually, baselines like CPO don't have lambda. We should keep them.
# But for "Lambda Sensitivity" plot, we only want ESAI with valid lambda.

# 4. Generate Plots

# A. Learning Curves (Reward)
plt.figure(figsize=(10, 6))
sns.lineplot(data=full_df, x="steps", y="reward", hue="lambda", style="algorithm", palette="viridis")
plt.title("Learning Curves: Reward vs Steps")
plt.xlabel("Environment Steps")
plt.ylabel("Episode Reward")
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/learning_curve_reward.png", dpi=300)
plt.close()

# B. Prosocial Ratio (if available)
if "pr" in full_df.columns:
    plt.figure(figsize=(10, 6))
    # Filter out CPO if it doesn't have PR
    pr_df = full_df.dropna(subset=["pr"])
    if not pr_df.empty:
        sns.lineplot(data=pr_df, x="steps", y="pr", hue="lambda", style="algorithm", palette="viridis")
        plt.title("Prosocial Ratio vs Steps")
        plt.xlabel("Environment Steps")
        plt.ylabel("Prosocial Ratio (0-1)")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/learning_curve_pr.png", dpi=300)
    plt.close()

# C. Lambda Sensitivity (Final Performance)
# Get final 10% of steps average for each run
summary_data = []
for (algo, lam), group in full_df.groupby(["algorithm", "lambda"]):
    if algo != "ESAI": continue
    # Take last 50 episodes
    last_idx = max(0, len(group) - 50)
    final_reward = group["reward"].iloc[last_idx:].mean()
    final_pr = group["pr"].iloc[last_idx:].mean() if "pr" in group.columns else 0
    final_ar = group["ar"].iloc[last_idx:].mean() if "ar" in group.columns else 0
    summary_data.append({
        "lambda": lam,
        "Final Reward": final_reward,
        "Final PR": final_pr,
        "Final AR": final_ar
    })

summary_df = pd.DataFrame(summary_data)

if not summary_df.empty:
    # Reward vs Lambda
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary_df, x="lambda", y="Final Reward", s=100)
    sns.lineplot(data=summary_df, x="lambda", y="Final Reward")
    plt.xscale('log')
    plt.title("Lambda Sensitivity: Final Reward")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/lambda_sensitivity_reward.png", dpi=300)
    plt.close()

    # PR vs Lambda
    if "Final PR" in summary_df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=summary_df, x="lambda", y="Final PR", s=100, color='green')
        sns.lineplot(data=summary_df, x="lambda", y="Final PR", color='green')
        plt.xscale('log')
        plt.title("Lambda Sensitivity: Prosocial Ratio")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/lambda_sensitivity_pr.png", dpi=300)
        plt.close()

    # Pareto: Reward vs PR
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=summary_df, x="Final PR", y="Final Reward", hue="lambda", palette="viridis", s=150)
    plt.title("Pareto Frontier: Reward vs Prosocial Ratio")
    plt.xlabel("Prosocial Ratio (Safety)")
    plt.ylabel("Task Reward (Performance)")
    plt.grid(True, alpha=0.3)
    
    # Add CPO baseline point if available
    cpo_group = full_df[full_df["algorithm"] == "CPO Baseline"]
    if not cpo_group.empty:
        last_idx = max(0, len(cpo_group) - 50)
        cpo_rew = cpo_group["reward"].iloc[last_idx:].mean()
        # Assume CPO PR is unknown or need to find it?
        # If metric not in json, we can't plot it on X-axis.
        plt.axhline(y=cpo_rew, color='red', linestyle='--', label="CPO Baseline Reward")
        plt.legend()
        
    plt.savefig(f"{OUTPUT_DIR}/pareto_frontier.png", dpi=300)
    plt.close()

print(f"Artifacts generated in {OUTPUT_DIR}")
