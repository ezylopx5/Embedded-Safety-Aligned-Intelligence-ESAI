#!/usr/bin/env python3
"""
Generate publication-quality figures for ALA @ AAMAS 2026 paper.
Plots learning curves, prosocial ratios, alignment regret, and ablation studies.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Use publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# Color palette (colorblind-friendly)
COLORS = {
    'esai': '#2ecc71',      # Green - our method
    'ppo': '#e74c3c',       # Red - baseline
    'cpo': '#3498db',       # Blue - constrained baseline
    'reward_shaping': '#9b59b6',  # Purple
    'no_attention': '#f39c12',    # Orange
    'no_hebbian': '#1abc9c',      # Teal
    'no_regret': '#e67e22',       # Dark orange
    'no_diffusion': '#34495e',    # Dark gray
}

LABELS = {
    'esai': 'ESAI-v3 (Ours)',
    'ppo': 'PPO Baseline',
    'cpo': 'CPO',
    'reward_shaping': 'Reward Shaping',
    'no_attention': 'w/o Attention',
    'no_hebbian': 'w/o Hebbian',
    'no_regret': 'w/o Alignment Regret',
    'no_diffusion': 'w/o Diffusion',
}


def load_metrics(results_dir: Path, experiment_name: str, seeds: List[int]) -> Dict[str, np.ndarray]:
    """Load metrics from multiple seeds and compute mean/std."""
    all_metrics = {
        'episode_reward': [],
        'prosocial_ratio': [],
        'alignment_regret': [],
        'steps': [],
    }
    
    for seed in seeds:
        metrics_file = results_dir / experiment_name / f'seed_{seed}' / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                for key in all_metrics:
                    if key in data:
                        all_metrics[key].append(data[key])
    
    # Convert to numpy and compute stats
    result = {}
    for key, values in all_metrics.items():
        if values:
            # Pad to same length if needed
            max_len = max(len(v) for v in values)
            padded = []
            for v in values:
                if len(v) < max_len:
                    v = v + [v[-1]] * (max_len - len(v))  # Pad with last value
                padded.append(v)
            arr = np.array(padded)
            result[f'{key}_mean'] = np.mean(arr, axis=0)
            result[f'{key}_std'] = np.std(arr, axis=0)
            result[f'{key}_raw'] = arr
    
    return result


def smooth_curve(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Figure 1: Learning curves comparing ESAI-v3 vs baselines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    experiments = [
        ('esai_lambda5', 'esai'),
        ('ppo_baseline', 'ppo'),
    ]
    
    for exp_name, color_key in experiments:
        metrics = load_metrics(results_dir, exp_name, seeds)
        if f'episode_reward_mean' in metrics:
            mean = smooth_curve(metrics['episode_reward_mean'])
            std = smooth_curve(metrics['episode_reward_std'])
            steps = np.arange(len(mean)) * 1000  # Assuming 1000 steps per log
            
            ax.plot(steps, mean, color=COLORS[color_key], label=LABELS[color_key], linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[color_key], alpha=0.2)
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Learning Curves on MoralTemptation Environment')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_learning_curves.pdf')
    plt.savefig(output_dir / 'fig1_learning_curves.png')
    plt.close()
    print(f"✓ Saved fig1_learning_curves.pdf")


def plot_prosocial_ratio(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Figure 2: Prosocial ratio over training - THE KEY RESULT."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    experiments = [
        ('esai_lambda5', 'esai'),
        ('ppo_baseline', 'ppo'),
    ]
    
    for exp_name, color_key in experiments:
        metrics = load_metrics(results_dir, exp_name, seeds)
        if 'prosocial_ratio_mean' in metrics:
            mean = smooth_curve(metrics['prosocial_ratio_mean'])
            std = smooth_curve(metrics['prosocial_ratio_std'])
            steps = np.arange(len(mean)) * 1000
            
            ax.plot(steps, mean, color=COLORS[color_key], label=LABELS[color_key], linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[color_key], alpha=0.2)
    
    # Add reference line at 0.5 (random policy)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Policy')
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Prosocial Ratio (Help / Total Engagements)')
    ax.set_title('Emergent Prosocial Behavior in MoralTemptation')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_prosocial_ratio.pdf')
    plt.savefig(output_dir / 'fig2_prosocial_ratio.png')
    plt.close()
    print(f"✓ Saved fig2_prosocial_ratio.pdf")


def plot_alignment_regret(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Figure 3: Alignment regret dynamics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics = load_metrics(results_dir, 'esai_lambda5', seeds)
    if 'alignment_regret_mean' in metrics:
        mean = smooth_curve(metrics['alignment_regret_mean'])
        std = smooth_curve(metrics['alignment_regret_std'])
        steps = np.arange(len(mean)) * 1000
        
        ax.plot(steps, mean, color=COLORS['esai'], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS['esai'], alpha=0.2)
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Alignment Regret')
    ax.set_title('Alignment Regret Dynamics (ESAI-v3)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_alignment_regret.pdf')
    plt.savefig(output_dir / 'fig3_alignment_regret.png')
    plt.close()
    print(f"✓ Saved fig3_alignment_regret.pdf")


def plot_ablation_study(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Figure 4: Ablation study showing contribution of each component."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ablations = [
        ('esai_lambda5', 'esai'),
        ('ablation_no_attention', 'no_attention'),
        ('ablation_no_hebbian', 'no_hebbian'),
        ('ablation_no_regret', 'no_regret'),
    ]
    
    # Plot reward curves
    for exp_name, color_key in ablations:
        metrics = load_metrics(results_dir, exp_name, seeds)
        if 'episode_reward_mean' in metrics:
            mean = smooth_curve(metrics['episode_reward_mean'])
            steps = np.arange(len(mean)) * 1000
            ax1.plot(steps, mean, color=COLORS[color_key], label=LABELS[color_key], linewidth=2)
    
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('(a) Learning Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot prosocial ratio curves
    for exp_name, color_key in ablations:
        metrics = load_metrics(results_dir, exp_name, seeds)
        if 'prosocial_ratio_mean' in metrics:
            mean = smooth_curve(metrics['prosocial_ratio_mean'])
            steps = np.arange(len(mean)) * 1000
            ax2.plot(steps, mean, color=COLORS[color_key], label=LABELS[color_key], linewidth=2)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Environment Steps')
    ax2.set_ylabel('Prosocial Ratio')
    ax2.set_title('(b) Prosocial Behavior')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_ablation_study.pdf')
    plt.savefig(output_dir / 'fig4_ablation_study.png')
    plt.close()
    print(f"✓ Saved fig4_ablation_study.pdf")


def plot_bar_comparison(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Figure 5: Bar chart comparing final performance metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    experiments = [
        ('ppo_baseline', 'ppo', 'PPO'),
        ('esai_lambda5', 'esai', 'ESAI-v3'),
    ]
    
    # Collect final metrics
    final_rewards = []
    final_pr = []
    labels = []
    colors = []
    
    for exp_name, color_key, label in experiments:
        metrics = load_metrics(results_dir, exp_name, seeds)
        if 'episode_reward_mean' in metrics:
            # Take last 10% of training as final performance
            mean = metrics['episode_reward_mean']
            final_idx = int(len(mean) * 0.9)
            final_rewards.append((np.mean(mean[final_idx:]), np.std(mean[final_idx:])))
        else:
            final_rewards.append((0, 0))
            
        if 'prosocial_ratio_mean' in metrics:
            mean = metrics['prosocial_ratio_mean']
            final_idx = int(len(mean) * 0.9)
            final_pr.append((np.mean(mean[final_idx:]), np.std(mean[final_idx:])))
        else:
            final_pr.append((0, 0))
            
        labels.append(label)
        colors.append(COLORS[color_key])
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Reward bar chart
    means = [r[0] for r in final_rewards]
    stds = [r[1] for r in final_rewards]
    bars1 = ax1.bar(x, means, width, yerr=stds, color=colors, capsize=5, edgecolor='black')
    ax1.set_ylabel('Final Episode Reward')
    ax1.set_title('(a) Task Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Prosocial ratio bar chart
    means = [r[0] for r in final_pr]
    stds = [r[1] for r in final_pr]
    bars2 = ax2.bar(x, means, width, yerr=stds, color=colors, capsize=5, edgecolor='black')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_ylabel('Final Prosocial Ratio')
    ax2.set_title('(b) Alignment Quality')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_bar_comparison.pdf')
    plt.savefig(output_dir / 'fig5_bar_comparison.png')
    plt.close()
    print(f"✓ Saved fig5_bar_comparison.pdf")


def generate_table(results_dir: Path, output_dir: Path, seeds: List[int] = [1, 2, 3, 4, 5]):
    """Generate LaTeX table for paper."""
    experiments = [
        ('ppo_baseline', 'PPO'),
        ('esai_lambda5', 'ESAI-v3 (Ours)'),
    ]
    
    table_data = []
    for exp_name, label in experiments:
        metrics = load_metrics(results_dir, exp_name, seeds)
        row = {'Method': label}
        
        if 'episode_reward_mean' in metrics:
            mean = metrics['episode_reward_mean']
            final_idx = int(len(mean) * 0.9)
            row['Reward'] = f"{np.mean(mean[final_idx:]):.2f} ± {np.std(mean[final_idx:]):.2f}"
        else:
            row['Reward'] = '-'
            
        if 'prosocial_ratio_mean' in metrics:
            mean = metrics['prosocial_ratio_mean']
            final_idx = int(len(mean) * 0.9)
            row['PR'] = f"{np.mean(mean[final_idx:]):.3f} ± {np.std(mean[final_idx:]):.3f}"
        else:
            row['PR'] = '-'
            
        table_data.append(row)
    
    # Generate LaTeX
    latex = r"""
\begin{table}[t]
\centering
\caption{Comparison on MoralTemptation environment (500K steps, 5 seeds)}
\label{tab:main_results}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Reward} $\uparrow$ & \textbf{Prosocial Ratio} $\uparrow$ \\
\midrule
"""
    for row in table_data:
        latex += f"{row['Method']} & {row['Reward']} & {row['PR']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_dir / 'table1_main_results.tex', 'w') as f:
        f.write(latex)
    print(f"✓ Saved table1_main_results.tex")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results-dir', type=str, default='results/moral_temptation',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='paper/figures',
                        help='Directory to save figures')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Seeds to include')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating paper figures...")
    print(f"   Results: {results_dir}")
    print(f"   Output:  {output_dir}")
    print(f"   Seeds:   {args.seeds}\n")
    
    # Generate all figures
    plot_learning_curves(results_dir, output_dir, args.seeds)
    plot_prosocial_ratio(results_dir, output_dir, args.seeds)
    plot_alignment_regret(results_dir, output_dir, args.seeds)
    plot_ablation_study(results_dir, output_dir, args.seeds)
    plot_bar_comparison(results_dir, output_dir, args.seeds)
    generate_table(results_dir, output_dir, args.seeds)
    
    print(f"\n✅ All figures saved to {output_dir}/")


if __name__ == '__main__':
    main()
