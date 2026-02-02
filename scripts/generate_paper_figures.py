#!/usr/bin/env python
"""
Generate publication-quality figures for ALA @ AAMAS 2026 paper.
Compares ESAI-v3 against PPO baseline on MoralTemptation environment.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['figure.figsize'] = (8, 6)
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# Color scheme
COLORS = {
    'ppo': '#E74C3C',      # Red
    'esai': '#2ECC71',     # Green
    'ablation_noatt': '#3498DB',   # Blue
    'ablation_noheb': '#9B59B6',   # Purple
    'ablation_noar': '#F39C12',    # Orange
}

LABELS = {
    'ppo': 'PPO Baseline',
    'esai': 'ESAI-v3 (λ=5)',
    'ablation_noatt': 'ESAI (no Attention)',
    'ablation_noheb': 'ESAI (no Hebbian)',
    'ablation_noar': 'ESAI (no AR)',
}


def load_metrics(exp_dir: str) -> Dict[str, np.ndarray]:
    """Load metrics from all seeds in an experiment directory."""
    metrics = {
        'rewards': [],
        'prosocial_ratios': [],
        'alignment_regrets': [],
        'iae_norms': [],
        'steps': None,
    }
    
    seed_dirs = sorted(glob.glob(os.path.join(exp_dir, 'seed_*')))
    
    for seed_dir in seed_dirs:
        metrics_file = os.path.join(seed_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                data = json.load(f)
            
            if 'eval_mean_reward' in data:
                metrics['rewards'].append(data['eval_mean_reward'])
            elif 'reward' in data:
                metrics['rewards'].append(data['reward'])
                
            if 'eval_prosocial_ratio' in data:
                metrics['prosocial_ratios'].append(data['eval_prosocial_ratio'])
            elif 'prosocial_ratio' in data:
                metrics['prosocial_ratios'].append(data['prosocial_ratio'])
                
            if 'alignment_regret' in data:
                metrics['alignment_regrets'].append(data['alignment_regret'])
                
            if 'iae_norm' in data:
                metrics['iae_norms'].append(data['iae_norm'])
    
    # Convert to arrays with proper shape
    for key in ['rewards', 'prosocial_ratios', 'alignment_regrets', 'iae_norms']:
        if metrics[key]:
            # Pad to same length
            max_len = max(len(x) for x in metrics[key])
            padded = []
            for x in metrics[key]:
                if len(x) < max_len:
                    x = list(x) + [x[-1]] * (max_len - len(x))
                padded.append(x)
            metrics[key] = np.array(padded)
    
    return metrics


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply smoothing to data."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_with_ci(ax, x, data, color, label, alpha=0.2):
    """Plot mean with 95% confidence interval."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n = data.shape[0]
    ci = 1.96 * std / np.sqrt(n)  # 95% CI
    
    ax.plot(x[:len(mean)], mean, color=color, label=label, linewidth=2)
    ax.fill_between(x[:len(mean)], mean - ci, mean + ci, color=color, alpha=alpha)


def figure1_learning_curves(results_dir: str, output_dir: str):
    """Figure 1: Learning curves comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Load data
    ppo_metrics = load_metrics(os.path.join(results_dir, 'ppo_baseline_paper'))
    esai_metrics = load_metrics(os.path.join(results_dir, 'esaiv3_lambda5_paper'))
    
    if ppo_metrics['rewards'] is not None and len(ppo_metrics['rewards']) > 0:
        x = np.arange(len(ppo_metrics['rewards'][0])) * 25000  # eval_interval
        plot_with_ci(ax, x, ppo_metrics['rewards'], COLORS['ppo'], LABELS['ppo'])
    
    if esai_metrics['rewards'] is not None and len(esai_metrics['rewards']) > 0:
        x = np.arange(len(esai_metrics['rewards'][0])) * 25000
        plot_with_ci(ax, x, esai_metrics['rewards'], COLORS['esai'], LABELS['esai'])
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Learning Curves: ESAI-v3 vs PPO Baseline')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500000)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_learning_curves.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig1_learning_curves.png'))
    plt.close()
    print("✓ Figure 1: Learning curves saved")


def figure2_prosocial_ratio(results_dir: str, output_dir: str):
    """Figure 2: Prosocial Ratio - THE KEY RESULT."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ppo_metrics = load_metrics(os.path.join(results_dir, 'ppo_baseline_paper'))
    esai_metrics = load_metrics(os.path.join(results_dir, 'esaiv3_lambda5_paper'))
    
    if ppo_metrics['prosocial_ratios'] is not None and len(ppo_metrics['prosocial_ratios']) > 0:
        x = np.arange(len(ppo_metrics['prosocial_ratios'][0])) * 25000
        plot_with_ci(ax, x, ppo_metrics['prosocial_ratios'], COLORS['ppo'], LABELS['ppo'])
    
    if esai_metrics['prosocial_ratios'] is not None and len(esai_metrics['prosocial_ratios']) > 0:
        x = np.arange(len(esai_metrics['prosocial_ratios'][0])) * 25000
        plot_with_ci(ax, x, esai_metrics['prosocial_ratios'], COLORS['esai'], LABELS['esai'])
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Fully Prosocial')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Prosocial Ratio')
    ax.set_title('Prosocial Behavior: ESAI-v3 vs PPO Baseline')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500000)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_prosocial_ratio.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig2_prosocial_ratio.png'))
    plt.close()
    print("✓ Figure 2: Prosocial ratio saved")


def figure3_alignment_regret(results_dir: str, output_dir: str):
    """Figure 3: Alignment Regret dynamics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    esai_metrics = load_metrics(os.path.join(results_dir, 'esaiv3_lambda5_paper'))
    
    if esai_metrics['alignment_regrets'] is not None and len(esai_metrics['alignment_regrets']) > 0:
        x = np.arange(len(esai_metrics['alignment_regrets'][0])) * 25000
        plot_with_ci(ax, x, esai_metrics['alignment_regrets'], COLORS['esai'], 'Alignment Regret')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Alignment Regret (AR)')
    ax.set_title('Alignment Regret Dynamics During Training')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 500000)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_alignment_regret.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig3_alignment_regret.png'))
    plt.close()
    print("✓ Figure 3: Alignment regret saved")


def figure4_bar_comparison(results_dir: str, output_dir: str):
    """Figure 4: Final performance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['PPO', 'ESAI-v3']
    
    # Load final metrics
    ppo_metrics = load_metrics(os.path.join(results_dir, 'ppo_baseline_paper'))
    esai_metrics = load_metrics(os.path.join(results_dir, 'esaiv3_lambda5_paper'))
    
    # Prosocial Ratio
    ax = axes[0]
    pr_means = []
    pr_stds = []
    
    if ppo_metrics['prosocial_ratios'] is not None and len(ppo_metrics['prosocial_ratios']) > 0:
        final_pr = ppo_metrics['prosocial_ratios'][:, -1]
        pr_means.append(np.mean(final_pr))
        pr_stds.append(np.std(final_pr))
    else:
        pr_means.append(0)
        pr_stds.append(0)
        
    if esai_metrics['prosocial_ratios'] is not None and len(esai_metrics['prosocial_ratios']) > 0:
        final_pr = esai_metrics['prosocial_ratios'][:, -1]
        pr_means.append(np.mean(final_pr))
        pr_stds.append(np.std(final_pr))
    else:
        pr_means.append(0)
        pr_stds.append(0)
    
    bars = ax.bar(methods, pr_means, yerr=pr_stds, capsize=5,
                  color=[COLORS['ppo'], COLORS['esai']], edgecolor='black')
    ax.set_ylabel('Prosocial Ratio')
    ax.set_title('Final Prosocial Ratio')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, mean in zip(bars, pr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=12)
    
    # Reward
    ax = axes[1]
    r_means = []
    r_stds = []
    
    if ppo_metrics['rewards'] is not None and len(ppo_metrics['rewards']) > 0:
        final_r = ppo_metrics['rewards'][:, -1]
        r_means.append(np.mean(final_r))
        r_stds.append(np.std(final_r))
    else:
        r_means.append(0)
        r_stds.append(0)
        
    if esai_metrics['rewards'] is not None and len(esai_metrics['rewards']) > 0:
        final_r = esai_metrics['rewards'][:, -1]
        r_means.append(np.mean(final_r))
        r_stds.append(np.std(final_r))
    else:
        r_means.append(0)
        r_stds.append(0)
    
    bars = ax.bar(methods, r_means, yerr=r_stds, capsize=5,
                  color=[COLORS['ppo'], COLORS['esai']], edgecolor='black')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Final Episode Reward')
    
    for bar, mean in zip(bars, r_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_bar_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig4_bar_comparison.png'))
    plt.close()
    print("✓ Figure 4: Bar comparison saved")


def generate_latex_table(results_dir: str, output_dir: str):
    """Generate LaTeX table for paper."""
    ppo_metrics = load_metrics(os.path.join(results_dir, 'ppo_baseline_paper'))
    esai_metrics = load_metrics(os.path.join(results_dir, 'esaiv3_lambda5_paper'))
    
    rows = []
    
    # PPO
    if ppo_metrics['rewards'] is not None and len(ppo_metrics['rewards']) > 0:
        pr = ppo_metrics['prosocial_ratios'][:, -1] if ppo_metrics['prosocial_ratios'] is not None else [0]
        r = ppo_metrics['rewards'][:, -1]
        rows.append(f"PPO Baseline & ${np.mean(r):.1f} \\pm {np.std(r):.1f}$ & ${np.mean(pr):.3f} \\pm {np.std(pr):.3f}$ & N/A \\\\")
    
    # ESAI
    if esai_metrics['rewards'] is not None and len(esai_metrics['rewards']) > 0:
        pr = esai_metrics['prosocial_ratios'][:, -1] if esai_metrics['prosocial_ratios'] is not None else [0]
        r = esai_metrics['rewards'][:, -1]
        ar = esai_metrics['alignment_regrets'][:, -1] if esai_metrics['alignment_regrets'] is not None else [0]
        rows.append(f"ESAI-v3 ($\\lambda=5$) & ${np.mean(r):.1f} \\pm {np.std(r):.1f}$ & $\\mathbf{{{np.mean(pr):.3f}}} \\pm {np.std(pr):.3f}$ & ${np.mean(ar):.3f} \\pm {np.std(ar):.3f}$ \\\\")
    
    table = """
\\begin{table}[h]
\\centering
\\caption{Performance comparison on MoralTemptation environment (5 seeds, 500k steps)}
\\label{tab:results}
\\begin{tabular}{lccc}
\\toprule
Method & Reward & Prosocial Ratio & Alignment Regret \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(os.path.join(output_dir, 'table_results.tex'), 'w') as f:
        f.write(table)
    print("✓ LaTeX table saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/moral_temptation',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='paper/figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication Figures for ALA @ AAMAS 2026")
    print("=" * 60)
    
    figure1_learning_curves(args.results_dir, args.output_dir)
    figure2_prosocial_ratio(args.results_dir, args.output_dir)
    figure3_alignment_regret(args.results_dir, args.output_dir)
    figure4_bar_comparison(args.results_dir, args.output_dir)
    generate_latex_table(args.results_dir, args.output_dir)
    
    print("=" * 60)
    print(f"All figures saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
