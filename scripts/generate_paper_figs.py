#!/usr/bin/env python3
"""Generate publication-ready figures using only the specified sources:
- /Users/haxx_sh/Desktop/MARL/MAR_results_moral_temptation
- /Users/haxx_sh/Desktop/MARL/ppo&cpo(trained-logs).txt
- /Users/haxx_sh/Desktop/MARL/moral_temptation 3/ppo_baseline_paper
- /Users/haxx_sh/Desktop/MARL/LogsOfESAI.txt

Outputs are written to `results/paper_figures/` as PNG and PDF.

This script intentionally restricts itself to those paths (no external data).
"""
from pathlib import Path
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# Use seaborn style when available, otherwise fall back to a stable default
try:
    plt.style.use('seaborn-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn')
    except Exception:
        plt.style.use('ggplot')

# Hard-coded allowed paths (as requested)
ROOT = Path('/Users/haxx_sh/Desktop/MARL')
ESAI_LOG = ROOT / 'LogsOfESAI.txt'
PPOCPO_LOG = ROOT / 'ppo&cpo(trained-logs).txt'
ESAI_RESULTS_DIR = ROOT / 'MAR_results_moral_temptation'
PPO_BASELINE_DIR = ROOT / 'moral_temptation 3' / 'ppo_baseline_paper'
OUT_DIR = ROOT / 'results' / 'paper_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_diag_blocks(text):
    """Return list of dicts with step, pr, E_norm, r_ext, raw block"""
    blocks = []
    # Split on lines that start and end with many '=' or DIAG markers
    parts = re.split(r"\n={10,}\n", text)
    for p in parts:
        if '[DIAG]' in p:
            step_m = re.search(r"Step\s+([0-9,]+)", p)
            pr_m = re.search(r"PR:\s+([0-9.]+)", p)
            e_m = re.search(r"\|\|E\|\|:\s*([0-9.]+)", p)
            r_m = re.search(r"Rewards:\s+r_ext=([-0-9.\.]+)", p)
            step = int(step_m.group(1).replace(',','')) if step_m else None
            pr = float(pr_m.group(1)) if pr_m else None
            e = float(e_m.group(1)) if e_m else None
            r = float(r_m.group(1)) if r_m else None
            blocks.append({'step': step, 'pr': pr, 'E_norm': e, 'r_ext': r, 'raw': p})
    return pd.DataFrame(blocks)


def load_esai_logs():
    if not ESAI_LOG.exists():
        print('Warning: LogsOfESAI.txt not found at', ESAI_LOG)
        return None
    txt = ESAI_LOG.read_text()
    return parse_diag_blocks(txt)


def load_ppocpo_logs():
    if not PPOCPO_LOG.exists():
        print('Warning: ppo&cpo logs not found at', PPOCPO_LOG)
        return None
    txt = PPOCPO_LOG.read_text()
    return parse_diag_blocks(txt)


def collect_final_metrics_from_dir(base_dir):
    """Search for per-seed config.json under base_dir and collect final metrics if present."""
    results = {}
    b = Path(base_dir)
    if not b.exists():
        return results
    for seed_dir in b.rglob('seed_*'):
        # try to find a saved config.json and optionally a summary
        cfg = seed_dir / 'config.json'
        if cfg.exists():
            try:
                cfgj = json.load(open(cfg))
            except Exception:
                cfgj = None
            results[seed_dir.name] = {'config': cfgj, 'path': str(seed_dir)}
    return results


def plot_pr(df_esai, df_ppobaseline, out_dir):
    fig, ax = plt.subplots(figsize=(6,4))
    if df_esai is not None and not df_esai.empty:
        d = df_esai.dropna(subset=['step','pr']).sort_values('step')
        ax.plot(d['step'], d['pr'], label='ESAI (LogsOfESAI)', lw=2)
    if df_ppobaseline is not None and not df_ppobaseline.empty:
        d2 = df_ppobaseline.dropna(subset=['step','pr']).sort_values('step')
        ax.plot(d2['step'], d2['pr'], label='PPO/CPO (ppo&cpo logs)', lw=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Prosocial Rate (PR)')
    ax.set_ylim(-0.01, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'pr_vs_steps.png', dpi=300)
    fig.savefig(out_dir / 'pr_vs_steps.pdf')
    plt.close(fig)


def plot_reward(df_esai, df_ppobaseline, out_dir):
    fig, ax = plt.subplots(figsize=(6,4))
    if df_esai is not None and not df_esai.empty:
        d = df_esai.dropna(subset=['step','r_ext']).sort_values('step')
        ax.plot(d['step'], d['r_ext'], label='ESAI (LogsOfESAI)', lw=2)
    if df_ppobaseline is not None and not df_ppobaseline.empty:
        d2 = df_ppobaseline.dropna(subset=['step','r_ext']).sort_values('step')
        ax.plot(d2['step'], d2['r_ext'], label='PPO/CPO (ppo&cpo logs)', lw=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('External reward')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'reward_vs_steps.png', dpi=300)
    fig.savefig(out_dir / 'reward_vs_steps.pdf')
    plt.close(fig)


def plot_forecaster_diff(df_esai, out_dir):
    fig, ax = plt.subplots(figsize=(6,4))
    if df_esai is None or df_esai.empty:
        return
    diffs = []
    steps = []
    for _, row in df_esai.iterrows():
        raw = row.get('raw','')
        m = re.search(r"HELP\(4\)=([0-9.]+), STEAL\(5\)=([0-9.]+)", raw)
        if m and row['step'] is not None:
            h = float(m.group(1)); s = float(m.group(2))
            steps.append(row['step']); diffs.append(s - h)
    if steps:
        ax.plot(steps, diffs, label='Forecaster STEAL-HELP ||E|| diff')
    ax.set_xlabel('Training step')
    ax.set_ylabel('||E^(STEAL)|| - ||E^(HELP)||')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'forecaster_diff.png', dpi=300)
    fig.savefig(out_dir / 'forecaster_diff.pdf')
    plt.close(fig)


def bar_final_comparison(df_esai, df_ppobaseline, out_dir):
    # Take last available PR and reward
    es_pr = df_esai['pr'].dropna().iloc[-1] if (df_esai is not None and not df_esai['pr'].dropna().empty) else np.nan
    es_r = df_esai['r_ext'].dropna().iloc[-1] if (df_esai is not None and not df_esai['r_ext'].dropna().empty) else np.nan
    ppo_pr = df_ppobaseline['pr'].dropna().iloc[-1] if (df_ppobaseline is not None and not df_ppobaseline['pr'].dropna().empty) else np.nan
    ppo_r = df_ppobaseline['r_ext'].dropna().iloc[-1] if (df_ppobaseline is not None and not df_ppobaseline['r_ext'].dropna().empty) else np.nan

    labels = ['ESAI', 'PPO/CPO']
    pr_vals = [es_pr, ppo_pr]
    r_vals = [es_r, ppo_r]

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(labels, pr_vals, color=['C0','C1'])
    axes[0].set_ylim(0,1)
    axes[0].set_ylabel('Final PR')

    axes[1].bar(labels, r_vals, color=['C0','C1'])
    axes[1].set_ylabel('Final external reward')

    fig.tight_layout()
    fig.savefig(out_dir / 'final_comparison.png', dpi=300)
    fig.savefig(out_dir / 'final_comparison.pdf')
    plt.close(fig)


def main():
    df_esai = load_esai_logs()
    df_ppocpo = load_ppocpo_logs()

    # Also inspect MAR_results folder for run names (no extra parsing beyond listing)
    runs = [p.name for p in ESAI_RESULTS_DIR.iterdir() if p.is_dir()]
    with open(OUT_DIR / 'manifest.txt', 'w') as f:
        f.write('Using only the following input sources:\n')
        f.write(str(ESAI_LOG) + '\n')
        f.write(str(PPOCPO_LOG) + '\n')
        f.write(str(ESAI_RESULTS_DIR) + '\n')
        f.write(str(PPO_BASELINE_DIR) + '\n')
        f.write('\nDetected subruns in MAR_results_moral_temptation:\n')
        for r in runs:
            f.write('- '+ r + '\n')

    plot_pr(df_esai, df_ppocpo, OUT_DIR)
    plot_reward(df_esai, df_ppocpo, OUT_DIR)
    plot_forecaster_diff(df_esai, OUT_DIR)
    bar_final_comparison(df_esai, df_ppocpo, OUT_DIR)

    print('Figures written to', OUT_DIR)

if __name__ == '__main__':
    main()
