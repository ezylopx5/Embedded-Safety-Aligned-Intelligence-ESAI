# Generate Paper Figures

This folder contains `generate_paper_figs.py` which creates publication-ready figures using only the following sources (as requested):

- `/Users/haxx_sh/Desktop/MARL/MAR_results_moral_temptation`
- `/Users/haxx_sh/Desktop/MARL/ppo&cpo(trained-logs).txt`
- `/Users/haxx_sh/Desktop/MARL/moral_temptation 3/ppo_baseline_paper`
- `/Users/haxx_sh/Desktop/MARL/LogsOfESAI.txt`

Outputs are written to `results/paper_figures/` as PNG and PDF.

Usage:

```bash
python3 scripts/generate_paper_figs.py
```

No arguments are required—the script uses the strict file locations above.

Produced figures:

- `pr_vs_steps.png` / `.pdf` — Prosocial Rate over training steps (ESAI vs PPO/CPO)
- `reward_vs_steps.png` / `.pdf` — External reward over training steps
- `forecaster_diff.png` / `.pdf` — Forecaster STEAL-HELP ||E|| difference over steps (ESAI only)
- `final_comparison.png` / `.pdf` — Bar chart comparing final PR and reward
- `manifest.txt` — Lists which subruns in `MAR_results_moral_temptation` were detected

If you need different file locations or additional plots (e.g., per-seed aggregates), tell me which exact files to include and I'll update the script accordingly.