"""Analyze IAE-harm correlation."""
import glob
import json
import gzip
import numpy as np
import scipy.stats as st
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


def pearson_for_run(run_dir):
    xs, ys = [], []
    for f in Path(run_dir).glob("per_step_eval*.jsonl*"):
        for rec in iter_jsonl(f):
            if rec.get("E_norm_t") is not None and rec.get("harm_t") is not None:
                xs.append(rec["E_norm_t"])
                ys.append(rec["harm_t"])
    if len(xs) < 3:
        return np.nan, 1.0
    return st.pearsonr(xs, ys)


def aggregate(env):
    rs, ps = [], []
    pattern = f"data/results/logs/{env}/corr_{env}_bias0/seed_*/"
    for p in sorted(glob.glob(pattern)):
        r, pval = pearson_for_run(p)
        if not np.isnan(r):
            rs.append(r)
            ps.append(pval)
    
    if len(rs) == 0:
        print(f"{env}: No valid data")
        return
    
    mean_r = np.mean(rs)
    se = st.sem(rs)
    ci = (mean_r - 1.96 * se, mean_r + 1.96 * se)
    frac_sig = (np.array(ps) < 1e-3).mean()
    
    print(f"{env:20s} r={mean_r:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}] frac(p<1e-3)={frac_sig:.2f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("IAE-Harm Correlation Analysis")
    print("="*60 + "\n")
    
    envs = ["moral_temptation", "social_distress", "mpe", "overcooked", "ssd"]
    for env in envs:
        aggregate(env)
    
    print("\n" + "="*60)