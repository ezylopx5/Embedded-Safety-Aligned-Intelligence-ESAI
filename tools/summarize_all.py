"""
Generate a summary report of all experimental results.
"""

import os
import json
import glob
import pandas as pd
import numpy as np


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_csv_metric(path, metric='PR'):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if len(df) > 0:
            return df.iloc[0][metric]
    return None


def summarize():
    print("\n" + "="*80)
    print(" "*25 + "ESAI-v3 RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # 1. Correlation
    print("━"*80)
    print("1. IAE-HARM CORRELATION")
    print("━"*80)
    os.system("python tools/analyze_correlation.py 2>/dev/null")
    
    # 2. Intervention
    print("\n" + "━"*80)
    print("2. CAUSAL INTERVENTION")
    print("━"*80)
    os.system("python tools/analyze_intervention.py 2>/dev/null")
    
    # 3. Ablations
    print("\n" + "━"*80)
    print("3. ARCHITECTURAL ABLATIONS")
    print("━"*80)
    os.system("python tools/analyze_ablations.py 2>/dev/null")
    
    # 4. Scaling
    print("\n" + "━"*80)
    print("4. ZERO-SHOT SCALING")
    print("━"*80)
    os.system("python tools/analyze_scaling.py 2>/dev/null")
    
    # 5. Bias
    print("\n" + "━"*80)
    print("5. BIAS MITIGATION")
    print("━"*80)
    os.system("python tools/analyze_bias.py 2>/dev/null")
    
    # 6. Wall-clock
    print("\n" + "━"*80)
    print("6. WALL-CLOCK OVERHEAD")
    print("━"*80)
    os.system("python tools/analyze_wallclock.py 2>/dev/null")
    
    print("\n" + "="*80)
    print(" "*28 + "END OF SUMMARY")
    print("="*80 + "\n")


if __name__ == '__main__':
    summarize()