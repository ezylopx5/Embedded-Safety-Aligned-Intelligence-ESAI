"""
Logging utilities for ESAI-v3 experiments.
Handles per-step JSONL logs and aggregated CSV metrics.
"""

import os
import json
import csv
import gzip
import math
from collections import deque
from pathlib import Path


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    """Save object as JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def bucket_similarity(sim: float):
    """
    Bucket similarity score [0,1] into 5 bins.
    Returns None if sim is invalid.
    """
    if sim is None or math.isnan(sim):
        return None
    return min(4, max(0, int(sim * 5.0)))


class EvalLogger:
    """
    Logger for evaluation episodes.
    
    Writes:
    - per_step_eval[_{exp_tag}].jsonl.gz: per-step data
    - eval_metrics[_{exp_tag}].csv: aggregated episode metrics
    
    Tracks:
    - PR (Prosocial Ratio)
    - AR (Alignment Regret)
    - ESI (Embedding Stability Index)
    - IPA (IAE Predictive Accuracy)
    """
    
    def __init__(self, log_dir, exp_tag=None, esi_window=20, eps=1e-8):
        ensure_dir(log_dir)
        self.log_dir = log_dir
        self.exp_tag = exp_tag
        suffix = f"_{exp_tag}" if exp_tag else ""
        
        self.step_path = os.path.join(log_dir, f"per_step_eval{suffix}.jsonl.gz")
        self.eval_csv = os.path.join(log_dir, f"eval_metrics{suffix}.csv")
        self._step_f = gzip.open(self.step_path, "wt")
        
        # Accumulators
        self.total_steps = 0
        self.total_decisions = 0
        self.help_count = 0
        self.ar_sum = 0.0
        self.ar_count = 0
        
        # ESI (sliding window stability)
        self.esi_window = esi_window
        self._e_hist = deque(maxlen=esi_window)
        self._esi_sum = 0.0
        self._esi_count = 0
        
        # IPA (predictive accuracy)
        self._ipa_sum = 0.0
        self._ipa_count = 0
        self._eps = eps
        self._pending_pred_norm = None
    
    def _esi_from_hist(self):
        """Compute ESI from sliding window of E norms."""
        if len(self._e_hist) < 2:
            return None
        vals = list(self._e_hist)
        mu = sum(vals) / len(vals)
        if mu <= self._eps:
            return None
        var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
        cv = math.sqrt(var) / (mu + self._eps)
        return 1.0 / (1.0 + cv)
    
    def log_step(self, t, episode_id, a_t=None, r_ext=None, E_vec=None, E_norm=None,
                 harm_t=None, pr_flag=None, ar_t=None, sim=None, ipa_t=None,
                 e_pred_next=None, extra: dict = None):
        """
        Log a single step.
        
        Args:
            t: timestep
            episode_id: episode number
            a_t: action taken
            r_ext: external reward
            E_vec: IAE vector (will compute norm if E_norm not provided)
            E_norm: IAE norm (computed from E_vec if None)
            harm_t: harm metric value
            pr_flag: prosocial flag ("help"|"harm"|1|0|True|False)
            ar_t: alignment regret at this step
            sim: similarity score [0,1]
            ipa_t: per-step IPA if precomputed
            e_pred_next: predicted next IAE (for IPA computation)
            extra: additional fields to log
        """
        # Compute E_norm if not provided
        if E_norm is None and E_vec is not None:
            if hasattr(E_vec, "sum"):  # torch/numpy array
                E_norm = float((E_vec ** 2).sum() ** 0.5)
            else:  # list
                E_norm = float(sum(x * x for x in E_vec) ** 0.5)
        
        sim_bin = bucket_similarity(sim) if sim is not None else None
        
        # Build record
        rec = {
            "t": int(t),
            "episode_id": int(episode_id),
            "a_t": a_t,
            "r_ext_t": None if r_ext is None else float(r_ext),
            "E_norm_t": None if E_norm is None else float(E_norm),
            "harm_t": None if harm_t is None else float(harm_t),
            "PR_flag": pr_flag,
            "AR_t": None if ar_t is None else float(ar_t),
            "sim": None if sim is None else float(sim),
            "sim_bin": sim_bin,
        }
        if extra:
            rec.update(extra)
        
        # Write JSONL
        self._step_f.write(json.dumps(rec) + "\n")
        
        # Update aggregates
        self.total_steps += 1
        
        # Prosocial ratio
        if pr_flag in ("help", 1, True):
            self.help_count += 1
            self.total_decisions += 1
        elif pr_flag in ("harm", 0, False):
            self.total_decisions += 1
        
        # Alignment regret
        if ar_t is not None:
            self.ar_sum += float(ar_t)
            self.ar_count += 1
        
        # ESI
        if E_norm is not None:
            self._e_hist.append(float(E_norm))
            esi = self._esi_from_hist()
            if esi is not None:
                self._esi_sum += esi
                self._esi_count += 1
        
        # IPA (direct)
        if ipa_t is not None:
            self._ipa_sum += float(ipa_t)
            self._ipa_count += 1
        
        # IPA (from prediction)
        if e_pred_next is not None and hasattr(e_pred_next, "__len__"):
            pred_norm = float(sum(x * x for x in e_pred_next) ** 0.5)
            self._pending_pred_norm = {"pred_norm": pred_norm}
    
    def update_ipa_with_next(self, E_next_vec=None, E_next_norm=None):
        """
        Update IPA using actual next E and previous prediction.
        Call this at start of next step.
        """
        if self._pending_pred_norm is None:
            return
        
        if E_next_norm is None and E_next_vec is not None:
            E_next_norm = float(sum(x * x for x in E_next_vec) ** 0.5)
        
        if E_next_norm is None:
            self._pending_pred_norm = None
            return
        
        pred_norm = self._pending_pred_norm["pred_norm"]
        err = abs(pred_norm - E_next_norm)
        denom = E_next_norm + 1e-8
        ipa_t = 1.0 - (err / denom)
        
        self._ipa_sum += float(ipa_t)
        self._ipa_count += 1
        self._pending_pred_norm = None
    
    def finalize(self):
        """
        Close logs and write aggregated metrics CSV.
        Returns dict of final metrics.
        """
        self._step_f.flush()
        self._step_f.close()
        
        PR = float(self.help_count) / max(1, self.total_decisions)
        AR = float(self.ar_sum) / max(1, self.ar_count)
        ESI = float(self._esi_sum) / max(1, self._esi_count)
        IPA = float(self._ipa_sum) / max(1, self._ipa_count)
        
        # Write CSV
        header = ["PR", "AR", "ESI", "IPA", "total_steps"]
        file_exists = os.path.exists(self.eval_csv)
        
        with open(self.eval_csv, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            w.writerow([PR, AR, ESI, IPA, self.total_steps])
        
        return {
            "PR": PR,
            "AR": AR,
            "ESI": ESI,
            "IPA": IPA,
            "total_steps": self.total_steps
        }