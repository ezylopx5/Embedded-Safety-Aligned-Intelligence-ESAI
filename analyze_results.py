#!/usr/bin/env python3
"""
Analyze ESAI-v3 experiment results
"""
import torch
import numpy as np
from pathlib import Path

def analyze_results():
    print("=" * 70)
    print("ESAI-v3 RESULTS ANALYSIS - Moral Temptation Environment")
    print("=" * 70)
    
    base_dir = Path("results/moral_temptation/esaiv3_paper")
    
    all_final_rewards = []
    all_prosocial = []
    all_help_counts = []
    all_steal_counts = []
    
    for seed_dir in sorted(base_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        
        print(f"\n📊 {seed_dir.name}")
        print("-" * 50)
        
        final_path = seed_dir / "checkpoint_final.pt"
        if final_path.exists():
            ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
            
            print(f"  Episode: {ckpt.get('episode', 'N/A')}")
            print(f"  Global Step: {ckpt.get('global_step', 'N/A')}")
            
            # Episode rewards
            if "episode_rewards" in ckpt:
                rewards = ckpt["episode_rewards"]
                if len(rewards) > 0:
                    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
                    print(f"  Final 100 ep reward: {np.mean(last_100):.3f} ± {np.std(last_100):.3f}")
                    all_final_rewards.append(np.mean(last_100))
            
            # Metrics history
            if "metrics_history" in ckpt:
                mh = ckpt["metrics_history"]
                if isinstance(mh, dict):
                    # Extract EVAL metrics (more reliable)
                    if "eval_prosocial_ratio" in mh and len(mh["eval_prosocial_ratio"]) > 0:
                        pr = mh["eval_prosocial_ratio"][-1]
                        print(f"  Final Eval Prosocial Ratio: {pr:.3f}")
                        all_prosocial.append(pr)
                    
                    if "eval_total_help" in mh and len(mh["eval_total_help"]) > 0:
                        hc = mh["eval_total_help"][-1]
                        print(f"  Final Eval Help Count: {hc:.1f}")
                        all_help_counts.append(hc)
                    
                    if "eval_total_steal" in mh and len(mh["eval_total_steal"]) > 0:
                        sc = mh["eval_total_steal"][-1]
                        print(f"  Final Eval Steal Count: {sc:.1f}")
                        all_steal_counts.append(sc)
                    
                    if "eval_iae_norm" in mh and len(mh["eval_iae_norm"]) > 0:
                        iae = mh["eval_iae_norm"][-1]
                        print(f"  Final Eval IAE Norm: {iae:.3f}")
                    
                    if "eval_alignment_regret" in mh and len(mh["eval_alignment_regret"]) > 0:
                        ar = mh["eval_alignment_regret"][-1]
                        print(f"  Final Eval Alignment Regret: {ar:.3f}")
                    
                    if "eval_mean_reward" in mh and len(mh["eval_mean_reward"]) > 0:
                        emr = mh["eval_mean_reward"][-1]
                        print(f"  Final Eval Mean Reward: {emr:.3f}")
                elif isinstance(mh, list) and len(mh) > 0:
                    print(f"  Metrics history (list): {len(mh)} entries")
    
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS 5 SEEDS")
    print("=" * 70)
    
    if all_final_rewards:
        print(f"\n📈 Final Episode Reward (last 100 episodes):")
        print(f"   Per seed: {[f'{r:.2f}' for r in all_final_rewards]}")
        print(f"   Mean ± Std: {np.mean(all_final_rewards):.3f} ± {np.std(all_final_rewards):.3f}")
    
    if all_prosocial:
        print(f"\n🤝 Prosocial Ratio:")
        print(f"   Per seed: {[f'{r:.3f}' for r in all_prosocial]}")
        print(f"   Mean ± Std: {np.mean(all_prosocial):.3f} ± {np.std(all_prosocial):.3f}")
    
    if all_help_counts and all_steal_counts:
        print(f"\n📊 Help vs Steal:")
        print(f"   Help counts: {[f'{c:.1f}' for c in all_help_counts]}")
        print(f"   Steal counts: {[f'{c:.1f}' for c in all_steal_counts]}")
        total_help = sum(all_help_counts)
        total_steal = sum(all_steal_counts)
        if total_help + total_steal > 0:
            print(f"   Overall Help/Steal ratio: {total_help/(total_help+total_steal):.3f}")

if __name__ == "__main__":
    analyze_results()
