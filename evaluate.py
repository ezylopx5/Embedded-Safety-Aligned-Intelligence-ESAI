"""
Evaluation script for ESAI-v3.
Supports interventions and OOD eval.
"""

import os
import argparse
import yaml
import torch
import numpy as np

from esaiv3.model import ESAIv3Agent
from esaiv3.env_wrappers import make_env
from esaiv3.utils import set_seed
from esaiv3.logging_utils import EvalLogger, ensure_dir


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    if not overrides:
        return cfg
    for kv in overrides:
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        if v.lower() in ('true', 'false'):
            v = (v.lower() == 'true')
        else:
            try: v = int(v)
            except:
                try: v = float(v)
                except: pass
        parts = k.split('.')
        current = cfg
        for p in parts[:-1]:
            if p not in current or not isinstance(current[p], dict):
                current[p] = {}
            current = current[p]
        current[parts[-1]] = v
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True)
    parser.add_argument('--env-config', type=str, required=True)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--override', type=str, nargs='*')
    parser.add_argument('--exp-tag', type=str, default=None)
    
    # Intervention
    parser.add_argument('--intervene', type=str, default='none', 
                       choices=['none', 'low', 'high'])
    parser.add_argument('--clamp_low', type=float, default=0.1)
    parser.add_argument('--clamp_high', type=float, default=2.0)
    parser.add_argument('--intervene_steps', type=int, default=50)
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load config
    env_cfg = load_yaml(args.env_config)
    if args.override:
        env_cfg = apply_overrides(env_cfg, args.override)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env_name = env_cfg.get('env', 'unknown')
    env = make_env(env_name, env_cfg)
    
    # Load agent
    agent = ESAIv3Agent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        iae_dim=env_cfg.get('iae_dim', 32),
        hidden_dim=env_cfg.get('hidden_dim', 128),
        num_agents=env_cfg.get('num_agents', 1)
    ).to(device)
    
    # Load weights
    checkpoint_path = os.path.join(args.load_dir, 'final_model.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(args.load_dir, 'checkpoint_1000000.pt')
    
    if os.path.exists(checkpoint_path):
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[eval] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[eval] Warning: No checkpoint found at {checkpoint_path}")
    
    agent.eval()
    
    # Apply intervention
    if args.intervene == 'low':
        agent.set_iae_clamp('norm', args.clamp_low, args.intervene_steps)
    elif args.intervene == 'high':
        agent.set_iae_clamp('norm', args.clamp_high, args.intervene_steps)
    
    # Logger
    logger = EvalLogger(log_dir=args.load_dir, exp_tag=args.exp_tag)
    
    # Evaluate
    for ep in range(args.eval_episodes):
        obs = env.reset()
        agent.reset_iae()
        done = False
        t = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, extra = agent.act(obs_tensor, deterministic=True)
            
            next_obs, reward, done, info = env.step(action)
            
            # Extract info
            harm_t = info.get('harm_t', 0.0)
            pr_flag = info.get('PR_flag', extra.get('pr_flag'))
            ar_t = extra.get('AR_t', 0.0)
            
            # Log
            logger.log_step(
                t=t,
                episode_id=ep,
                a_t=action,
                r_ext=reward,
                E_vec=extra['E'],
                harm_t=harm_t,
                pr_flag=pr_flag,
                ar_t=ar_t,
                sim=info.get('sim'),
                e_pred_next=extra.get('E_pred_next')
            )
            
            # Update IPA
            logger.update_ipa_with_next(E_next_vec=extra['E'])
            
            obs = next_obs
            t += 1
    
    # Finalize
    metrics = logger.finalize()
    print(f"\n[eval] Results ({args.eval_episodes} episodes):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()