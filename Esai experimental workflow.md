# ESAI Research Experiment: Complete Workflow

## Project Overview
**Goal**: Implement and validate ESAI (Embedded Safety-Aligned Intelligence) framework for multi-agent reinforcement learning with internal alignment embeddings.

**Based on**: Your research papers:
1. "Embedded Safety-Aligned Intelligence for Multi-Agent Reinforcement Learning" (ALA @ AAMAS 2026 Workshop - 6 pages)
2. "Learning Internal Alignment Embeddings for Scalable Multi-Agent Coordination" (ESAI-v3 - Full experimental paper)
3. "Embedded Safety-Aligned Intelligence via Differentiable Internal Alignment Embeddings" (Theoretical Framework)

**Key Focus**: The workshop paper emphasizes theoretical contributions and positions ESAI as a conceptual framework. Your empirical validation should support these theoretical claims.

---

## Phase 1: Environment Setup (Week 1)

### 1.1 Development Environment
```bash
# System Requirements
- macOS Silicon (M1/M2/M3)
- Python 3.10.12+
- CUDA support (if using GPU)
- 16GB+ RAM recommended

# Create project structure
mkdir -p ~/esai-research/{environments,models,experiments,data,results,logs}
cd ~/esai-research

# Python environment
python3 -m venv esai_env
source esai_env/bin/activate

# Core dependencies
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib seaborn
pip install gym pettingzoo gymnasium
pip install tensorboard wandb  # for experiment tracking
pip install scipy scikit-learn
```

### 1.2 Install MARL Environments
```bash
# Multi-Agent Particle Environment (MPE)
pip install pettingzoo[mpe]

# Sequential Social Dilemmas
git clone https://github.com/eugenevinitsky/sequential_social_dilemma_games.git
cd sequential_social_dilemma_games
pip install -e .

# Additional environments
pip install overcooked-ai  # For Overcooked coordination
pip install marlgrid  # For gridworld tasks
```

### 1.3 Project Structure
```
esai-research/
├── environments/
│   ├── moral_temptation.py      # Custom moral temptation grid
│   ├── social_distress.py       # Social distress diffusion
│   └── wrappers.py              # Environment wrappers
├── models/
│   ├── esai_agent.py            # Main ESAI-v3 agent
│   ├── iae_dynamics.py          # IAE update mechanisms
│   ├── counterfactual.py        # Counterfactual forecasting
│   ├── attention.py             # IAE-weighted attention
│   ├── hebbian.py               # Hebbian memory
│   └── graph_diffusion.py       # Graph diffusion operators
├── baselines/
│   ├── ppo.py                   # Vanilla PPO
│   ├── cpo.py                   # Constrained Policy Optimization
│   ├── reward_shaping.py        # Reward shaping baseline
│   └── multi_objective.py       # Multi-objective RL
├── experiments/
│   ├── run_experiment.py        # Main experiment runner
│   ├── ablations.py             # Ablation studies
│   ├── interventions.py         # Causal intervention probes
│   └── scaling_tests.py         # Zero-shot scaling experiments
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   ├── visualization.py         # Plotting and visualization
│   └── logging.py               # Experiment logging
└── configs/
    ├── default_config.yaml      # Default hyperparameters
    └── env_configs/             # Environment-specific configs
```

---

## Phase 2: Implementation (Weeks 2-4)

### 2.1 Core ESAI-v3 Components

#### Step 1: IAE Dynamics Module
```python
# File: models/iae_dynamics.py

import torch
import torch.nn as nn

class IAEDynamics(nn.Module):
    """
    Implements Equation 4 from paper:
    E_{i,t+1} = γ_E * E_{i,t} + g_φ(z_{i,t}, a_{i,t}, r_t^ext) - α * L @ E_{j,t}
    """
    def __init__(self, iae_dim=32, obs_dim=64, action_dim=6, hidden_dim=128):
        super().__init__()
        self.iae_dim = iae_dim
        self.gamma_E = 0.9  # Persistence factor
        self.alpha = 0.05   # Diffusion strength
        
        # Learned update function g_φ (MLP)
        self.update_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1, hidden_dim),  # +1 for reward
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, iae_dim)
        )
        
        # Apply gradient clipping for Lipschitz constraint (Lg = 1.0)
        for param in self.update_net.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    
    def forward(self, E_prev, obs, action, reward, laplacian, neighbor_E):
        """
        Args:
            E_prev: Previous IAE (batch_size, iae_dim)
            obs: Observation (batch_size, obs_dim)
            action: Action taken (batch_size, action_dim)
            reward: Extrinsic reward (batch_size, 1)
            laplacian: Graph Laplacian (N, N)
            neighbor_E: Neighbor IAEs (batch_size, num_neighbors, iae_dim)
        """
        # Learned dynamics
        g_phi_input = torch.cat([obs, action, reward], dim=-1)
        g_phi = self.update_net(g_phi_input)
        
        # Graph diffusion term
        diffusion = self.alpha * torch.matmul(laplacian, neighbor_E)
        
        # Update equation
        E_next = self.gamma_E * E_prev + g_phi - diffusion
        
        return E_next
```

#### Step 2: Counterfactual Forecasting
```python
# File: models/counterfactual.py

class CounterfactualForecaster(nn.Module):
    """
    Implements Equations 5-10: Differentiable counterfactual alignment penalty
    """
    def __init__(self, iae_dim=32, obs_dim=64, action_dim=6, hidden_dim=64):
        super().__init__()
        self.iae_dim = iae_dim
        
        # Forecast network h_ψ
        self.forecast_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1 + iae_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, iae_dim)
        )
        
        # EMA target network for stability
        self.target_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + 1 + iae_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, iae_dim)
        )
        self.target_net.load_state_dict(self.forecast_net.state_dict())
        
        self.tau_ema = 0.995  # EMA update rate
    
    def forecast(self, obs, action, reward, hebbian_read, use_target=True):
        """Forecast next IAE for given action"""
        net = self.target_net if use_target else self.forecast_net
        inputs = torch.cat([obs, action, reward, hebbian_read], dim=-1)
        return net(inputs)
    
    def compute_alignment_regret(self, E_next, obs, actions, reward, hebbian_read, 
                                 neighbor_E, tau=0.5, kappa=0.5):
        """
        Compute alignment regret (Equation 10)
        """
        batch_size = obs.shape[0]
        num_actions = actions.shape[1]
        
        # Forecast for all candidate actions
        E_forecasts = []
        for i in range(num_actions):
            action = actions[:, i]
            E_pred = self.forecast(obs, action, reward, hebbian_read, use_target=True)
            E_forecasts.append(E_pred)
        E_forecasts = torch.stack(E_forecasts, dim=1)  # (batch, num_actions, iae_dim)
        
        # Compute harm R(a) = ||E^(a)||_2
        R = torch.norm(E_forecasts, p=2, dim=-1)  # (batch, num_actions)
        
        # Softmin reference distribution (Equation 8)
        pi_ref = torch.softmax(-R / tau, dim=-1)  # (batch, num_actions)
        
        # Expected reference embedding (Equation 9)
        E_ref = torch.sum(pi_ref.unsqueeze(-1) * E_forecasts, dim=1)  # (batch, iae_dim)
        
        # Alignment regret (Equation 10)
        deviation_term = torch.norm(E_next - E_ref, p=2, dim=-1) ** 2
        neighbor_term = kappa * torch.mean(torch.norm(neighbor_E, p=2, dim=-1) ** 2)
        
        AR = deviation_term + neighbor_term
        
        return AR, E_ref
    
    def update_target(self):
        """Update EMA target network"""
        for target_param, param in zip(self.target_net.parameters(), 
                                       self.forecast_net.parameters()):
            target_param.data.copy_(
                self.tau_ema * target_param.data + (1 - self.tau_ema) * param.data
            )
```

#### Step 3: IAE-Weighted Attention
```python
# File: models/attention.py

class IAEAttention(nn.Module):
    """
    Implements Equation 14: IAE-weighted attention mechanism
    """
    def __init__(self, iae_dim=32, obs_dim=64):
        super().__init__()
        # Projection matrix W_a
        self.W_a = nn.Linear(iae_dim, obs_dim, bias=True)
    
    def forward(self, E, obs):
        """
        Args:
            E: Internal alignment embedding (batch_size, iae_dim)
            obs: Observation (batch_size, obs_dim)
        Returns:
            z_tilde: Attention-weighted observation
            alpha: Attention weights (for interpretability)
        """
        # Compute attention weights (Equation 14)
        alpha = torch.softmax(self.W_a(E), dim=-1)
        
        # Modulate observation
        z_tilde = alpha * obs
        
        return z_tilde, alpha
```

#### Step 4: Hebbian Memory
```python
# File: models/hebbian.py

class HebbianMemory(nn.Module):
    """
    Implements Equations 15-16: Hebbian affect-memory coupling
    """
    def __init__(self, iae_dim=32, obs_dim=64):
        super().__init__()
        self.iae_dim = iae_dim
        self.obs_dim = obs_dim
        
        self.eta_H = 1e-3  # Learning rate
        self.delta_H = 0.02  # Decay rate
        
        # Read projection
        self.W_r = nn.Linear(iae_dim * obs_dim, iae_dim)
    
    def update(self, H_prev, E, obs):
        """
        Update Hebbian matrix (Equation 15)
        H_{t+1} = (1 - δ_H) * H_t + η_H * (E_t ⊗ z_t)
        """
        # Outer product E ⊗ z
        outer_product = torch.bmm(E.unsqueeze(-1), obs.unsqueeze(1))
        
        # Update with decay
        H_next = (1 - self.delta_H) * H_prev + self.eta_H * outer_product
        
        return H_next
    
    def read(self, H):
        """
        Differentiable read operation (Equation 16)
        """
        batch_size = H.shape[0]
        H_flat = H.view(batch_size, -1)
        return self.W_r(H_flat)
```

#### Step 5: Graph Diffusion
```python
# File: models/graph_diffusion.py

class GraphDiffusion(nn.Module):
    """
    Implements graph diffusion with similarity weighting (Equations 17-18)
    """
    def __init__(self, num_agents, identity_dim=8, lambda_bias=0.01):
        super().__init__()
        self.num_agents = num_agents
        
        # Learned identity embeddings φ_i
        self.identity_embeddings = nn.Parameter(
            torch.randn(num_agents, identity_dim)
        )
        
        self.lambda_bias = lambda_bias  # Bias regularization strength
    
    def compute_similarity_matrix(self):
        """Compute similarity matrix S (Equation 17)"""
        # Cosine similarity between identity embeddings
        normalized = self.identity_embeddings / (
            torch.norm(self.identity_embeddings, dim=-1, keepdim=True) + 1e-8
        )
        S = torch.matmul(normalized, normalized.t())
        S = torch.clamp(S, min=0.0)  # β_ij = max(0, cos(φ_i, φ_j))
        return S
    
    def compute_laplacian(self, adjacency):
        """
        Compute normalized graph Laplacian
        L = I - D^{-1/2} A D^{-1/2}
        """
        # Degree matrix
        degree = torch.sum(adjacency, dim=-1)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(degree) + 1e-8))
        
        # Normalized Laplacian
        L = torch.eye(self.num_agents) - torch.matmul(
            torch.matmul(D_inv_sqrt, adjacency), D_inv_sqrt
        )
        
        return L
    
    def bias_regularization(self, adjacency):
        """
        Similarity-suppression regularizer (Equation 18)
        L_bias = λ_bias * ||A ⊙ S||_F^2
        """
        S = self.compute_similarity_matrix()
        return self.lambda_bias * torch.norm(adjacency * S, p='fro') ** 2
```

### 2.2 Complete ESAI Agent
```python
# File: models/esai_agent.py

class ESAIAgent(nn.Module):
    """
    Complete ESAI-v3 agent integrating all components
    """
    def __init__(self, obs_dim, action_dim, num_agents, iae_dim=32, hidden_dim=128):
        super().__init__()
        
        # Components
        self.iae_dynamics = IAEDynamics(iae_dim, obs_dim, action_dim, hidden_dim)
        self.counterfactual = CounterfactualForecaster(iae_dim, obs_dim, action_dim)
        self.attention = IAEAttention(iae_dim, obs_dim)
        self.hebbian = HebbianMemory(iae_dim, obs_dim)
        self.graph_diffusion = GraphDiffusion(num_agents)
        
        # Policy network (processes attention-weighted observations)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # State
        self.E = torch.zeros(1, iae_dim)  # Current IAE
        self.H = torch.zeros(1, iae_dim, obs_dim)  # Hebbian matrix
    
    def forward(self, obs, action=None, reward=None, neighbor_E=None):
        """
        Forward pass through ESAI architecture
        """
        # IAE-weighted attention
        z_tilde, alpha = self.attention(self.E, obs)
        
        # Policy and value
        action_logits = self.policy(z_tilde)
        value = self.value(z_tilde)
        
        # If action provided, update IAE
        if action is not None and reward is not None:
            # Read Hebbian memory
            hebbian_read = self.hebbian.read(self.H)
            
            # Compute graph Laplacian
            adjacency = self.graph_diffusion.compute_similarity_matrix()
            laplacian = self.graph_diffusion.compute_laplacian(adjacency)
            
            # Update IAE
            self.E = self.iae_dynamics(
                self.E, obs, action, reward, laplacian, neighbor_E
            )
            
            # Update Hebbian memory
            self.H = self.hebbian.update(self.H, self.E, obs)
        
        return action_logits, value, alpha
```

---

## Phase 3: Environment Implementation (Week 3)

### 3.1 Moral Temptation Grid
```python
# File: environments/moral_temptation.py

import gymnasium as gym
import numpy as np

class MoralTemptationGrid(gym.Env):
    """
    8x8 discrete grid with help (R_low=1) vs steal (R_high) actions
    Stealing triggers victim distress
    """
    def __init__(self, grid_size=8, temptation_gap=5, num_agents=4):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.R_low = 1.0  # Help reward
        self.R_high = 1.0 + temptation_gap  # Steal reward
        
        # Action space: {up, down, left, right, help, steal}
        self.action_space = gym.spaces.Discrete(6)
        
        # Observation: 5x5 local patch + position encoding
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(5*5 + 8,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Place agents randomly
        self.agent_positions = []
        for _ in range(self.num_agents):
            pos = self.np_random.integers(0, self.grid_size, size=2)
            self.agent_positions.append(pos)
        
        # Place victims and resources
        self.victims = []
        self.resources = []
        for _ in range(3):  # 3-5 resources
            pos = self.np_random.integers(0, self.grid_size, size=2)
            self.resources.append(pos)
        
        self.victim_distress = np.zeros(len(self.victims))
        self.timestep = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get observation for each agent"""
        obs = []
        for i, pos in enumerate(self.agent_positions):
            # 5x5 local patch centered on agent
            x, y = pos
            patch = np.zeros((5, 5))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        patch[dx+2, dy+2] = self.grid[nx, ny]
            
            # Position encoding (one-hot)
            pos_encoding = np.zeros(8)
            pos_encoding[i] = 1
            
            # Concatenate
            obs_i = np.concatenate([patch.flatten(), pos_encoding])
            obs.append(obs_i)
        
        return np.array(obs)
    
    def step(self, actions):
        """Execute actions for all agents"""
        rewards = np.zeros(self.num_agents)
        harms = np.zeros(self.num_agents)
        
        for i, action in enumerate(actions):
            if action < 4:  # Movement
                self._move_agent(i, action)
            elif action == 4:  # Help
                rewards[i] = self.R_low
                harms[i] = 0
            elif action == 5:  # Steal
                rewards[i] = self.R_high
                # Trigger victim distress
                nearest_victim = self._find_nearest_victim(i)
                if nearest_victim is not None:
                    self.victim_distress[nearest_victim] += 3.0
                    harms[i] = 3.0
        
        self.timestep += 1
        done = self.timestep >= 200
        
        obs = self._get_obs()
        info = {
            'harms': harms,
            'victim_distress': self.victim_distress.copy()
        }
        
        return obs, rewards, done, False, info
    
    def _move_agent(self, agent_idx, direction):
        """Move agent in cardinal direction"""
        pos = self.agent_positions[agent_idx]
        if direction == 0:  # Up
            pos[1] = max(0, pos[1] - 1)
        elif direction == 1:  # Down
            pos[1] = min(self.grid_size - 1, pos[1] + 1)
        elif direction == 2:  # Left
            pos[0] = max(0, pos[0] - 1)
        elif direction == 3:  # Right
            pos[0] = min(self.grid_size - 1, pos[0] + 1)
    
    def _find_nearest_victim(self, agent_idx):
        """Find nearest victim to agent"""
        if len(self.victims) == 0:
            return None
        
        agent_pos = self.agent_positions[agent_idx]
        distances = [np.linalg.norm(agent_pos - v) for v in self.victims]
        return np.argmin(distances)
```

### 3.2 Social Distress Diffusion
```python
# File: environments/social_distress.py

class SocialDistressDiffusion(gym.Env):
    """
    16 agents on 8x8 toroidal lattice
    External shocks inject negative IAE
    Tests graph diffusion and bias mitigation
    """
    def __init__(self, grid_size=8, num_agents=16):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.shock_interval = 50  # Timesteps between shocks
        
        self.action_space = gym.spaces.Discrete(5)  # 4 movements + help
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(grid_size*grid_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Place agents on grid
        positions = np.random.choice(
            self.grid_size * self.grid_size, 
            size=self.num_agents, 
            replace=False
        )
        self.agent_positions = [
            np.array([p % self.grid_size, p // self.grid_size]) 
            for p in positions
        ]
        
        self.distress_field = np.zeros((self.grid_size, self.grid_size))
        self.timestep = 0
        
        return self._get_obs(), {}
    
    def step(self, actions):
        # Apply external shocks periodically
        if self.timestep % self.shock_interval == 0:
            target_idx = np.random.randint(self.num_agents)
            target_pos = self.agent_positions[target_idx]
            self.distress_field[target_pos[0], target_pos[1]] -= 3.0
        
        # Execute actions
        rewards = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            if action < 4:
                self._move_agent(i, action)
            elif action == 4:  # Help nearby agents
                neighbors = self._get_neighbors(i)
                for n_idx in neighbors:
                    n_pos = self.agent_positions[n_idx]
                    self.distress_field[n_pos[0], n_pos[1]] += 0.5
                rewards[i] = 0.5
        
        # Diffuse distress (simple diffusion)
        self.distress_field = 0.9 * self.distress_field
        
        self.timestep += 1
        done = self.timestep >= 500
        
        return self._get_obs(), rewards, done, False, {}
    
    def _get_neighbors(self, agent_idx, radius=2):
        """Get agents within radius (toroidal)"""
        pos = self.agent_positions[agent_idx]
        neighbors = []
        for i, other_pos in enumerate(self.agent_positions):
            if i == agent_idx:
                continue
            # Toroidal distance
            dx = min(abs(pos[0] - other_pos[0]), 
                    self.grid_size - abs(pos[0] - other_pos[0]))
            dy = min(abs(pos[1] - other_pos[1]), 
                    self.grid_size - abs(pos[1] - other_pos[1]))
            if dx + dy <= radius:
                neighbors.append(i)
        return neighbors
```

---

## Phase 4: Baseline Implementations (Week 4)

### 4.1 Vanilla PPO
```python
# File: baselines/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    """Vanilla PPO baseline"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=3e-4):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
    
    def select_action(self, obs):
        with torch.no_grad():
            logits = self.policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def update(self, obs, actions, old_log_probs, advantages, returns):
        """PPO update step"""
        # Current policy
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Importance ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.value(obs).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=0.5
        )
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item()
        }
```

### 4.2 Reward Shaping Baseline
```python
# File: baselines/reward_shaping.py

class RewardShapingAgent(PPOAgent):
    """PPO with hand-designed prosocial bonus"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prosocial_bonus = 0.5
    
    def shape_reward(self, reward, action, info):
        """Add prosocial bonus for help actions"""
        if action == 4:  # Help action
            reward += self.prosocial_bonus
        return reward
```

---

## Phase 5: Training Pipeline (Week 5)

### 5.1 Main Training Loop
```python
# File: experiments/run_experiment.py

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml

class ESAITrainer:
    """Main training loop for ESAI experiments"""
    
    def __init__(self, config_path='configs/default_config.yaml'):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize environment
        self.env = self._create_environment()
        
        # Initialize agent
        self.agent = ESAIAgent(
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            num_agents=self.config['num_agents'],
            iae_dim=self.config['iae_dim'],
            hidden_dim=self.config['hidden_dim']
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Logging
        self.writer = SummaryWriter(f"runs/{self.config['experiment_name']}")
        
        # Temperature annealing
        self.tau = self.config['tau_initial']
        self.tau_min = self.config['tau_min']
        self.tau_decay_steps = self.config['tau_decay_steps']
    
    def train(self, num_episodes=10000):
        """Main training loop"""
        for episode in range(num_episodes):
            # Collect episode
            trajectory = self.collect_episode()
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages_returns(trajectory)
            
            # Update agent
            losses = self.update_agent(trajectory, advantages, returns)
            
            # Logging
            if episode % 100 == 0:
                self.log_metrics(episode, trajectory, losses)
            
            # Anneal temperature
            self.anneal_temperature(episode)
            
            # Save checkpoint
            if episode % 1000 == 0:
                self.save_checkpoint(episode)
    
    def collect_episode(self):
        """Collect one episode of experience"""
        obs, _ = self.env.reset()
        done = False
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'harms': [],
            'alignment_regrets': [],
            'attention_weights': []
        }
        
        while not done:
            obs_tensor = torch.FloatTensor(obs)
            
            # Forward pass
            with torch.no_grad():
                action_logits, value, attention = self.agent(obs_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # Environment step
            next_obs, rewards, done, truncated, info = self.env.step(actions.numpy())
            
            # Compute alignment regret
            with torch.no_grad():
                AR, _ = self.agent.counterfactual.compute_alignment_regret(
                    self.agent.E,
                    obs_tensor,
                    action_logits,
                    torch.FloatTensor(rewards).unsqueeze(-1),
                    self.agent.hebbian.read(self.agent.H),
                    self.agent.E,  # Placeholder for neighbor embeddings
                    tau=self.tau
                )
            
            # Shape rewards
            shaped_rewards = rewards - self.config['lambda_reg'] * AR.numpy()
            
            # Store
            trajectory['observations'].append(obs)
            trajectory['actions'].append(actions.numpy())
            trajectory['rewards'].append(shaped_rewards)
            trajectory['log_probs'].append(log_probs.numpy())
            trajectory['values'].append(value.numpy())
            trajectory['harms'].append(info.get('harms', np.zeros_like(rewards)))
            trajectory['alignment_regrets'].append(AR.numpy())
            trajectory['attention_weights'].append(attention.numpy())
            
            obs = next_obs
            done = done or truncated
        
        return trajectory
    
    def compute_advantages_returns(self, trajectory):
        """Compute GAE advantages and returns"""
        rewards = np.array(trajectory['rewards'])
        values = np.array(trajectory['values'])
        
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config['gamma'] * next_value - values[t]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return np.array(advantages), np.array(returns)
    
    def update_agent(self, trajectory, advantages, returns):
        """PPO update with ESAI components"""
        # Convert to tensors
        obs = torch.FloatTensor(np.array(trajectory['observations']))
        actions = torch.LongTensor(np.array(trajectory['actions']))
        old_log_probs = torch.FloatTensor(np.array(trajectory['log_probs']))
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Multiple PPO epochs
        for _ in range(self.config['ppo_epochs']):
            # Forward pass
            action_logits, values, _ = self.agent(obs)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1 - self.config['clip_epsilon'], 
                1 + self.config['clip_epsilon']
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Forecast loss
            forecast_loss = self._compute_forecast_loss(trajectory)
            
            # Bias regularization
            bias_loss = self.agent.graph_diffusion.bias_regularization(
                self.agent.graph_diffusion.compute_similarity_matrix()
            )
            
            # Total loss
            total_loss = (
                policy_loss + 
                0.5 * value_loss - 
                0.01 * entropy.mean() +
                self.config['lambda_forecast'] * forecast_loss +
                bias_loss
            )
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
            
            # Update EMA target
            self.agent.counterfactual.update_target()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'forecast_loss': forecast_loss.item(),
            'bias_loss': bias_loss.item()
        }
    
    def _compute_forecast_loss(self, trajectory):
        """Compute forecast network supervision loss"""
        # Implementation of Equation 6
        pass
    
    def anneal_temperature(self, episode):
        """Exponentially anneal temperature"""
        self.tau = max(
            self.tau_min,
            self.config['tau_initial'] * np.exp(-episode / self.tau_decay_steps)
        )
    
    def log_metrics(self, episode, trajectory, losses):
        """Log metrics to tensorboard"""
        # Task metrics
        total_reward = np.sum(trajectory['rewards'])
        total_harm = np.sum(trajectory['harms'])
        prosocial_ratio = np.mean(np.array(trajectory['actions']) == 4)
        
        # ESAI metrics
        mean_alignment_regret = np.mean(trajectory['alignment_regrets'])
        mean_iae_norm = torch.norm(self.agent.E).item()
        
        # Log
        self.writer.add_scalar('Metrics/Total_Reward', total_reward, episode)
        self.writer.add_scalar('Metrics/Total_Harm', total_harm, episode)
        self.writer.add_scalar('Metrics/Prosocial_Ratio', prosocial_ratio, episode)
        self.writer.add_scalar('Metrics/Alignment_Regret', mean_alignment_regret, episode)
        self.writer.add_scalar('Metrics/IAE_Norm', mean_iae_norm, episode)
        self.writer.add_scalar('Hyperparams/Temperature', self.tau, episode)
        
        for key, value in losses.items():
            self.writer.add_scalar(f'Losses/{key}', value, episode)
        
        print(f"Episode {episode}: Reward={total_reward:.2f}, "
              f"Harm={total_harm:.2f}, PR={prosocial_ratio:.3f}, "
              f"AR={mean_alignment_regret:.3f}")
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        torch.save({
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tau': self.tau
        }, f"checkpoints/esai_ep{episode}.pt")

if __name__ == "__main__":
    trainer = ESAITrainer('configs/moral_temptation.yaml')
    trainer.train(num_episodes=10000)
```

### 5.2 Configuration File
```yaml
# File: configs/moral_temptation.yaml

experiment_name: "moral_temptation_delta5"
environment: "moral_temptation"

# Environment settings
num_agents: 4
grid_size: 8
temptation_gap: 5

# ESAI architecture
iae_dim: 32
hidden_dim: 128
identity_dim: 8

# Training hyperparameters
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
ppo_epochs: 4
batch_size: 2048

# ESAI-specific
lambda_reg: 0.1        # Alignment penalty weight
lambda_forecast: 1.0   # Forecast loss weight
lambda_bias: 0.01      # Bias regularization

tau_initial: 1.0       # Initial temperature
tau_min: 0.01          # Minimum temperature
tau_decay_steps: 500000  # Annealing steps

# Graph diffusion
alpha: 0.05           # Diffusion strength
gamma_E: 0.9          # IAE persistence

# Hebbian
eta_H: 0.001          # Hebbian learning rate
delta_H: 0.02         # Hebbian decay

# Training
total_episodes: 1000000
eval_frequency: 5000
checkpoint_frequency: 10000
num_seeds: 5
```

---

## Phase 6: Evaluation & Metrics (Week 6)

### 6.1 Evaluation Metrics
```python
# File: utils/metrics.py

import numpy as np
from scipy.stats import pearsonr

class ESAIMetrics:
    """Compute all evaluation metrics from paper"""
    
    @staticmethod
    def prosocial_ratio(actions, help_action_id=4):
        """Fraction of cooperative actions (Equation from paper)"""
        return np.mean(actions == help_action_id)
    
    @staticmethod
    def alignment_regret(alignment_regrets):
        """Episode-averaged alignment regret"""
        return np.mean(alignment_regrets)
    
    @staticmethod
    def embedding_stability_index(iae_norms):
        """
        ESI = 1 / (1 + CV)
        where CV = std / mean
        """
        mean_norm = np.mean(iae_norms)
        std_norm = np.std(iae_norms)
        cv = std_norm / (mean_norm + 1e-8)
        return 1.0 / (1.0 + cv)
    
    @staticmethod
    def iae_harm_correlation(iae_norms, ground_truth_harms):
        """
        Pearson correlation between IAE magnitude and ground-truth harm
        Table 3 from paper
        """
        r, p_value = pearsonr(iae_norms, ground_truth_harms)
        return r, p_value
    
    @staticmethod
    def help_gap(actions, similarity_matrix, threshold_high=0.8, threshold_low=0.3):
        """
        In-group favoritism metric:
        P(help | similarity > 0.8) - P(help | similarity < 0.3)
        """
        high_sim_mask = similarity_matrix > threshold_high
        low_sim_mask = similarity_matrix < threshold_low
        
        # Help probability for high vs low similarity pairs
        p_help_high = np.mean(actions[high_sim_mask] == 4) if np.any(high_sim_mask) else 0
        p_help_low = np.mean(actions[low_sim_mask] == 4) if np.any(low_sim_mask) else 0
        
        return p_help_high - p_help_low
    
    @staticmethod
    def coordination_efficiency(deliveries, broken_plates, idle_time, episode_length):
        """
        Overcooked metric (Equation 20):
        CoordEff = soups_delivered/T - 0.5*plates_broken/T - 0.1*idle%
        """
        return (deliveries / episode_length - 
                0.5 * broken_plates / episode_length - 
                0.1 * idle_time / episode_length)
```

### 6.2 Visualization
```python
# File: utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

class ESAIVisualizer:
    """Visualization tools for ESAI analysis"""
    
    @staticmethod
    def plot_iae_trajectories(iae_history, harm_events, save_path='iae_trajectories.png'):
        """
        Figure 3 from paper: PCA projection of IAE trajectories
        """
        # PCA to 2D
        pca = PCA(n_components=2)
        iae_2d = pca.fit_transform(iae_history)
        
        # Plot trajectory
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            iae_2d[:, 0], 
            iae_2d[:, 1], 
            c=np.arange(len(iae_2d)),  # Color by timestep
            cmap='viridis',
            alpha=0.6
        )
        
        # Mark harm events
        for event_idx in harm_events:
            plt.scatter(
                iae_2d[event_idx, 0],
                iae_2d[event_idx, 1],
                color='red',
                marker='x',
                s=100,
                linewidths=3
            )
        
        plt.colorbar(scatter, label='Timestep')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('IAE Trajectory (PCA Projection)')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_attention_heatmap(attention_weights, feature_names, save_path='attention.png'):
        """
        Figure 2 from paper: Attention weight heatmap
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_weights.T,
            cmap='YlOrRd',
            xticklabels=50,  # Show every 50th timestep
            yticklabels=feature_names,
            cbar_kws={'label': 'Attention Weight'}
        )
        plt.xlabel('Timestep')
        plt.ylabel('Feature')
        plt.title('IAE-Weighted Attention Over Time')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_learning_curves(metrics_history, baseline_metrics, save_path='learning_curves.png'):
        """
        Compare ESAI vs baselines across training
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prosocial Ratio
        axes[0, 0].plot(metrics_history['prosocial_ratio'], label='ESAI-v3', linewidth=2)
        for name, values in baseline_metrics.items():
            axes[0, 0].plot(values['prosocial_ratio'], label=name, alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Prosocial Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Alignment Regret
        axes[0, 1].plot(metrics_history['alignment_regret'], label='ESAI-v3', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Alignment Regret')
        axes[0, 1].grid(alpha=0.3)
        
        # Total Reward
        axes[1, 0].plot(metrics_history['total_reward'], label='ESAI-v3', linewidth=2)
        for name, values in baseline_metrics.items():
            axes[1, 0].plot(values['total_reward'], label=name, alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Embedding Stability
        axes[1, 1].plot(metrics_history['esi'], label='ESAI-v3', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Embedding Stability Index')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
```

---

## Phase 7: Experiments to Run (Weeks 7-10)

### 7.1 Main Experiments (Table 2 from paper)

```bash
# Experiment 1: Moral Temptation with varying gaps
python experiments/run_experiment.py \
    --config configs/moral_temptation.yaml \
    --temptation_gap 1 \
    --seeds 5 \
    --name "moral_gap1"

python experiments/run_experiment.py \
    --config configs/moral_temptation.yaml \
    --temptation_gap 5 \
    --seeds 5 \
    --name "moral_gap5"

python experiments/run_experiment.py \
    --config configs/moral_temptation.yaml \
    --temptation_gap 10 \
    --seeds 5 \
    --name "moral_gap10"

# Experiment 2: Social Distress Diffusion
python experiments/run_experiment.py \
    --config configs/social_distress.yaml \
    --num_agents 16 \
    --seeds 5 \
    --name "social_distress"

# Experiment 3: MPE Cooperative Navigation
python experiments/run_experiment.py \
    --config configs/mpe.yaml \
    --num_agents 4 \
    --seeds 5 \
    --name "mpe_navigation"

# Experiment 4: Overcooked Coordination
python experiments/run_experiment.py \
    --config configs/overcooked.yaml \
    --num_agents 2 \
    --seeds 5 \
    --name "overcooked"

# Experiment 5: SSD Harvest
python experiments/run_experiment.py \
    --config configs/ssd_harvest.yaml \
    --num_agents 5 \
    --seeds 5 \
    --name "ssd_harvest"
```

### 7.2 Baseline Comparisons

```bash
# Run all baselines for each environment
for env in moral_temptation social_distress mpe overcooked ssd; do
    # ESAI-v3 (Full)
    python experiments/run_baseline.py --agent esai --env $env --seeds 5
    
    # Raw PPO
    python experiments/run_baseline.py --agent ppo --env $env --seeds 5
    
    # Reward Shaping
    python experiments/run_baseline.py --agent reward_shaping --env $env --seeds 5
    
    # CPO
    python experiments/run_baseline.py --agent cpo --env $env --seeds 5
    
    # Multi-Objective RL
    python experiments/run_baseline.py --agent mo_rl --env $env --seeds 5
done
```

### 7.3 Ablation Studies (Table 5)

```python
# File: experiments/ablations.py

def run_ablations():
    """Run all ablation experiments"""
    
    ablations = [
        {'name': 'Full', 'disable': []},
        {'name': 'No-Regret', 'disable': ['counterfactual']},
        {'name': 'No-Attention', 'disable': ['attention']},
        {'name': 'No-Hebbian', 'disable': ['hebbian']},
        {'name': 'No-Diffusion', 'disable': ['diffusion']},
        # Pairwise ablations
        {'name': 'Regret+Attention', 'disable': ['hebbian', 'diffusion']},
        {'name': 'Regret+Hebbian', 'disable': ['attention', 'diffusion']},
        {'name': 'Regret+Diffusion', 'disable': ['attention', 'hebbian']},
        {'name': 'Attention+Hebbian', 'disable': ['counterfactual', 'diffusion']},
        {'name': 'Attention+Diffusion', 'disable': ['counterfactual', 'hebbian']},
        {'name': 'Hebbian+Diffusion', 'disable': ['counterfactual', 'attention']},
    ]
    
    for ablation in ablations:
        print(f"Running ablation: {ablation['name']}")
        # Train with disabled components
        # ...
```

### 7.4 Interventional Causality (Table 4)

```python
# File: experiments/interventions.py

def run_intervention_experiments():
    """
    Test causal effect of IAE on behavior
    Table 4 from paper
    """
    
    # Load trained ESAI agent
    agent = load_checkpoint('checkpoints/esai_best.pt')
    
    interventions = [
        {'name': 'Baseline', 'scale': 1.0},
        {'name': 'Suppression', 'scale': 0.1},
        {'name': 'Amplification', 'scale': 3.0}
    ]
    
    results = []
    
    for intervention in interventions:
        print(f"Testing intervention: {intervention['name']}")
        
        # Collect 100 evaluation episodes
        prosocial_ratios = []
        total_harms = []
        
        for ep in range(100):
            obs, _ = env.reset()
            done = False
            actions_taken = []
            harm_total = 0
            
            while not done:
                # Apply intervention: scale IAE before action selection
                original_E = agent.E.clone()
                agent.E = agent.E * intervention['scale']
                
                # Select action
                action, _ = agent.select_action(obs)
                
                # Restore IAE
                agent.E = original_E
                
                # Step environment
                obs, reward, done, _, info = env.step(action)
                
                actions_taken.append(action)
                harm_total += info.get('harm', 0)
            
            pr = np.mean(np.array(actions_taken) == 4)  # Help action
            prosocial_ratios.append(pr)
            total_harms.append(harm_total)
        
        results.append({
            'intervention': intervention['name'],
            'prosocial_ratio': np.mean(prosocial_ratios),
            'prosocial_ratio_std': np.std(prosocial_ratios),
            'total_harm': np.mean(total_harms),
            'total_harm_std': np.std(total_harms)
        })
    
    # Statistical testing
    from scipy.stats import ttest_ind
    
    baseline_pr = results[0]['prosocial_ratio']
    suppression_pr = results[1]['prosocial_ratio']
    
    t_stat, p_value = ttest_ind(
        prosocial_ratios_baseline,
        prosocial_ratios_suppression
    )
    
    print(f"\nIntervention Results:")
    print(f"Suppression effect: {(suppression_pr - baseline_pr) / baseline_pr * 100:.1f}%")
    print(f"p-value: {p_value:.2e}")
    
    return results
```

### 7.5 Zero-Shot Scaling (Table 6)

```python
# File: experiments/scaling_tests.py

def run_scaling_experiments():
    """
    Train on 4 agents, test on 16 agents (zero-shot)
    Table 6 from paper
    """
    
    # Train on 4 agents
    print("Training on 4 agents...")
    agent_4 = train_esai(num_agents=4, episodes=100000)
    
    # Evaluate on different scales
    scales = [4, 8, 12, 16]
    results = {}
    
    for N in scales:
        print(f"\nEvaluating on {N} agents...")
        
        # Create environment with N agents
        env_N = create_environment(num_agents=N)
        
        # Evaluate (no retraining)
        prosocial_ratios = []
        for ep in range(100):
            obs, _ = env_N.reset()
            done = False
            actions = []
            
            while not done:
                # Use trained 4-agent policy
                action, _ = agent_4.select_action(obs)
                obs, _, done, _, _ = env_N.step(action)
                actions.append(action)
            
            pr = np.mean(np.array(actions) == 4)
            prosocial_ratios.append(pr)
        
        results[N] = {
            'prosocial_ratio': np.mean(prosocial_ratios),
            'std': np.std(prosocial_ratios)
        }
    
    # Compute retention
    pr_4 = results[4]['prosocial_ratio']
    pr_16 = results[16]['prosocial_ratio']
    retention = pr_16 / pr_4
    
    print(f"\n4→16 Agent Retention: {retention:.2%}")
    
    return results
```

### 7.6 IAE-Harm Correlation Analysis (Table 3)

```python
# File: experiments/correlation_analysis.py

def analyze_iae_harm_correlation():
    """
    Validate IAE semantics via correlation with ground-truth harm
    Table 3 from paper
    """
    
    environments = [
        ('Moral Temptation', 'victim_distress'),
        ('Social Distress', 'negative_affect_injection'),
        ('MPE', 'collision_frequency'),
        ('Overcooked', 'plate_breakage'),
        ('SSD', 'resource_depletion')
    ]
    
    results = []
    
    for env_name, harm_metric in environments:
        print(f"\nAnalyzing {env_name}...")
        
        # Load trained agent
        agent = load_checkpoint(f'checkpoints/{env_name}_best.pt')
        env = create_environment(env_name)
        
        # Collect 500 episodes
        iae_norms = []
        harm_values = []
        
        for ep in range(500):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = agent.select_action(obs)
                obs, _, done, _, info = env.step(action)
                
                # Record IAE norm and harm
                iae_norms.append(torch.norm(agent.E).item())
                harm_values.append(info.get(harm_metric, 0))
        
        # Compute Pearson correlation
        from scipy.stats import pearsonr
        r, p_value = pearsonr(iae_norms, harm_values)
        
        results.append({
            'environment': env_name,
            'harm_metric': harm_metric,
            'pearson_r': r,
            'p_value': p_value
        })
        
        print(f"Pearson r = {r:.3f}, p = {p_value:.2e}")
    
    # Overall mean
    mean_r = np.mean([r['pearson_r'] for r in results])
    print(f"\nMean correlation across environments: r = {mean_r:.2f}")
    
    return results
```

---

## Phase 8: Analysis & Paper Generation (Weeks 11-12)

### 8.1 Statistical Analysis
```python
# File: utils/statistical_tests.py

from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np

def run_statistical_tests(esai_metrics, baseline_metrics):
    """
    Run all statistical comparisons from paper
    """
    
    # ESAI vs Raw PPO (Table 2)
    t_stat_pr, p_value_pr = ttest_ind(
        esai_metrics['prosocial_ratio'],
        baseline_metrics['ppo']['prosocial_ratio']
    )
    
    # Cohen's d effect size
    def cohens_d(x1, x2):
        return (np.mean(x1) - np.mean(x2)) / np.sqrt(
            (np.std(x1)**2 + np.std(x2)**2) / 2
        )
    
    d_pr = cohens_d(
        esai_metrics['prosocial_ratio'],
        baseline_metrics['ppo']['prosocial_ratio']
    )
    
    print(f"\nESAI vs Raw PPO:")
    print(f"  Prosocial Ratio: t = {t_stat_pr:.2f}, p < {p_value_pr:.2e}, d = {d_pr:.2f}")
    
    # ESAI vs CPO
    t_stat_cpo, p_value_cpo = ttest_ind(
        esai_metrics['prosocial_ratio'],
        baseline_metrics['cpo']['prosocial_ratio']
    )
    d_cpo = cohens_d(
        esai_metrics['prosocial_ratio'],
        baseline_metrics['cpo']['prosocial_ratio']
    )
    
    print(f"\nESAI vs CPO:")
    print(f"  Prosocial Ratio: t = {t_stat_cpo:.2f}, p = {p_value_cpo:.3f}, d = {d_cpo:.2f}")
    
    return {
        'esai_vs_ppo': {'t': t_stat_pr, 'p': p_value_pr, 'd': d_pr},
        'esai_vs_cpo': {'t': t_stat_cpo, 'p': p_value_cpo, 'd': d_cpo}
    }
```

### 8.2 Generate Results Tables & Figures

```python
# File: generate_paper_results.py

import pandas as pd
import matplotlib.pyplot as plt

def generate_all_results():
    """Generate all tables and figures for paper"""
    
    # Table 2: Primary Performance Metrics
    generate_table_2()
    
    # Table 3: IAE-Harm Correlation
    generate_table_3()
    
    # Table 4: Interventional Causality
    generate_table_4()
    
    # Table 5: Ablation Analysis
    generate_table_5()
    
    # Table 6: Zero-Shot Scaling
    generate_table_6()
    
    # Figure 2: Attention Heatmap
    generate_figure_2()
    
    # Figure 3: IAE Trajectories
    generate_figure_3()

def generate_table_2():
    """Table 2: Primary metrics on Moral Temptation"""
    
    # Load results from all methods
    methods = ['ESAI-v3', 'No-Regret', 'No-Attention', 'No-Hebbian', 
               'No-Diffusion', 'CPO', 'Reward Shaping', 'Multi-Obj RL', 'Raw PPO']
    
    data = []
    for method in methods:
        results = load_results(f'results/{method}_moral_temptation.pkl')
        data.append({
            'Method': method,
            'PR': f"{results['pr_mean']:.2f} ± {results['pr_std']:.2f}",
            'AR': f"{results['ar_mean']:.2f} ± {results['ar_std']:.2f}",
            'ESI': f"{results['esi_mean']:.2f} ± {results['esi_std']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Export to LaTeX
    latex = df.to_latex(index=False, escape=False)
    with open('results/table2.tex', 'w') as f:
        f.write(latex)
    
    print("Table 2 generated!")
    return df

# Similar functions for other tables...
```

---

## Phase 9: Automation with Clawd (Optional)

Since you mentioned using Clawd as an autonomous agent, here's how to set it up:

### 9.1 Clawd Configuration

```python
# File: clawd_config.py

CLAWD_TASKS = {
    "experiment_runner": {
        "description": "Run all ESAI experiments sequentially",
        "command": "python run_all_experiments.sh",
        "duration_estimate": "48 hours",
        "gpu_required": True
    },
    
    "monitor_training": {
        "description": "Monitor tensorboard and check for anomalies",
        "command": "python monitor_training.py",
        "interval": "1 hour"
    },
    
    "generate_results": {
        "description": "Generate all result tables and figures",
        "command": "python generate_paper_results.py",
        "depends_on": ["experiment_runner"]
    },
    
    "hyperparameter_sweep": {
        "description": "Grid search over IAE dimension and learning rate",
        "command": "python hyperparam_sweep.py",
        "duration_estimate": "72 hours",
        "gpu_required": True
    }
}
```

### 9.2 Master Automation Script

```bash
# File: run_all_experiments.sh

#!/bin/bash

set -e  # Exit on error

echo "Starting ESAI Research Automation Pipeline..."

# Phase 1: Main Experiments
echo "Phase 1: Running main experiments..."
for env in moral_temptation social_distress mpe overcooked ssd; do
    python experiments/run_experiment.py --env $env --seeds 5
done

# Phase 2: Baselines
echo "Phase 2: Running baseline comparisons..."
python experiments/run_all_baselines.py

# Phase 3: Ablations
echo "Phase 3: Running ablation studies..."
python experiments/ablations.py

# Phase 4: Interventions
echo "Phase 4: Running intervention probes..."
python experiments/interventions.py

# Phase 5: Scaling tests
echo "Phase 5: Running zero-shot scaling..."
python experiments/scaling_tests.py

# Phase 6: Correlation analysis
echo "Phase 6: Analyzing IAE-harm correlations..."
python experiments/correlation_analysis.py

# Phase 7: Generate results
echo "Phase 7: Generating tables and figures..."
python generate_paper_results.py

echo "All experiments complete!"
echo "Results saved to: results/"
echo "Checkpoints saved to: checkpoints/"
```

---

## Summary Timeline

| **Week** | **Phase** | **Tasks** |
|----------|-----------|-----------|
| 1 | Setup | Environment installation, project structure |
| 2 | Core Implementation | IAE dynamics, counterfactual, attention, Hebbian |
| 3 | Environments | Moral Temptation, Social Distress, MPE wrappers |
| 4 | Baselines | PPO, CPO, reward shaping, multi-objective RL |
| 5 | Training Pipeline | Main training loop, logging, checkpointing |
| 6 | Evaluation | Metrics, visualization, statistical tests |
| 7-8 | Main Experiments | Run all 5 environments × 6 methods × 5 seeds |
| 9 | Ablations & Interventions | 11 ablations + 3 interventions × 100 episodes |
| 10 | Scaling & Correlation | Zero-shot scaling + correlation analysis |
| 11-12 | Analysis & Writing | Generate tables, figures, write paper |

---

## Expected Computational Requirements

- **Total GPU Hours**: ~300 hours (assuming A100 40GB)
- **Storage**: ~50GB for checkpoints and results
- **Experiments**: ~150 total runs (5 envs × 6 methods × 5 seeds)
- **Recommended**: Cloud compute (Lightning AI, Lambda Labs, RunPod)

---

## Next Steps

1. **Start with Phase 1** (environment setup) this week
2. **Implement core ESAI components** (Phases 2-3) over next 2 weeks
3. **Test on Moral Temptation** first before scaling to other environments
4. **Use Weights & Biases** (wandb) for experiment tracking
5. **Version control** everything with git

This workflow mirrors exactly what your papers describe. Start small (single environment, single baseline), validate it works, then scale up systematically. Good luck with your research!