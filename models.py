"""
Neural network architectures for MEC RL algorithms.
Includes:
1. CategoricalActor: Independent Categorical policy for baseline PPO.
2. DiffusionDenoiser: Discrete diffusion denoiser conditioning on state and timestep,
   with multi-head attention across users for combinatorial coordination.
3. CriticNetwork: State-value network V(s) for GAE advantage estimation.
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for neural network layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep k."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class CriticNetwork(nn.Module):
    """
    State-value estimator V(s).
    Maps observation s of shape (B, obs_dim) to scalar value (B, 1).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class CategoricalActor(nn.Module):
    """
    Independent Categorical Actor for multi-user offloading baseline.
    Predicts independent categorical logits for each of the M users over K actions.
    Total joint action space: K^M.
    Factorized policy: pi(a | s) = prod_{m=1}^M pi(a_m | s).
    """

    def __init__(self, obs_dim: int, n_users: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.n_users = n_users
        self.n_actions = n_actions

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.action_head = layer_init(nn.Linear(hidden_dim, n_users * n_actions), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, M, K)."""
        feats = self.backbone(obs)
        logits = self.action_head(feats)
        return logits.view(-1, self.n_users, self.n_actions)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples joint action vector of length M.
        
        Returns:
            actions: (B, M) int64
            log_prob: (B,) float32, sum over users
            entropy: (B,) float32, sum over users
        """
        logits = self.forward(obs)  # (B, M, K)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates log-prob and entropy of given actions.
        
        Returns:
            log_prob: (B,) float32
            entropy: (B,) float32
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class DiffusionDenoiser(nn.Module):
    """
    Neural Denoiser f_theta(s_t, x_k, k) for Discrete Diffusion Policy.
    
    Inputs:
        s_t: Environment state observation of shape (B, obs_dim).
        x_k: Noisy discrete action of shape (B, M) with values in {0, ..., K-1}.
        k: Diffusion timestep integer tensor of shape (B,) with values in {1, ..., T}.
        
    Outputs:
        Reverse transition logits of shape (B, M, K).
        Each user's next action distribution p_theta(x_{k-1}^m | x_k, s_t, k).
    """

    def __init__(
        self,
        obs_dim: int,
        n_users: int,
        n_actions: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_users = n_users
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Embeddings: vocabulary of size n_actions + 1 (actions 0..n_actions-1, and mask token n_actions)
        self.action_emb = nn.Embedding(n_actions + 1, hidden_dim)
        self.user_pos_emb = nn.Parameter(torch.zeros(1, n_users, hidden_dim))
        nn.init.normal_(self.user_pos_emb, std=0.02)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )

        self.state_proj = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )

        # Transformer encoder layers to capture inter-user offloading coordination
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Output projection for each user to K action logits
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, n_actions), std=0.01),
        )

    def forward(
        self,
        obs: torch.Tensor,
        x_k: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass predicting logits for each user's reverse transition.
        
        Args:
            obs: (B, obs_dim)
            x_k: (B, M) int64 in [0, K-1]
            k: (B,) int64 in [1, T]
            
        Returns:
            logits: (B, M, K)
        """
        batch_size = obs.shape[0]

        # Embed inputs
        act_tokens = self.action_emb(x_k)  # (B, M, hidden_dim)
        act_tokens = act_tokens + self.user_pos_emb  # add user positional identities

        t_tokens = self.time_emb(k).unsqueeze(1)  # (B, 1, hidden_dim)
        s_tokens = self.state_proj(obs).unsqueeze(1)  # (B, 1, hidden_dim)

        # Combine: broadcast state and time conditioning onto each user token
        user_tokens = act_tokens + s_tokens + t_tokens  # (B, M, hidden_dim)

        # Cross-user attention
        hidden = self.transformer(user_tokens)  # (B, M, hidden_dim)

        # Predict logits
        logits = self.out_head(hidden)  # (B, M, K)
        return logits

    def evaluate_step(
        self,
        obs: torch.Tensor,
        x_k: torch.Tensor,
        x_prev: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probability and entropy of transition x_k -> x_{prev}.
        
        Returns:
            log_prob: (B,) summed across users
            entropy: (B,) summed across users
        """
        logits = self.forward(obs, x_k, k)  # (B, M, K)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(x_prev).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def step_distribution(
        self,
        obs: torch.Tensor,
        x_k: torch.Tensor,
        k: torch.Tensor,
    ) -> Categorical:
        """Returns the Categorical distribution over x_{k-1}."""
        logits = self.forward(obs, x_k, k)
        return Categorical(logits=logits)
