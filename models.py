import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Categorical Actor
# Diffusion Denoiser
# Critic Network

#how strongly an input affects the output
def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    #orthogonal matrix initialization for neural network layers
    nn.init.orthogonal_(layer.weight, std) #initialsed to std
    nn.init.constant_(layer.bias, bias_const) #initialsed to bias_const
    return layer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    #noised/masked -> refined - this gives us the timestep embedding 
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


#determines the state made by model - how good/bad it is
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), #128 features
            nn.Tanh(), #add non linearity
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)



class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_users: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.n_users = n_users
        self.n_actions = n_actions

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), #x->128
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.action_head = layer_init(nn.Linear(hidden_dim, n_users * n_actions), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(obs)
        logits = self.action_head(feats)
        return logits.view(-1, self.n_users, self.n_actions)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)  # (B[batch size], M[no of users], K[no of actions])
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_prob, entropy

#comapre with actual target -> improves ppo
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


#input: noisy action->output: clean action
#input:curent state environment + noisy action at time step k  
class DiffusionDenoiser(nn.Module):
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

        #embeddings: vocabulary of size n_actions + 1 (actions 0..n_actions-1, and mask token n_actions)
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
            nn.SiLU(), #SLUU Activation
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
        )

        #transformer encoder layers to capture inter-user offloading coordination
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        #defines coordinatio b/w users
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        #output projection for each user to K action logits
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
        batch_size = obs.shape[0]

        #embed inputs
        act_tokens = self.action_emb(x_k)  # (B, M, hidden_dim)
        act_tokens = act_tokens + self.user_pos_emb  # add user positional identities

        t_tokens = self.time_emb(k).unsqueeze(1)  # (B, 1, hidden_dim)
        s_tokens = self.state_proj(obs).unsqueeze(1)  # (B, 1, hidden_dim)

        #combine: broadcast state and time conditioning onto each user token
        user_tokens = act_tokens + s_tokens + t_tokens  # (B, M, hidden_dim)

        #cross-user attention
        hidden = self.transformer(user_tokens)  # (B, M, hidden_dim)

        #predict logits
        logits = self.out_head(hidden)  # (B, M, K)
        return logits

    def evaluate_step(
        self,
        obs: torch.Tensor,
        x_k: torch.Tensor,
        x_prev: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
        #returns the categorical distribution over x_{k-1}
        logits = self.forward(obs, x_k, k)
        return Categorical(logits=logits)
