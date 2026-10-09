"""
Discrete Diffusion processes, trajectory handling, and PMD objectives.
Supports:
1. Categorical discrete forward transition schedules and noising q(x_k | x_0).
2. Reverse denoising trajectory generation with optional Top-p (nucleus) sampling (App. C.3).
3. Trajectory log-probability calculation for Discrete Diffusion PPO (Nelson Sir's Formulation).
4. Single-step ratio and analytical Reverse-KL calculation for RL-D^2 RKL (Ma et al. 2026).
5. Weighted ELBO calculation for RL-D^2 Forward-KL (FKL) (Ma et al. 2026).
6. Automatic Temperature Tuning via dual hard-KL constraints (Ma et al. 2026 App. C.2).
"""

from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence


def top_p_filter(logits: torch.Tensor, p: float = 0.98) -> torch.Tensor:
    """
    Nucleus (Top-p) filtering for categorical logits.
    Retains only the smallest set of actions whose cumulative probability exceeds p.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative probability > p (except always keep at least the first/highest token)
    sorted_mask = cum_probs - sorted_probs > p

    # Set masked logits to -infinity
    sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))

    # Scatter back to original indices
    filtered_logits = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return filtered_logits


class TemperatureTuner:
    """
    Automatic Temperature Tuning via Hard KL Divergence Constraints (App. C.2 of Ma et al. 2026).
    Enforces constraint: d_KL(pi, pi_old) <= eps_kl.
    Solves Lagrangian dual:
        min_lambda g(lambda) = lambda * eps_kl + lambda * log(1/B * sum exp(A_i / lambda))
    """

    def __init__(self, init_temp: float = 0.1, target_kl: float = 0.05, lr: float = 1e-2):
        self.target_kl = target_kl
        # Parameterize as log_temp to guarantee lambda > 0
        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-4, init_temp)), dtype=torch.float32))
        self.optimizer = optim.Adam([self.log_temp], lr=lr)

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())

    def get_temperature_tensor(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def update(self, advantages: torch.Tensor) -> float:
        """
        Updates temperature lambda by minimizing dual objective g(lambda).
        advantages: (B,) 1D tensor
        """
        self.optimizer.zero_grad()
        temp = torch.exp(self.log_temp)  # lambda > 0

        # Stabilized logsumexp: lambda * (logsumexp(A / lambda) - log(B))
        batch_size = advantages.shape[0]
        scaled_adv = advantages.detach() / temp
        log_z = torch.logsumexp(scaled_adv, dim=0) - math.log(batch_size)

        dual_loss = temp * self.target_kl + temp * log_z
        dual_loss.backward()
        self.optimizer.step()

        return float(temp.item())


class CategoricalDiffusion:
    """
    Discrete diffusion process manager for combinatorial action spaces.
    """

    def __init__(
        self,
        n_users: int,
        n_actions: int,
        num_timesteps: int = 4,
        beta_min: float = 0.05,
        beta_max: float = 0.8,
    ):
        self.n_users = n_users
        self.n_actions = n_actions
        self.num_timesteps = num_timesteps
        self.K = n_actions
        self.M = n_users

        # Linear beta schedule for forward process: Q_k = (1 - beta_k) I + beta_k 1 pi^T
        self.betas = np.linspace(beta_min, beta_max, num_timesteps, dtype=np.float32)
        # Cumulative alphas: alpha_bar_k = prod_{j=1}^k (1 - beta_j)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)

    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample initial noise state x_T from uniform categorical distribution.
        """
        return torch.randint(
            low=0,
            high=self.n_actions,
            size=(batch_size, self.n_users),
            dtype=torch.long,
            device=device,
        )

    def q_sample(self, x_0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Forward process sampling: adds noise to clean action x_0 at diffusion step k.
        With probability alpha_bar_k, keep x_0; with probability 1 - alpha_bar_k, sample uniform action.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        k_cpu = k.cpu().numpy() - 1  # 0-indexed
        alpha_bars = torch.as_tensor(self.alphas_cumprod[k_cpu], dtype=torch.float32, device=device).unsqueeze(1)

        # Mask whether to retain or corrupt
        keep_mask = torch.rand((batch_size, self.n_users), device=device) < alpha_bars
        noise = torch.randint(0, self.n_actions, size=(batch_size, self.n_users), device=device)

        return torch.where(keep_mask, x_0, noise)

    @torch.no_grad()
    def sample_trajectory(
        self,
        denoiser: nn.Module,
        obs: torch.Tensor,
        deterministic: bool = False,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Samples a complete reverse denoising trajectory:
        x_T -> x_{T-1} -> ... -> x_0 = a_t
        """
        batch_size = obs.shape[0]
        device = obs.device

        x_curr = self.sample_prior(batch_size, device)

        trajectory = []
        step_log_probs = []
        total_traj_log_prob = torch.zeros(batch_size, device=device)
        c_prior = self.n_users * math.log(1.0 / self.n_actions)

        for k in range(self.num_timesteps, 0, -1):
            k_tensor = torch.full((batch_size,), k, dtype=torch.long, device=device)
            logits = denoiser(obs, x_curr, k_tensor)  # (B, M, K)

            if top_p is not None and top_p < 1.0:
                logits = top_p_filter(logits, p=top_p)

            dist = Categorical(logits=logits)

            if deterministic and k == 1:
                x_prev = torch.argmax(logits, dim=-1)
            else:
                x_prev = dist.sample()

            step_lp = dist.log_prob(x_prev).sum(dim=-1)

            trajectory.append((x_curr.clone(), x_prev.clone(), k_tensor))
            step_log_probs.append(step_lp)
            total_traj_log_prob += step_lp

            x_curr = x_prev

        final_action = x_curr
        total_traj_log_prob = total_traj_log_prob + c_prior

        return {
            "action": final_action,
            "trajectory": trajectory,
            "trajectory_log_prob": total_traj_log_prob,
            "step_log_probs": torch.stack(step_log_probs, dim=1),
        }

    def evaluate_trajectory_log_prob(
        self,
        denoiser: nn.Module,
        obs: torch.Tensor,
        trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates log-probability of stored trajectory under denoiser parameters theta.
        """
        batch_size = obs.shape[0]
        device = obs.device
        total_log_prob = torch.zeros(batch_size, device=device)
        total_entropy = torch.zeros(batch_size, device=device)
        c_prior = self.n_users * math.log(1.0 / self.n_actions)

        for x_k, x_prev, k_tensor in trajectory:
            logits = denoiser(obs, x_k, k_tensor)
            dist = Categorical(logits=logits)
            step_lp = dist.log_prob(x_prev).sum(dim=-1)
            step_ent = dist.entropy().sum(dim=-1)

            total_log_prob += step_lp
            total_entropy += step_ent

        return total_log_prob + c_prior, total_entropy.mean()

    def evaluate_rkl_steps(
        self,
        denoiser: nn.Module,
        old_denoiser: nn.Module,
        obs: torch.Tensor,
        trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates per-step single-step ratios eta_k and Reverse-KL divergences
        for RL-D^2 with RKL (Ma et al. 2026).
        """
        batch_size = obs.shape[0]
        device = obs.device
        ratios_list = []
        kls_list = []
        entropies_list = []

        for x_k, x_prev, k_tensor in trajectory:
            curr_logits = denoiser(obs, x_k, k_tensor)
            curr_dist = Categorical(logits=curr_logits)

            with torch.no_grad():
                old_logits = old_denoiser(obs, x_k, k_tensor)
                old_dist = Categorical(logits=old_logits)
                old_lp = old_dist.log_prob(x_prev).sum(dim=-1)

            curr_lp = curr_dist.log_prob(x_prev).sum(dim=-1)

            # Single-step importance ratio
            step_ratio = torch.exp(curr_lp - old_lp)
            ratios_list.append(step_ratio)

            # Analytical Reverse KL divergence across users
            step_kl = kl_divergence(curr_dist, old_dist).sum(dim=-1)
            kls_list.append(step_kl)

            entropies_list.append(curr_dist.entropy().sum(dim=-1))

        step_ratios = torch.stack(ratios_list, dim=1)  # (B, T)
        step_kls = torch.stack(kls_list, dim=1)        # (B, T)
        mean_entropy = torch.stack(entropies_list, dim=1).mean()

        return step_ratios, step_kls, mean_entropy

    def evaluate_fkl_weighted_elbo(
        self,
        denoiser: nn.Module,
        obs: torch.Tensor,
        clean_actions: torch.Tensor,
        advantages: torch.Tensor,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates Forward-KL (FKL) weighted ELBO loss (Eq. FKL Loss in Ma et al. 2026):
            L_FKL = - sum_{i} w_i * L_ELBO(a_0^i, s; theta)
        where weights w_i = softmax(A(s, a_0^i) / lambda).
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Softmax weights based on advantages
        weights = F.softmax(advantages / max(1e-4, temperature), dim=0).detach()  # (B,)

        # Sample random diffusion timestep k in {1, ..., T} for each batch item
        k = torch.randint(1, self.num_timesteps + 1, size=(batch_size,), device=device)

        # Corrupt clean action using forward process q(x_k | x_0)
        x_k = self.q_sample(clean_actions, k)

        # Predict clean/reverse logits from noised action
        logits = denoiser(obs, x_k, k)  # (B, M, K)
        dist = Categorical(logits=logits)

        # Reconstruction log likelihood for clean actions
        recon_log_prob = dist.log_prob(clean_actions).sum(dim=-1)  # (B,)

        # Weighted ELBO loss
        weighted_loss = -torch.sum(weights * recon_log_prob)
        mean_entropy = dist.entropy().sum(dim=-1).mean()

        return weighted_loss, mean_entropy
