from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence

#diffusion works by corrupting the action and then learning to reverse the corruption
def top_p_filter(logits: torch.Tensor, p: float = 0.98) -> torch.Tensor:
    if p >= 1.0 or p <= 0.0:
        return logits

    #sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs=torch.cumsum(sorted_probs, dim=-1)

    #mask tokens with cumulative probability > p (except always keep at least the first/highest token)
    sorted_mask=cum_probs-sorted_probs > p

    #set masked logits to -infinity
    sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))

    #scatter back to original indices
    filtered_logits = torch.empty_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return filtered_logits

#adjusts temperature and controls the distribution of probabilities of outcomes
class TemperatureTuner:
    def __init__(
        self,
        init_temp: float = 0.2,
        target_kl: float = 0.5,
        lr: float = 1e-2,
        min_temp: float = 0.01,
        max_temp: float = 1.0,
    ):
        self.target_kl = target_kl
        self.min_temp = min_temp
        self.max_temp = max_temp
        #parameterize as log_temp to guarantee lambda>0
        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-4, init_temp)), dtype=torch.float32))
        self.optimizer = optim.Adam([self.log_temp], lr=lr)

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())

    def get_temperature_tensor(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def update(self, advantages: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        temp = torch.exp(self.log_temp)  # lambda > 0

        # standardize advantages across the batch
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_adv = (advantages - adv_mean) / adv_std

        #stabilized logsumexp: lambda * (logsumexp(norm_adv / lambda) - log(B))
        batch_size = norm_adv.shape[0]
        scaled_adv = norm_adv.detach() / temp
        log_z = torch.logsumexp(scaled_adv, dim=0) - math.log(batch_size)

        dual_loss = temp * self.target_kl + temp * log_z
        dual_loss.backward()
        self.optimizer.step()

        # clamp temperature to configured safe range [min_temp, max_temp]
        with torch.no_grad():
            self.log_temp.clamp_(math.log(self.min_temp), math.log(self.max_temp))

        return float(temp.item())

class CategoricalDiffusion:
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

        #linear beta schedule for forward process: Q_k = (1 - beta_k) I + beta_k 1 π^T
        # where π = [1/K, ..., 1/K] is the uniform categorical distribution (Nelson 2026, Eq. 10-11)
        self.betas = np.linspace(beta_min, beta_max, num_timesteps, dtype=np.float32)
        #cumulative alphas: alpha_bar_k = prod_{j=1}^k (1 - beta_j)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas)

    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Eq. 15: x_T ~ Cat(1/K, ..., 1/K) - uniform categorical initialization"""
        return torch.randint(
            low=0,
            high=self.n_actions,
            size=(batch_size, self.n_users),
            dtype=torch.long,
            device=device,
        )

    def q_sample(self, x_0: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Eq. 13: Forward process q(x_k | x_{k-1}) with uniform categorical mixing.

        Keep with prob alpha_bar, else sample uniform from {0, ..., K-1}.
        Unlike Ma 2026, NO mask token - noise is uniform over actions.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        k_cpu = k.cpu().numpy() - 1  # 0-indexed
        alpha_bars = torch.as_tensor(self.alphas_cumprod[k_cpu], dtype=torch.float32, device=device).unsqueeze(1)

        # Forward process with uniform categorical mixing (Nelson 2026)
        # Q_k(j, l) = 1-β_k + β_k/K if j=l, β_k/K if j≠l
        # Implementation: keep with prob alpha_bar, else uniform random
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

            # Nelson 2026 reverse: always sample from distribution, no mask logic
            # If deterministic and k == 1: take argmax, else sample
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

            #single-step importance ratio (Eq. in Ma et al. 2026 Sec. 4.1)
            step_ratio = torch.exp(curr_lp - old_lp)
            ratios_list.append(step_ratio)

            #analytical Reverse KL divergence across users
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
        temperature: float = 0.5,
        max_weight_factor: float = 4.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = obs.shape[0]
        device = obs.device

        # Standardize advantages across the batch
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        norm_adv = (advantages - adv_mean) / adv_std

        # Exponential weights based on standardized advantages and temperature (Ma et al. 2026 FKL Loss Eq.)
        temp = max(1e-4, temperature)
        weights = torch.exp(norm_adv / temp).detach()  # (B,)

        # Normalize weights
        weights = weights / weights.sum()

        #sample random diffusion timestep k in {1, ..., T} for each batch item
        k = torch.randint(1, self.num_timesteps + 1, size=(batch_size,), device=device)

        #corrupt clean action using forward process q(x_k | x_0)
        x_k = self.q_sample(clean_actions, k)

        #predict clean/reverse logits from noised action
        logits = denoiser(obs, x_k, k)  # (B, M, K)
        dist = Categorical(logits=logits)

        #reconstruction log likelihood for clean actions
        recon_log_prob = dist.log_prob(clean_actions).sum(dim=-1)  # (B,)

        #weighted ELBO loss
        weighted_loss = -torch.sum(weights * recon_log_prob)
        mean_entropy = dist.entropy().sum(dim=-1).mean()

        return weighted_loss, mean_entropy