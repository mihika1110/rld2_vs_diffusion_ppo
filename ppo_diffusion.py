"""
Discrete Diffusion PPO for Combinatorial Discrete Action Spaces.
Implementation based on Nelson Sir's formulation:
Treats discrete diffusion generation as an augmented inner Markov process and optimizes
PPO using the likelihood ratio of the complete sampled denoising trajectory.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mec_env import MECEnv
from models import DiffusionDenoiser, CriticNetwork
from diffusion import CategoricalDiffusion


class DiscreteDiffusionPPOAgent:
    """
    Discrete Diffusion PPO agent (Nelson Sir's Formulation).
    Applies PPO over the sampled denoising trajectory tau^d = (x_T, ..., x_0).
    """

    def __init__(
        self,
        env: MECEnv,
        num_timesteps: int = 4,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.005,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.env = env
        self.device = torch.device(device)
        self.num_timesteps = num_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.diffusion = CategoricalDiffusion(
            n_users=env.n_users,
            n_actions=env.n_actions,
            num_timesteps=num_timesteps,
        )

        self.actor = DiffusionDenoiser(
            obs_dim=env.observation_dim,
            n_users=env.n_users,
            n_actions=env.n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = CriticNetwork(
            obs_dim=env.observation_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        top_p: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Samples an action using the reverse discrete diffusion trajectory with optional Top-p.
        """
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            sample_dict = self.diffusion.sample_trajectory(
                self.actor, obs_t, deterministic=deterministic, top_p=top_p
            )
            action = sample_dict["action"].squeeze(0).cpu().numpy()
            traj_log_prob = sample_dict["trajectory_log_prob"].item()
            value = self.critic(obs_t).item()

        return action, traj_log_prob, value

    def train_epoch(
        self,
        rollout_steps: int = 1024,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Collects on-policy diffusion rollouts and performs trajectory PPO updates.
        """
        obs_buf = []
        action_buf = []
        old_traj_lp_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        # Store trajectory tensors: for each step k, store (x_k, x_{prev}, k)
        # We store list of trajectories
        trajectories_buf = []

        cost_metrics = []
        time_metrics = []
        energy_metrics = []
        offload_metrics = []

        obs, _ = self.env.reset()

        # Step 1: Collect rollouts
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for _ in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                sample_dict = self.diffusion.sample_trajectory(self.actor, obs_t)
                action_t = sample_dict["action"]
                traj = sample_dict["trajectory"]  # list of (x_k, x_prev, k_t)
                traj_lp = sample_dict["trajectory_log_prob"]
                value = self.critic(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                obs_buf.append(obs)
                action_buf.append(action_np)
                old_traj_lp_buf.append(traj_lp.item())
                reward_buf.append(reward)
                done_buf.append(done)
                value_buf.append(value.item())

                # Store trajectory items on CPU/tensors
                trajectories_buf.append(
                    [(x_k.squeeze(0).cpu(), x_prev.squeeze(0).cpu(), k.squeeze(0).cpu()) for x_k, x_prev, k in traj]
                )

                cost_metrics.append(info["cost_mean"])
                time_metrics.append(info["time_mean"])
                energy_metrics.append(info["energy_mean"])
                offload_metrics.append(info["offload_ratio"])

                obs = next_obs
                if done:
                    obs, _ = self.env.reset()

            # Bootstrap last value
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_obs_t).item()

        # Step 2: Compute GAE and returns
        values = np.array(value_buf + [last_value], dtype=np.float32)
        rewards = np.array(reward_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        advantages = np.zeros(rollout_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(rollout_steps)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        # Convert to tensors
        b_obs = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
        b_old_traj_lp = torch.as_tensor(np.array(old_traj_lp_buf), dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Pre-stack trajectory tensors for efficient batched mini-batch indexing
        # For each diffusion step k in [0..T-1]
        T = self.num_timesteps
        stacked_xk = []
        stacked_xprev = []
        stacked_k = []
        for step_idx in range(T):
            x_k_step = torch.stack([trajectories_buf[i][step_idx][0] for i in range(rollout_steps)], dim=0).to(self.device)
            x_prev_step = torch.stack([trajectories_buf[i][step_idx][1] for i in range(rollout_steps)], dim=0).to(self.device)
            k_step = torch.stack([trajectories_buf[i][step_idx][2] for i in range(rollout_steps)], dim=0).to(self.device)
            stacked_xk.append(x_k_step)
            stacked_xprev.append(x_prev_step)
            stacked_k.append(k_step)

        # Step 3: Trajectory PPO Optimization Epochs
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        indices = np.arange(rollout_steps)
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_old_lp = b_old_traj_lp[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_ret = b_returns[mb_idx]

                # Reconstruct mini-batch trajectory structure
                mb_traj = []
                for step_idx in range(T):
                    mb_traj.append((
                        stacked_xk[step_idx][mb_idx],
                        stacked_xprev[step_idx][mb_idx],
                        stacked_k[step_idx][mb_idx],
                    ))

                # Re-evaluate trajectory log probability under theta
                new_traj_lp, mean_ent = self.diffusion.evaluate_trajectory_log_prob(
                    self.actor, mb_obs, mb_traj
                )

                # Denoising trajectory likelihood ratio rho_t(theta) (Eq. 28)
                log_ratio = new_traj_lp - mb_old_lp
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                # Clipped actor objective (Eq. 33)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * mean_ent

                # Critic objective (Eq. 35)
                new_val = self.critic(mb_obs)
                critic_loss = 0.5 * F.mse_loss(new_val, mb_ret)

                # Optimizations
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += mean_ent.item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(cost_metrics)),
            "mean_time": float(np.mean(time_metrics)),
            "mean_energy": float(np.mean(energy_metrics)),
            "mean_offload": float(np.mean(offload_metrics)),
            "actor_loss": total_actor_loss / max(1, n_updates),
            "critic_loss": total_critic_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": total_approx_kl / max(1, n_updates),
        }
