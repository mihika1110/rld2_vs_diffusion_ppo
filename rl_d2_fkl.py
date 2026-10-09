"""
RL-D^2 with Forward-KL (FKL) Objective for Combinatorial Discrete Action Spaces.
Implementation based on Ma et al. (2026):
Minimizes forward KL divergence to the Policy Mirror Descent (PMD) target distribution
via advantage-weighted ELBO reconstruction loss:
    L_FKL(theta) = - E [ sum_{a_0} softmax(A(s, a_0) / lambda) * L_ELBO(a_0, s; theta) ]
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mec_env import MECEnv
from models import DiffusionDenoiser, CriticNetwork
from diffusion import CategoricalDiffusion, TemperatureTuner


class RLD2FKLAgent:
    """
    RL-D^2 agent with Forward-KL (FKL) weighted ELBO objective (Ma et al. 2026).
    """

    def __init__(
        self,
        env: MECEnv,
        num_timesteps: int = 4,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        fkl_lambda: float = 0.1,  # initial PMD temperature lambda
        auto_tune_temp: bool = True,  # Appendix C.2
        target_kl: float = 0.05,
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
        self.fkl_lambda = fkl_lambda
        self.auto_tune_temp = auto_tune_temp
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.diffusion = CategoricalDiffusion(
            n_users=env.n_users,
            n_actions=env.n_actions,
            num_timesteps=num_timesteps,
        )

        if self.auto_tune_temp:
            self.temp_tuner = TemperatureTuner(init_temp=fkl_lambda, target_kl=target_kl, lr=1e-2)
        else:
            self.temp_tuner = None

        self.actor = DiffusionDenoiser(
            obs_dim=env.observation_dim,
            n_users=env.n_users,
            n_actions=env.n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.old_actor = DiffusionDenoiser(
            obs_dim=env.observation_dim,
            n_users=env.n_users,
            n_actions=env.n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.eval()

        self.critic = CriticNetwork(
            obs_dim=env.observation_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

    def update_old_actor(self) -> None:
        """Sync old actor with current actor weights."""
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_actor.eval()

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        top_p: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Samples an action using the learned reverse discrete diffusion process with optional Top-p.
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
        Collects rollouts and optimizes the Forward-KL (FKL) weighted ELBO objective.
        """
        self.update_old_actor()

        obs_buf = []
        action_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        cost_metrics = []
        time_metrics = []
        energy_metrics = []
        offload_metrics = []

        obs, _ = self.env.reset()

        # Step 1: Collect rollouts using old policy
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for _ in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                sample_dict = self.diffusion.sample_trajectory(self.old_actor, obs_t)
                action_t = sample_dict["action"]
                value = self.critic(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                obs_buf.append(obs)
                action_buf.append(action_np)
                reward_buf.append(reward)
                done_buf.append(done)
                value_buf.append(value.item())

                cost_metrics.append(info["cost_mean"])
                time_metrics.append(info["time_mean"])
                energy_metrics.append(info["energy_mean"])
                offload_metrics.append(info["offload_ratio"])

                obs = next_obs
                if done:
                    obs, _ = self.env.reset()

            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_obs_t).item()

        # Step 2: GAE Advantage Estimation
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

        b_obs = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
        b_clean_actions = torch.as_tensor(np.array(action_buf), dtype=torch.long, device=self.device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Appendix C.2: Auto-tune temperature parameter lambda
        if self.temp_tuner is not None:
            curr_lambda = self.temp_tuner.update(b_advantages)
        else:
            curr_lambda = self.fkl_lambda

        # Step 3: FKL Policy Improvement via Weighted ELBO
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(rollout_steps)
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_acts = b_clean_actions[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_ret = b_returns[mb_idx]

                # FKL Weighted ELBO Loss (Eq. FKL Loss in Ma et al. 2026)
                weighted_elbo_loss, mean_ent = self.diffusion.evaluate_fkl_weighted_elbo(
                    denoiser=self.actor,
                    obs=mb_obs,
                    clean_actions=mb_acts,
                    advantages=mb_adv,
                    temperature=curr_lambda,
                )

                actor_loss = weighted_elbo_loss - self.ent_coef * mean_ent

                # Critic Loss
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
                n_updates += 1

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_cost": float(np.mean(cost_metrics)),
            "mean_time": float(np.mean(time_metrics)),
            "mean_energy": float(np.mean(energy_metrics)),
            "mean_offload": float(np.mean(offload_metrics)),
            "actor_loss": total_actor_loss / max(1, n_updates),
            "critic_loss": total_critic_loss / max(1, n_updates),
            "temperature": float(curr_lambda),
            "entropy": total_entropy / max(1, n_updates),
        }
