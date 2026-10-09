"""
RL-D^2 with Reverse-KL (RKL) Objective for Combinatorial Discrete Action Spaces.
Implementation based on Ma et al. (2026):
Treats policy improvement as a distributional matching problem to the Policy Mirror Descent (PMD)
target distribution, utilizing tractable single-step denoising ratios in an augmented MDP,
PPO-style clipping, and analytical Reverse-KL regularization with automatic temperature tuning (App. C.2).
"""

from typing import Dict, List, Tuple, Optional, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mec_env import MECEnv
from models import DiffusionDenoiser, CriticNetwork
from diffusion import CategoricalDiffusion, TemperatureTuner


class RLD2RKLAgent:
    """
    RL-D^2 agent with Reverse-KL (RKL) objective and automatic temperature tuning (Ma et al. 2026).
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
        rkl_lambda: float = 0.05,  # initial PMD Reverse-KL temperature
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
        self.clip_coef = clip_coef
        self.rkl_lambda = rkl_lambda
        self.auto_tune_temp = auto_tune_temp
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.diffusion = CategoricalDiffusion(
            n_users=env.n_users,
            n_actions=env.n_actions,
            num_timesteps=num_timesteps,
        )

        if self.auto_tune_temp:
            self.temp_tuner = TemperatureTuner(init_temp=rkl_lambda, target_kl=target_kl, lr=1e-2)
        else:
            self.temp_tuner = None

        self.actor = DiffusionDenoiser(
            obs_dim=env.observation_dim,
            n_users=env.n_users,
            n_actions=env.n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Frozen reference network
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
        Collects on-policy rollouts and optimizes the Reverse-KL objective with single-step clipping.
        """
        self.update_old_actor()

        obs_buf = []
        action_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        trajectories_buf = []

        cost_metrics = []
        time_metrics = []
        energy_metrics = []
        offload_metrics = []

        obs, _ = self.env.reset()

        # Step 1: On-Policy Diffusion Data Collection (Section 4.2)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            for _ in range(rollout_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                sample_dict = self.diffusion.sample_trajectory(self.old_actor, obs_t)
                action_t = sample_dict["action"]
                traj = sample_dict["trajectory"]
                value = self.critic(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                obs_buf.append(obs)
                action_buf.append(action_np)
                reward_buf.append(reward)
                done_buf.append(done)
                value_buf.append(value.item())

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

            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.critic(last_obs_t).item()

        # Step 2: Compute Advantages and Returns
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
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Appendix C.2: Temperature auto-tuning using raw advantages
        if self.temp_tuner is not None:
            curr_lambda = self.temp_tuner.update(b_advantages)
        else:
            curr_lambda = self.rkl_lambda

        # Normalize advantages for policy gradient
        b_adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Pre-stack trajectories
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

        # Step 3: RKL Policy Improvement Optimization
        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_rkl = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(rollout_steps)
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_adv = b_adv_norm[mb_idx]
                mb_ret = b_returns[mb_idx]

                mb_traj = []
                for step_idx in range(T):
                    mb_traj.append((
                        stacked_xk[step_idx][mb_idx],
                        stacked_xprev[step_idx][mb_idx],
                        stacked_k[step_idx][mb_idx],
                    ))

                # Evaluate single-step ratios and analytical Reverse-KL
                step_ratios, step_kls, mean_ent = self.diffusion.evaluate_rkl_steps(
                    self.actor, self.old_actor, mb_obs, mb_traj
                )

                adv_expanded = mb_adv.unsqueeze(1).expand_as(step_ratios)

                # PPO-style single-step clipping
                surr1 = step_ratios * adv_expanded
                surr2 = torch.clamp(step_ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_expanded
                clipped_pg = torch.min(surr1, surr2)

                # RKL Objective with (optionally auto-tuned) PMD temperature curr_lambda
                actor_loss = - (clipped_pg - curr_lambda * step_kls).mean() - self.ent_coef * mean_ent

                # Critic loss
                new_val = self.critic(mb_obs)
                critic_loss = 0.5 * F.mse_loss(new_val, mb_ret)

                # Updates
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
                total_rkl += step_kls.mean().item()
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
            "reverse_kl": total_rkl / max(1, n_updates),
            "temperature": float(curr_lambda),
            "entropy": total_entropy / max(1, n_updates),
        }
