"""
Independent Categorical PPO Baseline for Multi-User MEC.
Predicts factorized independent categorical distributions for each user.
"""

from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mec_env import MECEnv
from models import CategoricalActor, CriticNetwork


class CategoricalPPOAgent:
    """
    Independent Categorical PPO algorithm.
    """

    def __init__(
        self,
        env: MECEnv,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.actor = CategoricalActor(
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

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Samples an action for environment execution.
        """
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_t, log_prob_t, _ = self.actor.sample(obs_t, deterministic=deterministic)
            value_t = self.critic(obs_t)

        return (
            actions_t.squeeze(0).cpu().numpy(),
            log_prob_t.item(),
            value_t.item(),
        )

    def train_epoch(
        self,
        rollout_steps: int = 1024,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Collects on-policy rollouts and performs PPO updates.
        """
        # Storage
        obs_buf = []
        action_buf = []
        log_prob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

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
                action_t, log_prob_t, _ = self.actor.sample(obs_t)
                value_t = self.critic(obs_t)

                action_np = action_t.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                obs_buf.append(obs)
                action_buf.append(action_np)
                log_prob_buf.append(log_prob_t.item())
                reward_buf.append(reward)
                done_buf.append(done)
                value_buf.append(value_t.item())

                cost_metrics.append(info["cost_mean"])
                time_metrics.append(info["time_mean"])
                energy_metrics.append(info["energy_mean"])
                offload_metrics.append(info["offload_ratio"])

                obs = next_obs
                if done:
                    obs, _ = self.env.reset()

            # Bootstrap value for final step
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
        b_actions = torch.as_tensor(np.array(action_buf), dtype=torch.long, device=self.device)
        b_log_probs = torch.as_tensor(np.array(log_prob_buf), dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Step 3: PPO Optimization Epochs
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
                mb_acts = b_actions[mb_idx]
                mb_old_lp = b_log_probs[mb_idx]
                mb_adv = b_advantages[mb_idx]
                mb_ret = b_returns[mb_idx]

                # Actor evaluation
                new_lp, entropy = self.actor.evaluate_actions(mb_obs, mb_acts)
                log_ratio = new_lp - mb_old_lp
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy.mean()

                # Critic objective
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
                total_entropy += entropy.mean().item()
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
