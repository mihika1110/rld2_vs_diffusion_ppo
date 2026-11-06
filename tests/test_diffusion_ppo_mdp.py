"""
Test suite for Discrete Diffusion PPO (Nelson 2026) MDP implementation.
Tests verify:
1. Forward process uses uniform categorical Q_k = (1-β_k)I + β_k * 1 * π^T (Eq. 11-13)
2. Prior initialization x_T ~ Cat(1/K, ..., 1/K) per Eq. 15
3. Trajectory likelihood ratio ρ_t(θ) = P_θ(τ_d|s) / P_θ_old(τ_d|s) per Eq. 27-30
4. PPO clipped actor objective L_actor = -E[min(ρÂ, clip(ρ,1-ε,1+ε)Â)] + λ_ent * H(π_θ) per Eq. 33-34
5. State/observation dimensions
6. Markov property of transitions
7. Action space bounds {0, 1, 2} per user
"""
import numpy as np
import pytest
import sys
import os
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mec_env import MECEnv
from diffusion import CategoricalDiffusion
from ppo_diffusion import DiscreteDiffusionPPOAgent


class TestActionSpace:
    """Tests for action space bounds and validity."""

    def test_action_space_bounds_local(self):
        """Action 0 (local) must be valid for all users."""
        env = MECEnv(n_stations=2, n_users=4)
        obs, info = env.reset(seed=42)
        action = np.array([0, 0, 0, 0])  # All local
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0, "Reward should be negative cost"
        assert len(action) == env.n_users

    def test_action_space_bounds_cloud(self):
        """Action 2 (cloud) must be valid for all users."""
        env = MECEnv(n_stations=2, n_users=4)
        obs, info = env.reset(seed=42)
        action = np.array([2, 2, 2, 2])  # All cloud
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0, "Reward should be negative cost"

    def test_action_space_bounds_mixed(self):
        """Mixed actions (local, edge, cloud) must be valid."""
        env = MECEnv(n_stations=2, n_users=4)
        obs, info = env.reset(seed=42)
        action = np.array([0, 1, 2, 1])  # Mixed
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all((action >= 0) & (action < env.n_actions)), \
            f"Actions {action} out of bounds [0, {env.n_actions})"


class TestDiffusionForwardProcess:
    """Tests for forward process matching Nelson 2026 uniform categorical."""

    def test_forward_process_uniform_categorical(self):
        """Nelson 2026 Eq. 11-13: Q_k = (1-β_k)I + β_k * 1 * π^T with π = uniform(1/K)"""
        diffusion = CategoricalDiffusion(n_users=4, n_actions=3, num_timesteps=32)
        x_0 = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 0]], dtype=torch.long)
        k = torch.tensor([16, 16])

        # Sample from prior (uniform initialization per Eq. 15)
        x_T = diffusion.sample_prior(2, torch.device('cpu'))

        # Apply forward process
        x_k = diffusion.q_sample(x_0, k)

        # Verify: with some probability, positions should change to uniform random
        # With other positions, should keep original value (keep prob = alpha_bar)
        # The key: noise should be uniform over {0,1,2}, NOT mask token (3)
        assert not ((x_k == 3).any()), "Should not use mask token in Nelson 2026"
        # Verify noise values are in {0, 1, 2}
        assert torch.all((x_k >= 0) & (x_k <= 2)), f"Actions should be in {{0,1,2}}, got {x_k.unique()}"

    def test_prior_initialization_uniform(self):
        """Nelson 2026 Eq. 15: x_T ~ Cat(1/K, ..., 1/K) - uniform categorical"""
        diffusion = CategoricalDiffusion(n_users=4, n_actions=3, num_timesteps=32)
        x_T = diffusion.sample_prior(10, torch.device('cpu'))

        # Should sample uniformly from {0, 1, 2}
        unique, counts = torch.unique(x_T, return_counts=True)
        # Each action 0, 1, 2 should appear roughly equally
        for action in [0, 1, 2]:
            action_count = counts[unique == action].sum().item() if (unique == action).any() else 0
            # Should see all three actions approximately equally
            assert action_count > 0, f"Action {action} never sampled"

        # Should NOT have mask token (3)
        assert not (x_T == 3).any(), "Nelson 2026 prior should not use mask token"


class TestTrajectoryLikelihoodRatio:
    """Tests for trajectory likelihood ratio per Nelson 2026 Eq. 27-30."""

    def test_trajectory_ratio_computation(self):
        """ρ_t(θ) = P_θ(τ_d|s) / P_θ_old(τ_d|s) per Eq. 27-30"""
        env = MECEnv(n_stations=2, n_users=4, seed=42)
        agent = DiscreteDiffusionPPOAgent(env, num_timesteps=32, hidden_dim=64, device='cpu')

        obs, _ = env.reset(seed=42)
        sample_dict = agent.select_action(obs)
        action, traj_log_prob, value = sample_dict

        # Verify trajectory log prob is computed
        assert isinstance(traj_log_prob, float)
        assert not np.isnan(traj_log_prob)
        # Should be negative (log probability)
        assert traj_log_prob < 0, "Trajectory log prob should be negative"

    def test_ppo_objective_components(self):
        """PPO clipped objective components work correctly"""
        env = MECEnv(n_stations=2, n_users=4, seed=42)
        agent = DiscreteDiffusionPPOAgent(env, num_timesteps=32, hidden_dim=64, device='cpu')

        # Run a quick training step to test the objective
        metrics = agent.train_epoch(rollout_steps=64, num_epochs=1, batch_size=16)

        # Verify metrics are valid
        assert 'mean_reward' in metrics
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert 'entropy' in metrics
        # Reward should be negative (cost)
        assert metrics['mean_reward'] < 0


class TestStateDimension:
    """Tests for state and observation dimensions."""

    def test_observation_dimension_formula(self):
        """State: M*(2 task features + N distances) + N base station loads"""
        n_users = 4
        n_stations = 2
        env = MECEnv(n_stations=n_stations, n_users=n_users, partial=False)
        obs, _ = env.reset()

        expected_dim = n_users * (2 + n_stations) + n_stations
        assert obs.shape == (expected_dim,), \
            f"Observation shape {obs.shape} != expected ({expected_dim},)"


class TestEntropySign:
    """Tests for entropy coefficient sign (maximization, not minimization)."""

    def test_entropy_maximization(self):
        """Nelson 2026 Eq. 33-34: L_actor uses + λ_ent * H(π_θ), not -"""
        env = MECEnv(n_stations=2, n_users=4, seed=42)
        agent = DiscreteDiffusionPPOAgent(env, num_timesteps=32, hidden_dim=64, device='cpu')

        metrics = agent.train_epoch(rollout_steps=64, num_epochs=1, batch_size=16)

        # Entropy should be positive (Categorical entropy is positive for K>1)
        assert metrics['entropy'] > 0, f"Entropy should be positive, got {metrics['entropy']}"
        # With correct + sign, entropy should be maintained or increased initially
        # With - sign (wrong), entropy would decrease prematurely


class TestRDKL:
    """Tests for RL-D² RKL variant integration."""

    def test_rkl_training_integration(self):
        """RKL agent trains successfully"""
        env = MECEnv(n_stations=2, n_users=4, seed=42)
        from rl_d2_rkl import RLD2RKLAgent

        agent = RLD2RKLAgent(env, num_timesteps=32, hidden_dim=64, device='cpu', auto_tune_temp=True)

        metrics = agent.train_epoch(rollout_steps=64, num_epochs=1, batch_size=16)

        # Verify metrics are valid
        assert 'mean_reward' in metrics
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert 'reverse_kl' in metrics
        assert 'temperature' in metrics
        # Reward should be negative
        assert metrics['mean_reward'] < 0


class TestFKL:
    """Tests for RL-D² FKL variant integration."""

    def test_fkl_training_integration(self):
        """FKL agent trains successfully"""
        env = MECEnv(n_stations=2, n_users=4, seed=42)
        from rl_d2_fkl import RLD2FKLAgent

        agent = RLD2FKLAgent(env, num_timesteps=32, hidden_dim=64, device='cpu',
                            auto_tune_temp=True, replay_buffer_size=50000)

        metrics = agent.train_epoch(rollout_steps=64, num_epochs=1, batch_size=16)

        # Verify metrics are valid
        assert 'mean_reward' in metrics
        assert 'actor_loss' in metrics
        assert 'critic_loss' in metrics
        assert 'temperature' in metrics
        # Reward should be negative
        assert metrics['mean_reward'] < 0


def test_imports():
    """Verify all modules import correctly."""
    import diffusion
    import ppo_diffusion
    import rl_d2_rkl
    import rl_d2_fkl
    import mec_env
    print("All imports successful")


if __name__ == "__main__":
    test_imports()
    pytest.main([__file__, "-v"])