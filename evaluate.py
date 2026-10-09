"""
Evaluation suite for MEC policies:
- Random policy
- Greedy policy (instantaneous weighted cost minimization)
- Categorical PPO baseline
- Discrete Diffusion PPO (Nelson Sir's Formulation)
- RL-D^2 with RKL (Ma et al. 2026)
- RL-D^2 with FKL (Ma et al. 2026)
With support for Top-p (nucleus) sampling (App. C.3).
"""

from typing import Dict, List, Any, Optional
import time
import numpy as np
import torch
import pandas as pd

from mec_env import MECEnv


def evaluate_policy(
    policy_type: str,
    env: MECEnv,
    agent: Optional[Any] = None,
    num_episodes: int = 5,
    seed: int = 100,
    deterministic: bool = True,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates a policy over multiple full episodes.
    """
    returns = []
    costs = []
    times = []
    energies = []
    offload_ratios = []
    user_cpus = []
    bs_cpus = []
    step_latencies = []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 1000
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0

        while not done:
            t0 = time.perf_counter()
            if policy_type == "random":
                action = env.sample_random_action()
            elif policy_type == "greedy":
                action = env.get_greedy_action()
            elif policy_type in ["ppo_baseline"]:
                if agent is None:
                    raise ValueError(f"Agent instance required for policy_type '{policy_type}'")
                action, _, _ = agent.select_action(obs, deterministic=deterministic)
            elif policy_type in ["diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]:
                if agent is None:
                    raise ValueError(f"Agent instance required for policy_type '{policy_type}'")
                # Pass top_p if supported
                if hasattr(agent, "select_action"):
                    import inspect
                    sig = inspect.signature(agent.select_action)
                    if "top_p" in sig.parameters:
                        action, _, _ = agent.select_action(obs, deterministic=deterministic, top_p=top_p)
                    else:
                        action, _, _ = agent.select_action(obs, deterministic=deterministic)
                else:
                    action, _, _ = agent.select_action(obs, deterministic=deterministic)
            else:
                raise ValueError(f"Unknown policy_type '{policy_type}'")
            t1 = time.perf_counter()

            step_latencies.append((t1 - t0) * 1000.0)  # ms

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_return += reward
            costs.append(step_info["cost_mean"])
            times.append(step_info["time_mean"])
            energies.append(step_info["energy_mean"])
            offload_ratios.append(step_info["offload_ratio"])
            user_cpus.append(step_info["user_cpu_capacity"])
            bs_cpus.append(step_info["bs_cpu_capacity"])

            done = terminated or truncated

        returns.append(ep_return)

    return {
        "policy": policy_type,
        "n_users": env.n_users,
        "n_stations": env.n_stations,
        "episodes": num_episodes,
        "top_p": top_p,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_latency_s": float(np.mean(times)),
        "mean_energy_j": float(np.mean(energies)),
        "mean_offload_pct": float(np.mean(offload_ratios) * 100.0),
        "mean_user_cpu": float(np.mean(user_cpus)),
        "mean_bs_cpu": float(np.mean(bs_cpus)),
        "decision_time_ms": float(np.mean(step_latencies)),
    }


def print_evaluation_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prints a formatted evaluation table."""
    df = pd.DataFrame(results)
    cols = [
        "policy",
        "n_users",
        "mean_return",
        "mean_cost",
        "mean_latency_s",
        "mean_energy_j",
        "mean_offload_pct",
        "decision_time_ms",
    ]
    display_df = df[[c for c in cols if c in df.columns]].copy()
    print("\n" + "=" * 90)
    print("MEC POLICY EVALUATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(display_df.to_string(index=False))
    print("=" * 90 + "\n")
    return df
