"""
Standardized Checkpoint Evaluation and Visualization Re-generator.
Evaluates all saved checkpoints on identical, shared test episodes (eval_seed=1000, 10 episodes),
generates fair, unskewed comparisons, and re-renders all 4 high-res figures.
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from mec_env import MECEnv
from ppo_baseline import CategoricalPPOAgent
from ppo_diffusion import DiscreteDiffusionPPOAgent
from rl_d2_rkl import RLD2RKLAgent
from rl_d2_fkl import RLD2FKLAgent
from evaluate import evaluate_policy, print_evaluation_summary
from train import plot_comparison_results, plot_scalability_analysis


def run_standardized_checkpoint_evaluation(
    user_counts: List[int] = [2, 4, 8],
    seeds: List[int] = [42, 43],
    num_test_episodes: int = 10,
    shared_eval_seed: int = 1000,
    diffusion_timesteps: int = 4,
    device: str = "cpu",
):
    print("================================================================================")
    print("STANDARDIZED CHECKPOINT EVALUATION BENCHMARK")
    print(f"Shared Evaluation Seed: {shared_eval_seed} | Test Episodes per Policy: {num_test_episodes}")
    print("All policies are evaluated on the EXACT SAME test environments and task arrivals.")
    print("================================================================================\n")

    # Load existing training history
    history_file = "results/training_history.csv"
    if os.path.exists(history_file):
        df_all_history = pd.read_csv(history_file)
    else:
        df_all_history = pd.DataFrame()

    all_eval = []
    algos = ["ppo_baseline", "diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]

    for n_users in user_counts:
        print(f"\n--- Evaluating Scale: M = {n_users} Users (Action Space: 3^{n_users} = {3**n_users}) ---")
        eval_env = MECEnv(n_stations=2, n_users=n_users, seed=999)

        # 1. Standardized evaluation of baseline heuristics on the shared test set
        for seed in seeds:
            rand_res = evaluate_policy("random", eval_env, num_episodes=num_test_episodes, seed=shared_eval_seed)
            rand_res["seed"] = seed
            greedy_res = evaluate_policy("greedy", eval_env, num_episodes=num_test_episodes, seed=shared_eval_seed)
            greedy_res["seed"] = seed
            all_eval.extend([rand_res, greedy_res])

        # 2. Evaluate saved trained agents
        for seed in seeds:
            for algo in algos:
                ckpt_path = f"checkpoints/{algo}_u{n_users}_s{seed}.pt"
                if not os.path.exists(ckpt_path):
                    print(f"Warning: Checkpoint {ckpt_path} not found. Skipping.")
                    continue

                ckpt = torch.load(ckpt_path, map_location=device)
                best_epoch = ckpt.get("best_epoch", "N/A")
                best_rew = ckpt.get("best_reward", 0.0)

                # Instantiate agent
                if algo == "ppo_baseline":
                    agent = CategoricalPPOAgent(env=eval_env, device=device)
                elif algo == "diffusion_ppo":
                    agent = DiscreteDiffusionPPOAgent(env=eval_env, num_timesteps=diffusion_timesteps, device=device)
                elif algo == "rl_d2_rkl":
                    agent = RLD2RKLAgent(env=eval_env, num_timesteps=diffusion_timesteps, device=device)
                elif algo == "rl_d2_fkl":
                    agent = RLD2FKLAgent(env=eval_env, num_timesteps=diffusion_timesteps, device=device)
                else:
                    continue

                agent.actor.load_state_dict(ckpt["actor_state_dict"])
                if "critic_state_dict" in ckpt:
                    agent.critic.load_state_dict(ckpt["critic_state_dict"])

                # Standard stochastic evaluation on shared test set
                eval_metrics = evaluate_policy(
                    policy_type=algo,
                    env=eval_env,
                    agent=agent,
                    num_episodes=num_test_episodes,
                    seed=shared_eval_seed,
                    deterministic=False,
                    top_p=None,
                )
                eval_metrics["seed"] = seed
                all_eval.append(eval_metrics)
                print(f"Evaluated {algo:14s} | Users={n_users} | Seed={seed} (from Epoch {best_epoch}): Mean Cost = {eval_metrics['mean_cost']:.3f}")

                # Top-p sampling evaluation for diffusion policies
                if algo in ["diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]:
                    eval_topp = evaluate_policy(
                        policy_type=algo,
                        env=eval_env,
                        agent=agent,
                        num_episodes=num_test_episodes,
                        seed=shared_eval_seed,
                        deterministic=False,
                        top_p=0.98,
                    )
                    eval_topp["seed"] = seed
                    eval_topp["policy"] = f"{algo}_topp"
                    all_eval.append(eval_topp)

        # Generate comparison plot for this user scale
        if not df_all_history.empty:
            df_hist_curr = df_all_history[df_all_history["n_users"] == n_users]
        else:
            df_hist_curr = pd.DataFrame()
        df_eval_curr = pd.DataFrame([e for e in all_eval if e["n_users"] == n_users])
        plot_comparison_results(df_hist_curr, df_eval_curr, n_users=n_users)

    # Save complete standardized evaluation metrics
    df_all_eval = pd.DataFrame(all_eval)
    df_all_eval.to_csv("results/evaluation_metrics.csv", index=False)
    print("\nSaved updated evaluation metrics to results/evaluation_metrics.csv")

    # Generate multi-scale scalability figure
    plot_scalability_analysis(df_all_eval)

    # Print summary table
    print("\n==========================================================================================")
    print("STANDARDIZED MEC POLICY EVALUATION BENCHMARK SUMMARY (FAIR TEST SET)")
    print("==========================================================================================")
    print_evaluation_summary(all_eval)


if __name__ == "__main__":
    run_standardized_checkpoint_evaluation(
        user_counts=[2, 4, 8],
        seeds=[42, 43],
        num_test_episodes=10,
        shared_eval_seed=1000,
        diffusion_timesteps=4,
    )

