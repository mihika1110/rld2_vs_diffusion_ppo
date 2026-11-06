import os
import time
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import functools

from mec_env import MECEnv
from ppo_baseline import CategoricalPPOAgent
from ppo_diffusion import DiscreteDiffusionPPOAgent
from rl_d2_rkl import RLD2RKLAgent
from rl_d2_fkl import RLD2FKLAgent
from evaluate import evaluate_policy, print_evaluation_summary

print = functools.partial(print, flush=True)


def ensure_dirs():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def train_single_agent(
    algo_name: str,
    n_users: int = 4,
    n_stations: int = 2,
    seed: int = 42,
    total_epochs: int = 10,
    rollout_steps: int = 1024,
    diffusion_timesteps: int = 32,
    device: str = "cpu",
) -> Tuple[Any, List[Dict[str, Any]]]:

    np.random.seed(seed)
    torch.manual_seed(seed)

    #creates a mec environment 
    env = MECEnv(n_stations=n_stations, n_users=n_users, seed=seed)

    if algo_name == "ppo_baseline":
        agent = CategoricalPPOAgent(env=env, device=device)
    elif algo_name == "diffusion_ppo":
        agent = DiscreteDiffusionPPOAgent(
            env=env,
            num_timesteps=diffusion_timesteps,
            device=device,
        )
    elif algo_name == "rl_d2_rkl":
        agent = RLD2RKLAgent(
            env=env,
            num_timesteps=diffusion_timesteps,
            auto_tune_temp=True,
            device=device,
        )
    elif algo_name == "rl_d2_fkl":
        total_samples = total_epochs * rollout_steps
        fkl_buffer_size = max(64, int(0.05 * total_samples))  # 5% of total samples
        agent = RLD2FKLAgent(
            env=env,
            num_timesteps=diffusion_timesteps,
            replay_buffer_size=fkl_buffer_size,
            auto_tune_temp=True,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    history = []
    print(f"\n--- Training {algo_name} | Users={n_users} | Seed={seed} ---")
    start_time = time.time()

    best_reward = -float("inf")
    best_actor_state = None
    best_critic_state = None
    best_epoch = 1

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        metrics = agent.train_epoch(rollout_steps=rollout_steps)
        elapsed = time.time() - t0

        metrics["epoch"] = epoch
        metrics["total_steps"] = epoch*rollout_steps
        metrics["algo"] = algo_name
        metrics["n_users"] = n_users
        metrics["seed"] = seed
        metrics["epoch_time_s"] = elapsed
        history.append(metrics)

        # Track best performing epoch by mean_reward
        if metrics["mean_reward"] > best_reward:
            best_reward = metrics["mean_reward"]
            best_epoch = epoch
            best_actor_state = {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()}
            best_critic_state = {k: v.cpu().clone() for k, v in agent.critic.state_dict().items()}

        temp_str = f" | Temp: {metrics['temperature']:.3f}" if "temperature" in metrics else ""
        print(
            f"Epoch {epoch:2d}/{total_epochs} | "
            f"Steps: {epoch*rollout_steps:5d} | "
            f"Reward: {metrics['mean_reward']:.3f} | "
            f"Cost: {metrics['mean_cost']:.3f} | "
            f"Offload%: {metrics['mean_offload']*100:.1f}%{temp_str} | "
            f"Time: {elapsed:.2f}s"
        )

    total_time = time.time() - start_time
    print(f"Finished {algo_name} in {total_time:.1f}s")

    # Restore best checkpoint weights to agent for evaluation and saving
    if best_actor_state is not None:
        agent.actor.load_state_dict(best_actor_state)
        agent.critic.load_state_dict(best_critic_state)

    #save best checkpoint
    save_path = f"checkpoints/{algo_name}_u{n_users}_s{seed}.pt"
    checkpoint = {
        "algo": algo_name,
        "n_users": n_users,
        "n_stations": n_stations,
        "best_epoch": best_epoch,
        "best_reward": best_reward,
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Saved best weights (from Epoch {best_epoch}, Reward: {best_reward:.3f}) to {save_path}")

    return agent, history


def plot_comparison_results(df_history: pd.DataFrame, df_eval: pd.DataFrame, n_users: int):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # 1.training reward curves
    ax = axes[0]
    algos = df_history["algo"].unique()
    colors = {
        "ppo_baseline": "#1f77b4",
        "diffusion_ppo": "#2ca02c",
        "rl_d2_rkl": "#ff7f0e",
        "rl_d2_fkl": "#9467bd",
    }
    labels = {
        "ppo_baseline": "Independent Categorical PPO",
        "diffusion_ppo": "Discrete Diffusion PPO (Nelson Sir's Formulation)",
        "rl_d2_rkl": "RL-D² RKL (Ma et al. 2026)",
        "rl_d2_fkl": "RL-D² FKL (Ma et al. 2026)",
    }

    for algo in algos:
        sub = df_history[df_history["algo"] == algo]
        grouped = sub.groupby("total_steps")["mean_reward"].agg(["mean", "std"]).reset_index()
        mean = grouped["mean"].values
        std = grouped["std"].fillna(0).values
        steps = grouped["total_steps"].values

        ax.plot(steps, mean, label=labels.get(algo, algo), color=colors.get(algo, None), lw=2.2)
        ax.fill_between(steps, mean - std, mean + std, color=colors.get(algo, None), alpha=0.18)

    ax.set_title(f"Learning Curves (M={n_users} Users, K={3**n_users} Actions)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Reward (Negative Cost)", fontsize=11)
    ax.legend(frameon=True, loc="lower right", fontsize=9)

    # 2.evaluation mean cost comparison
    ax = axes[1]
    policy_order = ["random", "greedy", "ppo_baseline", "diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]
    eval_sub = df_eval[df_eval["policy"].isin(policy_order)].copy()
    grouped_eval = eval_sub.groupby("policy")["mean_cost"].agg(["mean", "std"]).reindex(policy_order)
    grouped_eval["std"] = grouped_eval["std"].fillna(0.0)
    grouped_eval = grouped_eval.dropna(subset=["mean"])

    bar_label_map = {
        "random": "Random",
        "greedy": "Greedy",
        "ppo_baseline": "Categorical\nPPO",
        "diffusion_ppo": "Discrete\nDiffusion PPO",
        "rl_d2_rkl": "RL-D²\n(RKL)",
        "rl_d2_fkl": "RL-D²\n(FKL)",
    }
    bar_color_map = {
        "random": "#7f7f7f",
        "greedy": "#bcbd22",
        "ppo_baseline": "#1f77b4",
        "diffusion_ppo": "#2ca02c",
        "rl_d2_rkl": "#ff7f0e",
        "rl_d2_fkl": "#9467bd",
    }

    current_policies = list(grouped_eval.index)
    current_labels = [bar_label_map.get(p, p) for p in current_policies]
    current_colors = [bar_color_map.get(p, "#333333") for p in current_policies]
    x_pos = np.arange(len(grouped_eval))

    bars = ax.bar(
        x_pos,
        grouped_eval["mean"].values,
        yerr=grouped_eval["std"].values,
        capsize=5,
        color=current_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(current_labels, fontsize=9)
    ax.set_title(f"Evaluation Cost Comparison (Lower is Better)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted Cost (Latency + Energy)", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 3.latency vs energy trade-off
    ax = axes[2]
    grouped_latency = (
        eval_sub.groupby("policy")[["mean_latency_s", "mean_energy_j"]]
        .mean()
        .reindex(policy_order)
        .dropna(subset=["mean_latency_s", "mean_energy_j"])
    )
    for pol in grouped_latency.index:
        lat = grouped_latency.loc[pol, "mean_latency_s"]
        en = grouped_latency.loc[pol, "mean_energy_j"]
        color = bar_color_map.get(pol, "#333333")
        label = bar_label_map.get(pol, pol).replace("\n", " ")
        ax.scatter(lat, en, color=color, s=140, edgecolors="black", zorder=4)
        ax.annotate(
            label,
            xy=(lat, en),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Latency vs Energy Trade-off", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Execution Latency (s)", fontsize=11)
    ax.set_ylabel("Mean Energy Consumption (J)", fontsize=11)

    plt.tight_layout()
    plot_file = f"results/comparison_users_{n_users}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {plot_file}")


def plot_scalability_analysis(df_eval: pd.DataFrame):
    unique_users = sorted(df_eval["n_users"].unique())
    if len(unique_users) < 2:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    policies = ["random", "greedy", "ppo_baseline", "diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]
    colors = {
        "random": "#7f7f7f",
        "greedy": "#bcbd22",
        "ppo_baseline": "#1f77b4",
        "diffusion_ppo": "#2ca02c",
        "rl_d2_rkl": "#ff7f0e",
        "rl_d2_fkl": "#9467bd",
    }
    labels = {
        "random": "Random",
        "greedy": "Greedy",
        "ppo_baseline": "Independent PPO",
        "diffusion_ppo": "Discrete Diffusion PPO",
        "rl_d2_rkl": "RL-D² (RKL)",
        "rl_d2_fkl": "RL-D² (FKL)",
    }

    # 1.cost vs number of users
    ax = axes[0]
    for pol in policies:
        sub = df_eval[df_eval["policy"] == pol]
        if sub.empty:
            continue
        grouped = sub.groupby("n_users")["mean_cost"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", lw=2.2, label=labels.get(pol, pol), color=colors.get(pol))
    ax.set_title("Scalability: Cost vs Number of Users", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Users (M)", fontsize=11)
    ax.set_ylabel("Mean Weighted Cost", fontsize=11)
    ax.set_xticks(unique_users)
    ax.legend(frameon=True, fontsize=9)

    # 2.decision time vs users
    ax = axes[1]
    for pol in policies:
        sub = df_eval[df_eval["policy"] == pol]
        if sub.empty:
            continue
        grouped = sub.groupby("n_users")["decision_time_ms"].mean()
        ax.plot(grouped.index, grouped.values, marker="s", lw=2.2, label=labels.get(pol, pol), color=colors.get(pol))
    ax.set_title("Inference Time vs Number of Users", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Users (M)", fontsize=11)
    ax.set_ylabel("Decision Latency per Step (ms)", fontsize=11)
    ax.set_xticks(unique_users)
    ax.legend(frameon=True, fontsize=9)

    # 3.offloading percentage vs users
    ax = axes[2]
    for pol in policies:
        sub = df_eval[df_eval["policy"] == pol]
        if sub.empty:
            continue
        grouped = sub.groupby("n_users")["mean_offload_pct"].mean()
        ax.plot(grouped.index, grouped.values, marker="^", lw=2.2, label=labels.get(pol, pol), color=colors.get(pol))
    ax.set_title("Offloading Fraction vs Number of Users", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Users (M)", fontsize=11)
    ax.set_ylabel("Offloading Percentage (%)", fontsize=11)
    ax.set_xticks(unique_users)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    scale_file = "results/scalability_analysis.png"
    plt.savefig(scale_file, dpi=300)
    plt.close()
    print(f"Saved scalability analysis figure to {scale_file}")


def run_experiments(
    user_counts: List[int] = [2, 4, 8],
    seeds: List[int] = [42, 43],
    total_epochs: int = 10,
    rollout_steps: int = 1024,
    diffusion_timesteps: int = 4,
):

    ensure_dirs()
    all_history = []
    all_eval = []

    for n_users in user_counts:
        print(f"\n========================================================")
        print(f"STARTING COMPREHENSIVE EXPERIMENTS FOR M = {n_users} USERS")
        print(f"Action Space Size: (2+1)^{n_users} = {3**n_users} Joint Actions")
        print(f"========================================================")

        eval_env = MECEnv(n_stations=2, n_users=n_users, seed=999)

        # Standardized evaluation parameters across all policies
        SHARED_EVAL_SEED = 1000
        EVAL_EPISODES = 10

        # 1. Evaluate baseline non-learning policies on the shared test set
        for seed in seeds:
            rand_res = evaluate_policy("random", eval_env, num_episodes=EVAL_EPISODES, seed=SHARED_EVAL_SEED)
            rand_res["seed"] = seed
            greedy_res = evaluate_policy("greedy", eval_env, num_episodes=EVAL_EPISODES, seed=SHARED_EVAL_SEED)
            greedy_res["seed"] = seed
            all_eval.extend([rand_res, greedy_res])

        # 2. Train learning algorithms across seeds
        algos = ["ppo_baseline", "diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]

        for seed in seeds:
            for algo in algos:
                agent, history = train_single_agent(
                    algo_name=algo,
                    n_users=n_users,
                    seed=seed,
                    total_epochs=total_epochs,
                    rollout_steps=rollout_steps,
                    diffusion_timesteps=diffusion_timesteps,
                )
                all_history.extend(history)

                # Standard policy evaluation on the identical shared test episodes
                eval_metrics = evaluate_policy(
                    policy_type=algo,
                    env=eval_env,
                    agent=agent,
                    num_episodes=EVAL_EPISODES,
                    seed=SHARED_EVAL_SEED,
                    deterministic=False,
                    top_p=None,
                )
                eval_metrics["seed"] = seed
                all_eval.append(eval_metrics)

                # Top-p sampling evaluation for diffusion policies on the identical shared test episodes
                if algo in ["diffusion_ppo", "rl_d2_rkl", "rl_d2_fkl"]:
                    eval_metrics_topp = evaluate_policy(
                        policy_type=algo,
                        env=eval_env,
                        agent=agent,
                        num_episodes=EVAL_EPISODES,
                        seed=SHARED_EVAL_SEED,
                        deterministic=False,
                        top_p=0.98,
                    )
                    eval_metrics_topp["seed"] = seed
                    eval_metrics_topp["policy"] = f"{algo}_topp"
                    all_eval.append(eval_metrics_topp)

        #plot for this scale
        df_hist_curr = pd.DataFrame([h for h in all_history if h["n_users"] == n_users])
        df_eval_curr = pd.DataFrame([e for e in all_eval if e["n_users"] == n_users])
        plot_comparison_results(df_hist_curr, df_eval_curr, n_users=n_users)

    #save complete csv results
    df_all_history = pd.DataFrame(all_history)
    df_all_eval = pd.DataFrame(all_eval)
    df_all_history.to_csv("results/training_history.csv", index=False)
    df_all_eval.to_csv("results/evaluation_metrics.csv", index=False)

    #multi-scale analysis plot if applicable
    plot_scalability_analysis(df_all_eval)

    print("\nSaved full logs to results/training_history.csv and results/evaluation_metrics.csv")
    print_evaluation_summary(all_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEC RL Diffusion Comprehensive Runner")
    parser.add_argument("--users", type=int, nargs="+", default=[2, 4, 8], help="List of user counts, e.g. 2 4 8")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per seed")
    parser.add_argument("--rollout_steps", type=int, default=1024, help="Rollout steps per epoch")
    parser.add_argument("--diff_steps", type=int, default=4, help="Diffusion steps T")
    args = parser.parse_args()

    run_experiments(
        user_counts=args.users,
        seeds=args.seeds,
        total_epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        diffusion_timesteps=args.diff_steps,
    )
