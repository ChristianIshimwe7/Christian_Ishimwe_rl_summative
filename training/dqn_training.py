"""
DQN Training Script – Medical Triage Environment
=================================================
Trains a Deep Q-Network agent using Stable Baselines 3.
Runs 10 hyperparameter experiments and saves results + plots.

Author : Christian Ishimwe
Course : Machine Learning & Robotics - ALU

Usage:
    python training/dqn_training.py               # run all 10 experiments
    python training/dqn_training.py --run 0       # run single experiment index
    python training/dqn_training.py --best        # train best config longer
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MedicalTriageEnv

RESULTS_DIR = "results/dqn"
MODEL_DIR   = "models/dqn"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 100_000   # per experiment (increase to 500k for final)

# ─────────────────────────────────────────────
#  10 Hyperparameter Configurations
# ─────────────────────────────────────────────
DQN_EXPERIMENTS = [
    # Run | lr      | gamma | bs   | buf    | eps_s | eps_e  | tau   | net arch
    {"run": 1,  "learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.2,  "exploration_final_eps": 0.05, "tau": 1.0,  "policy_kwargs": {"net_arch": [128, 128]}},
    {"run": 2,  "learning_rate": 5e-4,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.3,  "exploration_final_eps": 0.05, "tau": 1.0,  "policy_kwargs": {"net_arch": [128, 128]}},
    {"run": 3,  "learning_rate": 1e-4,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 100_000, "exploration_fraction": 0.3,  "exploration_final_eps": 0.05, "tau": 1.0,  "policy_kwargs": {"net_arch": [256, 256]}},
    {"run": 4,  "learning_rate": 1e-3,  "gamma": 0.95, "batch_size": 128, "buffer_size": 50_000,  "exploration_fraction": 0.2,  "exploration_final_eps": 0.10, "tau": 0.5,  "policy_kwargs": {"net_arch": [128, 128]}},
    {"run": 5,  "learning_rate": 5e-4,  "gamma": 0.95, "batch_size": 128, "buffer_size": 50_000,  "exploration_fraction": 0.4,  "exploration_final_eps": 0.05, "tau": 0.5,  "policy_kwargs": {"net_arch": [256, 128]}},
    {"run": 6,  "learning_rate": 1e-3,  "gamma": 0.99, "batch_size": 32,  "buffer_size": 100_000, "exploration_fraction": 0.2,  "exploration_final_eps": 0.01, "tau": 1.0,  "policy_kwargs": {"net_arch": [64,  64]}},
    {"run": 7,  "learning_rate": 2e-4,  "gamma": 0.99, "batch_size": 64,  "buffer_size": 200_000, "exploration_fraction": 0.3,  "exploration_final_eps": 0.05, "tau": 0.01, "policy_kwargs": {"net_arch": [256, 256]}},
    {"run": 8,  "learning_rate": 1e-3,  "gamma": 0.90, "batch_size": 64,  "buffer_size": 50_000,  "exploration_fraction": 0.5,  "exploration_final_eps": 0.10, "tau": 1.0,  "policy_kwargs": {"net_arch": [128, 64]}},
    {"run": 9,  "learning_rate": 5e-4,  "gamma": 0.99, "batch_size": 256, "buffer_size": 100_000, "exploration_fraction": 0.1,  "exploration_final_eps": 0.02, "tau": 0.5,  "policy_kwargs": {"net_arch": [512, 256]}},
    {"run": 10, "learning_rate": 1e-4,  "gamma": 0.95, "batch_size": 128, "buffer_size": 200_000, "exploration_fraction": 0.25, "exploration_final_eps": 0.05, "tau": 0.01, "policy_kwargs": {"net_arch": [256, 256]}},
]


# ─────────────────────────────────────────────
#  Reward Logging Callback
# ─────────────────────────────────────────────
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self.num_timesteps)
            self._current_reward = 0.0
        return True


# ─────────────────────────────────────────────
#  Train one DQN experiment
# ─────────────────────────────────────────────
def train_dqn(config: dict, timesteps: int = TOTAL_TIMESTEPS) -> dict:
    run_id = config["run"]
    print(f"\n{'='*55}")
    print(f"  DQN Experiment {run_id:02d} / 10")
    print(f"  lr={config['learning_rate']}  gamma={config['gamma']}  "
          f"bs={config['batch_size']}  buf={config['buffer_size']}")
    print(f"{'='*55}")

    env    = Monitor(MedicalTriageEnv())
    eval_e = Monitor(MedicalTriageEnv())

    callback  = RewardLoggerCallback()
    eval_cb   = EvalCallback(
        eval_e,
        best_model_save_path=os.path.join(MODEL_DIR, f"run_{run_id:02d}"),
        log_path=os.path.join(RESULTS_DIR, f"run_{run_id:02d}"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate      = config["learning_rate"],
        gamma              = config["gamma"],
        batch_size         = config["batch_size"],
        buffer_size        = config["buffer_size"],
        exploration_fraction    = config["exploration_fraction"],
        exploration_final_eps   = config["exploration_final_eps"],
        tau                = config["tau"],
        policy_kwargs      = config["policy_kwargs"],
        learning_starts    = 1000,
        train_freq         = 4,
        target_update_interval = 1000,
        verbose            = 1,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[callback, eval_cb],
        progress_bar=True,
    )

    # Save final model
    save_path = os.path.join(MODEL_DIR, f"dqn_run_{run_id:02d}_final")
    model.save(save_path)

    # Evaluate
    mean_r, std_r = _evaluate(model, n_episodes=20)
    result = {
        "run":             run_id,
        "learning_rate":   config["learning_rate"],
        "gamma":           config["gamma"],
        "batch_size":      config["batch_size"],
        "buffer_size":     config["buffer_size"],
        "exploration_fraction": config["exploration_fraction"],
        "exploration_final_eps": config["exploration_final_eps"],
        "tau":             config["tau"],
        "net_arch":        str(config["policy_kwargs"]["net_arch"]),
        "mean_reward":     round(mean_r, 2),
        "std_reward":      round(std_r,  2),
        "episode_rewards": callback.episode_rewards,
    }

    env.close()
    eval_e.close()
    return result


def _evaluate(model, n_episodes=20) -> tuple:
    env = MedicalTriageEnv()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done  = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ─────────────────────────────────────────────
#  Plot results
# ─────────────────────────────────────────────
def plot_dqn_results(all_results: list):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Hyperparameter Experiments – Medical Triage", fontsize=14)

    # 1. Mean reward per run
    ax = axes[0, 0]
    runs  = [r["run"] for r in all_results]
    means = [r["mean_reward"] for r in all_results]
    stds  = [r["std_reward"]  for r in all_results]
    bars  = ax.bar(runs, means, yerr=stds, capsize=4,
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(runs))))
    ax.set_xlabel("Run #")
    ax.set_ylabel("Mean Reward (20 episodes)")
    ax.set_title("Mean Final Reward per Run")
    ax.set_xticks(runs)

    # 2. Cumulative reward curves (all runs)
    ax = axes[0, 1]
    for r in all_results:
        rw = r["episode_rewards"]
        if rw:
            cum = np.cumsum(rw)
            ax.plot(cum, label=f"Run {r['run']}", alpha=0.7, linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Curves")
    ax.legend(fontsize=6, ncol=2)

    # 3. Learning rate vs mean reward
    ax = axes[1, 0]
    lrs = sorted(set(r["learning_rate"] for r in all_results))
    for lr in lrs:
        subset = [r for r in all_results if r["learning_rate"] == lr]
        ax.scatter([lr] * len(subset), [r["mean_reward"] for r in subset],
                   s=80, label=f"lr={lr}")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Mean Reward")
    ax.set_title("LR vs Mean Reward")
    ax.set_xscale("log")
    ax.legend(fontsize=7)

    # 4. Gamma vs mean reward
    ax = axes[1, 1]
    gammas = sorted(set(r["gamma"] for r in all_results))
    for g in gammas:
        subset = [r for r in all_results if r["gamma"] == g]
        ax.scatter([g] * len(subset), [r["mean_reward"] for r in subset],
                   s=80, label=f"γ={g}")
    ax.set_xlabel("Gamma (discount factor)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Gamma vs Mean Reward")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dqn_hyperparameter_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n[DQN] Plot saved → {out}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",  type=int, default=-1,
                        help="Run single experiment index (0-9). -1 = all.")
    parser.add_argument("--best", action="store_true",
                        help="Train best config for 500k steps.")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    if args.best:
        best_config = DQN_EXPERIMENTS[2]  # Run 3 tends to converge well
        print("[DQN] Training best config for 500k steps...")
        result = train_dqn(best_config, timesteps=500_000)
        print(f"[DQN BEST] Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        return

    experiments = [DQN_EXPERIMENTS[args.run]] if args.run >= 0 else DQN_EXPERIMENTS

    all_results = []
    for cfg in experiments:
        result = train_dqn(cfg, timesteps=args.timesteps)
        all_results.append(result)
        # Save intermediate JSON
        with open(os.path.join(RESULTS_DIR, "dqn_results.json"), "w") as f:
            # Strip episode_rewards for readability
            to_save = [{k: v for k, v in r.items() if k != "episode_rewards"}
                       for r in all_results]
            json.dump(to_save, f, indent=2)

    if len(all_results) > 1:
        plot_dqn_results(all_results)

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Run':>4} {'LR':>8} {'Gamma':>6} {'Batch':>6} {'Buffer':>8} "
          f"{'Mean R':>8} {'Std R':>7}")
    print("-"*70)
    for r in all_results:
        print(f"{r['run']:>4} {r['learning_rate']:>8.0e} {r['gamma']:>6.2f} "
              f"{r['batch_size']:>6} {r['buffer_size']:>8} "
              f"{r['mean_reward']:>8.2f} {r['std_reward']:>7.2f}")
    print("="*70)

    best = max(all_results, key=lambda x: x["mean_reward"])
    print(f"\n[DQN] Best run: #{best['run']} with mean reward = {best['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
