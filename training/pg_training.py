"""
Policy Gradient Training Script – Medical Triage Environment
=============================================================
Trains three policy-gradient agents using Stable Baselines 3:
  • REINFORCE  (custom implementation wrapping SB3 A2C with no critic)
  • PPO        (Proximal Policy Optimization)
  • A2C        (Advantage Actor-Critic)

Each algorithm runs 10 hyperparameter experiments.

Author : Christian Ishimwe
Course : Machine Learning & Robotics - ALU

Usage:
    python training/pg_training.py                    # all algorithms, all runs
    python training/pg_training.py --algo ppo         # PPO only
    python training/pg_training.py --algo a2c --run 3 # A2C, run index 3
    python training/pg_training.py --best ppo         # train best PPO config
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import MedicalTriageEnv

RESULTS_DIR_PPO      = "results/ppo"
RESULTS_DIR_A2C      = "results/a2c"
RESULTS_DIR_REINFORCE = "results/reinforce"
MODEL_DIR_PG         = "models/pg"

for d in [RESULTS_DIR_PPO, RESULTS_DIR_A2C, RESULTS_DIR_REINFORCE, MODEL_DIR_PG]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 100_000


# ─────────────────────────────────────────────
#  Hyperparameter grids
# ─────────────────────────────────────────────

PPO_EXPERIMENTS = [
    {"run":1,  "learning_rate":3e-4, "gamma":0.99, "n_steps":2048, "batch_size":64,  "n_epochs":10, "clip_range":0.2,  "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.95, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":2,  "learning_rate":1e-4, "gamma":0.99, "n_steps":2048, "batch_size":64,  "n_epochs":10, "clip_range":0.2,  "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.95, "policy_kwargs":{"net_arch":[256,256]}},
    {"run":3,  "learning_rate":3e-4, "gamma":0.95, "n_steps":1024, "batch_size":128, "n_epochs":5,  "clip_range":0.2,  "ent_coef":0.001, "vf_coef":0.5,  "gae_lambda":0.90, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":4,  "learning_rate":5e-4, "gamma":0.99, "n_steps":512,  "batch_size":64,  "n_epochs":10, "clip_range":0.3,  "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.95, "policy_kwargs":{"net_arch":[64, 64]}},
    {"run":5,  "learning_rate":3e-4, "gamma":0.99, "n_steps":2048, "batch_size":256, "n_epochs":10, "clip_range":0.2,  "ent_coef":0.05,  "vf_coef":0.25, "gae_lambda":0.95, "policy_kwargs":{"net_arch":[256,128]}},
    {"run":6,  "learning_rate":1e-3, "gamma":0.99, "n_steps":2048, "batch_size":64,  "n_epochs":20, "clip_range":0.1,  "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.98, "policy_kwargs":{"net_arch":[256,256]}},
    {"run":7,  "learning_rate":2e-4, "gamma":0.99, "n_steps":4096, "batch_size":128, "n_epochs":10, "clip_range":0.2,  "ent_coef":0.0,   "vf_coef":0.5,  "gae_lambda":0.95, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":8,  "learning_rate":3e-4, "gamma":0.90, "n_steps":2048, "batch_size":64,  "n_epochs":10, "clip_range":0.2,  "ent_coef":0.02,  "vf_coef":0.75, "gae_lambda":0.95, "policy_kwargs":{"net_arch":[512,256]}},
    {"run":9,  "learning_rate":1e-4, "gamma":0.99, "n_steps":1024, "batch_size":32,  "n_epochs":5,  "clip_range":0.25, "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.92, "policy_kwargs":{"net_arch":[128, 64]}},
    {"run":10, "learning_rate":3e-4, "gamma":0.99, "n_steps":2048, "batch_size":64,  "n_epochs":10, "clip_range":0.2,  "ent_coef":0.01,  "vf_coef":0.5,  "gae_lambda":0.95, "policy_kwargs":{"net_arch":[256,256,128]}},
]

A2C_EXPERIMENTS = [
    {"run":1,  "learning_rate":7e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.0,   "vf_coef":0.5, "gae_lambda":1.0,  "policy_kwargs":{"net_arch":[128,128]}},
    {"run":2,  "learning_rate":1e-3, "gamma":0.99, "n_steps":5,   "ent_coef":0.01,  "vf_coef":0.5, "gae_lambda":1.0,  "policy_kwargs":{"net_arch":[256,256]}},
    {"run":3,  "learning_rate":5e-4, "gamma":0.99, "n_steps":10,  "ent_coef":0.01,  "vf_coef":0.25,"gae_lambda":0.95, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":4,  "learning_rate":7e-4, "gamma":0.95, "n_steps":20,  "ent_coef":0.001, "vf_coef":0.5, "gae_lambda":1.0,  "policy_kwargs":{"net_arch":[64, 64]}},
    {"run":5,  "learning_rate":2e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.05,  "vf_coef":0.5, "gae_lambda":0.90, "policy_kwargs":{"net_arch":[256,128]}},
    {"run":6,  "learning_rate":7e-4, "gamma":0.99, "n_steps":50,  "ent_coef":0.01,  "vf_coef":0.75,"gae_lambda":1.0,  "policy_kwargs":{"net_arch":[256,256]}},
    {"run":7,  "learning_rate":1e-3, "gamma":0.90, "n_steps":5,   "ent_coef":0.0,   "vf_coef":0.5, "gae_lambda":1.0,  "policy_kwargs":{"net_arch":[128, 64]}},
    {"run":8,  "learning_rate":3e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.01,  "vf_coef":0.5, "gae_lambda":0.98, "policy_kwargs":{"net_arch":[512,256]}},
    {"run":9,  "learning_rate":7e-4, "gamma":0.99, "n_steps":128, "ent_coef":0.02,  "vf_coef":0.5, "gae_lambda":0.95, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":10, "learning_rate":5e-4, "gamma":0.95, "n_steps":20,  "ent_coef":0.01,  "vf_coef":0.5, "gae_lambda":0.95, "policy_kwargs":{"net_arch":[256,256]}},
]

# REINFORCE uses A2C with n_steps=episode_length, no bootstrapping (gamma controls)
REINFORCE_EXPERIMENTS = [
    {"run":1,  "learning_rate":1e-3, "gamma":0.99, "n_steps":200, "ent_coef":0.0,  "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":2,  "learning_rate":5e-4, "gamma":0.99, "n_steps":200, "ent_coef":0.0,  "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[256,256]}},
    {"run":3,  "learning_rate":1e-3, "gamma":0.95, "n_steps":200, "ent_coef":0.01, "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":4,  "learning_rate":2e-4, "gamma":0.99, "n_steps":100, "ent_coef":0.0,  "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[64, 64]}},
    {"run":5,  "learning_rate":1e-3, "gamma":0.99, "n_steps":200, "ent_coef":0.05, "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[256,128]}},
    {"run":6,  "learning_rate":3e-4, "gamma":0.90, "n_steps":200, "ent_coef":0.01, "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[128,128]}},
    {"run":7,  "learning_rate":1e-3, "gamma":0.99, "n_steps":200, "ent_coef":0.0,  "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[512,256]}},
    {"run":8,  "learning_rate":5e-4, "gamma":0.99, "n_steps":50,  "ent_coef":0.02, "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[128, 64]}},
    {"run":9,  "learning_rate":2e-3, "gamma":0.99, "n_steps":200, "ent_coef":0.0,  "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[256,256]}},
    {"run":10, "learning_rate":1e-4, "gamma":0.95, "n_steps":200, "ent_coef":0.01, "vf_coef":0.0, "gae_lambda":1.0, "policy_kwargs":{"net_arch":[128,128]}},
]


# ─────────────────────────────────────────────
#  Callback
# ─────────────────────────────────────────────
class EntropyRewardCallback(BaseCallback):
    """Logs per-episode reward and policy entropy."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards:  list = []
        self.entropy_log:      list = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += float(np.mean(self.locals.get("rewards", [0])))
        if any(self.locals.get("dones", [False])):
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0

        # Log entropy from rollout buffer when available
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "action_dist"):
            try:
                dist = self.model.policy.action_dist
                if dist is not None:
                    ent = dist.entropy().mean().item()
                    self.entropy_log.append(ent)
            except Exception:
                pass
        return True


# ─────────────────────────────────────────────
#  Generic SB3 trainer
# ─────────────────────────────────────────────
def train_sb3(algo_cls, config: dict, results_dir: str,
              model_dir: str, algo_name: str,
              timesteps: int = TOTAL_TIMESTEPS) -> dict:
    run_id = config["run"]
    print(f"\n{'='*60}")
    print(f"  {algo_name} Experiment {run_id:02d}/10")
    print(f"  lr={config['learning_rate']}  gamma={config['gamma']}  "
          f"n_steps={config.get('n_steps','N/A')}")
    print(f"{'='*60}")

    env    = Monitor(MedicalTriageEnv())
    eval_e = Monitor(MedicalTriageEnv())

    cb    = EntropyRewardCallback()
    eval_cb = EvalCallback(
        eval_e,
        best_model_save_path=os.path.join(model_dir, f"{algo_name.lower()}_run{run_id:02d}"),
        log_path=os.path.join(results_dir, f"run_{run_id:02d}"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Build kwargs – strip 'run' key
    kwargs = {k: v for k, v in config.items() if k != "run"}

    model = algo_cls("MlpPolicy", env, **kwargs, verbose=1)
    model.learn(
        total_timesteps=timesteps,
        callback=[cb, eval_cb],
        progress_bar=True,
    )

    save_path = os.path.join(model_dir, f"{algo_name.lower()}_run{run_id:02d}_final")
    model.save(save_path)

    mean_r, std_r = _evaluate_model(model)
    result = {
        "run":           run_id,
        "algo":          algo_name,
        "learning_rate": config["learning_rate"],
        "gamma":         config["gamma"],
        "n_steps":       config.get("n_steps", "N/A"),
        "ent_coef":      config.get("ent_coef", "N/A"),
        "vf_coef":       config.get("vf_coef",  "N/A"),
        "gae_lambda":    config.get("gae_lambda","N/A"),
        "net_arch":      str(config["policy_kwargs"]["net_arch"]),
        "mean_reward":   round(mean_r, 2),
        "std_reward":    round(std_r,  2),
        "episode_rewards": cb.episode_rewards,
        "entropy_log":     cb.entropy_log,
    }

    env.close()
    eval_e.close()
    return result


def _evaluate_model(model, n_episodes=20) -> tuple:
    env = MedicalTriageEnv()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            total += r
            done = term or trunc
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────
def plot_pg_results(results_by_algo: dict):
    """Side-by-side subplot for each algorithm."""

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Policy Gradient Experiments – Medical Triage", fontsize=14)

    algo_list = ["reinforce", "ppo", "a2c"]
    algo_titles = {"reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}

    for row_idx, algo in enumerate(algo_list):
        results = results_by_algo.get(algo, [])
        if not results:
            continue

        # Col 0: Mean reward per run
        ax = axes[row_idx, 0]
        runs  = [r["run"] for r in results]
        means = [r["mean_reward"] for r in results]
        stds  = [r["std_reward"]  for r in results]
        ax.bar(runs, means, yerr=stds, capsize=3,
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(runs))))
        ax.set_title(f"{algo_titles[algo]} – Mean Reward per Run")
        ax.set_xlabel("Run #")
        ax.set_ylabel("Mean Reward")
        ax.set_xticks(runs)

        # Col 1: Cumulative reward curves
        ax = axes[row_idx, 1]
        for r in results:
            rw = r["episode_rewards"]
            if rw:
                ax.plot(np.cumsum(rw), label=f"Run {r['run']}", alpha=0.7, linewidth=1)
        ax.set_title(f"{algo_titles[algo]} – Cumulative Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(fontsize=5, ncol=2)

        # Col 2: Entropy curves
        ax = axes[row_idx, 2]
        for r in results:
            ent = r["entropy_log"]
            if ent:
                # Smooth
                window = max(1, len(ent) // 50)
                smoothed = np.convolve(ent, np.ones(window)/window, mode="valid")
                ax.plot(smoothed, label=f"Run {r['run']}", alpha=0.7, linewidth=1)
        ax.set_title(f"{algo_titles[algo]} – Policy Entropy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.legend(fontsize=5, ncol=2)

    plt.tight_layout()
    out = "results/pg_hyperparameter_analysis.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n[PG] Plot saved → {out}")


def plot_algorithm_comparison(results_by_algo: dict):
    """Compare all 4 algorithms (including DQN if results exist)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Algorithm Comparison – Medical Triage RL", fontsize=14)

    algo_colors = {"dqn": "#e74c3c", "reinforce": "#3498db",
                   "ppo": "#2ecc71", "a2c": "#f39c12"}
    algo_labels = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}

    # Best run per algorithm
    ax = axes[0]
    for algo, results in results_by_algo.items():
        if results:
            best = max(results, key=lambda x: x["mean_reward"])
            ax.bar(algo_labels.get(algo, algo), best["mean_reward"],
                   yerr=best["std_reward"], capsize=5,
                   color=algo_colors.get(algo, "gray"))
    ax.set_title("Best Run Mean Reward per Algorithm")
    ax.set_ylabel("Mean Reward (20 eval episodes)")
    ax.set_xlabel("Algorithm")

    # Reward learning curves for best run per algo
    ax = axes[1]
    for algo, results in results_by_algo.items():
        if results:
            best = max(results, key=lambda x: x["mean_reward"])
            rw = best["episode_rewards"]
            if rw:
                # Rolling mean
                window = max(1, len(rw) // 20)
                smoothed = np.convolve(rw, np.ones(window)/window, mode="valid")
                ax.plot(smoothed, label=algo_labels.get(algo, algo),
                        color=algo_colors.get(algo, "gray"), linewidth=2)
    ax.set_title("Episode Reward (Best Run, Smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()

    plt.tight_layout()
    out = "results/algorithm_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[COMPARISON] Plot saved → {out}")


def print_table(results: list, algo: str):
    print(f"\n{'='*80}")
    print(f"  {algo} Results Summary")
    print(f"{'='*80}")
    print(f"{'Run':>4} {'LR':>8} {'Gamma':>6} {'NSteps':>8} "
          f"{'EntCoef':>8} {'NetArch':>15} {'MeanR':>8} {'StdR':>7}")
    print("-"*80)
    for r in results:
        print(f"{r['run']:>4} {r['learning_rate']:>8.0e} {r['gamma']:>6.2f} "
              f"{str(r['n_steps']):>8} {str(r.get('ent_coef','N/A')):>8} "
              f"{r['net_arch']:>15} {r['mean_reward']:>8.2f} {r['std_reward']:>7.2f}")
    print("="*80)
    best = max(results, key=lambda x: x["mean_reward"])
    print(f"Best: Run #{best['run']}  Mean Reward = {best['mean_reward']:.2f}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      type=str, default="all",
                        choices=["all", "ppo", "a2c", "reinforce"],
                        help="Algorithm to train.")
    parser.add_argument("--run",       type=int, default=-1,
                        help="Single run index (0-9). -1 = all.")
    parser.add_argument("--best",      type=str, default="",
                        help="Train best config for this algo (500k steps).")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    all_results: dict = {}

    def run_algo(algo_name: str, algo_cls, experiments: list, results_dir: str):
        exps = [experiments[args.run]] if args.run >= 0 else experiments
        results = []
        for cfg in exps:
            r = train_sb3(algo_cls, cfg, results_dir, MODEL_DIR_PG,
                          algo_name.upper(), timesteps=args.timesteps)
            results.append(r)
            save_json(results, os.path.join(results_dir, f"{algo_name}_results.json"))
        print_table(results, algo_name.upper())
        return results

    if args.best:
        bmap = {"ppo": (PPO, PPO_EXPERIMENTS, RESULTS_DIR_PPO),
                "a2c": (A2C, A2C_EXPERIMENTS, RESULTS_DIR_A2C),
                "reinforce": (A2C, REINFORCE_EXPERIMENTS, RESULTS_DIR_REINFORCE)}
        algo_cls, exps, rdir = bmap[args.best]
        best_cfg = max(exps, key=lambda x: x["run"])  # placeholder – replace with best run idx
        print(f"Training best {args.best.upper()} config for 500k steps...")
        r = train_sb3(algo_cls, best_cfg, rdir, MODEL_DIR_PG,
                      args.best.upper(), timesteps=500_000)
        print(f"Best {args.best.upper()} Mean Reward: {r['mean_reward']:.2f}")
        return

    if args.algo in ("all", "reinforce"):
        all_results["reinforce"] = run_algo(
            "reinforce", A2C, REINFORCE_EXPERIMENTS, RESULTS_DIR_REINFORCE)

    if args.algo in ("all", "a2c"):
        all_results["a2c"] = run_algo(
            "a2c", A2C, A2C_EXPERIMENTS, RESULTS_DIR_A2C)

    if args.algo in ("all", "ppo"):
        all_results["ppo"] = run_algo(
            "ppo", PPO, PPO_EXPERIMENTS, RESULTS_DIR_PPO)

    # Merge DQN results if they exist
    dqn_path = "results/dqn/dqn_results.json"
    if os.path.exists(dqn_path):
        with open(dqn_path) as f:
            dqn_r = json.load(f)
            all_results["dqn"] = [
                {**r, "episode_rewards": [], "entropy_log": []} for r in dqn_r
            ]

    if len(all_results) > 0:
        plot_pg_results(all_results)
        plot_algorithm_comparison(all_results)


def save_json(data: list, path: str):
    to_save = [{k: v for k, v in r.items()
                if k not in ("episode_rewards", "entropy_log")} for r in data]
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)


if __name__ == "__main__":
    main()
