"""
main.py – Medical Triage RL Agent Demo
=======================================
Entry point for running the best-performing trained agent in the
full Panda3D 3D hospital ward simulation.

Author : Christian Ishimwe
Course : Machine Learning & Robotics - ALU

Usage:
    python main.py                          # auto-select best model
    python main.py --model models/dqn/...  # specify model path
    python main.py --algo ppo              # choose algorithm family
    python main.py --random                # random agent baseline
    python main.py --episodes 5            # number of episodes
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────
def load_best_model(algo: str = "auto"):
    """Load best saved model from models/ directory."""
    from stable_baselines3 import DQN, PPO, A2C

    search_map = {
        "dqn":       (DQN, "models/dqn"),
        "ppo":       (PPO, "models/pg"),
        "a2c":       (A2C, "models/pg"),
        "reinforce": (A2C, "models/pg"),
    }

    if algo == "auto":
        # Try DQN first, then PPO, A2C
        for name, (cls, folder) in search_map.items():
            model_path = _find_best_model_in(folder, name)
            if model_path:
                print(f"[main] Auto-selected: {name.upper()} – {model_path}")
                return cls.load(model_path), name
        raise FileNotFoundError("No trained model found. Run training scripts first.")

    cls, folder = search_map[algo]
    model_path = _find_best_model_in(folder, algo)
    if model_path is None:
        raise FileNotFoundError(f"No {algo.upper()} model found in {folder}/")
    return cls.load(model_path), algo


def _find_best_model_in(folder: str, algo: str) -> str | None:
    if not os.path.isdir(folder):
        return None
    candidates = []
    for fname in os.listdir(folder):
        if fname.endswith(".zip") and (algo.lower() in fname.lower() or "best" in fname):
            candidates.append(os.path.join(folder, fname))
    if not candidates:
        # Try subdirectories (EvalCallback saves best_model.zip inside subfolders)
        for subdir in os.listdir(folder):
            subpath = os.path.join(folder, subdir)
            if os.path.isdir(subpath):
                bm = os.path.join(subpath, "best_model.zip")
                if os.path.exists(bm):
                    candidates.append(bm)
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────
#  Run simulation
# ─────────────────────────────────────────────
def run_episode(model, env, renderer=None, deterministic=True, verbose=True):
    obs, info = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    if verbose:
        print("\n" + "="*55)
        print("  NEW EPISODE")
        print("="*55)

    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        if verbose and step % 10 == 0:
            act_str = "HOLD" if action == 0 else f"TREAT P{action}"
            print(f"  Step {step:3d} | Action: {act_str:10s} | "
                  f"R: {reward:+6.1f} | Total: {total_reward:7.2f} | "
                  f"Alive: {info['patients_alive']} | "
                  f"Deaths: {info['total_deaths']}")

        if renderer:
            renderer.update(env, action)
            time.sleep(0.05)

    if verbose:
        print("-"*55)
        print(f"  Episode ended at step {step}")
        print(f"  Total reward : {total_reward:.2f}")
        print(f"  Treated      : {info['total_treated']}")
        print(f"  Deaths       : {info['total_deaths']}")
        print("="*55)

    return total_reward, info


def run_simulation(model, n_episodes: int = 5, use_renderer: bool = True,
                   random_agent: bool = False):
    from environment.custom_env import MedicalTriageEnv

    env = MedicalTriageEnv(render_mode="human" if use_renderer else None)
    renderer = None

    if use_renderer:
        try:
            from environment.rendering import get_renderer
            renderer = get_renderer()
            print("[main] Panda3D renderer initialised.")
        except Exception as e:
            print(f"[main] Renderer unavailable ({e}). Running in terminal mode.")
            from environment.rendering import TerminalRenderer
            renderer = TerminalRenderer()

    episode_rewards = []
    episode_infos   = []

    for ep in range(n_episodes):
        print(f"\n[main] Episode {ep+1}/{n_episodes}")
        m = None if random_agent else model
        r, info = run_episode(m, env, renderer=renderer, verbose=True)
        episode_rewards.append(r)
        episode_infos.append(info)

    env.close()
    if renderer and hasattr(renderer, "close"):
        renderer.close()

    # Summary
    print("\n" + "="*55)
    print("  SIMULATION COMPLETE")
    print("="*55)
    print(f"  Episodes     : {n_episodes}")
    print(f"  Mean reward  : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Best episode : {max(episode_rewards):.2f}")
    print(f"  Worst episode: {min(episode_rewards):.2f}")
    total_treated = sum(i["total_treated"] for i in episode_infos)
    total_deaths  = sum(i["total_deaths"]  for i in episode_infos)
    print(f"  Total treated: {total_treated}")
    print(f"  Total deaths : {total_deaths}")
    print("="*55)


# ─────────────────────────────────────────────
#  Static demo (random agent, no model)
# ─────────────────────────────────────────────
def run_static_demo():
    """
    Task requirement: static file showing random-action agent in the
    custom environment (no trained model, just environment + rendering).
    """
    print("\n[main] Running STATIC DEMO – random agent, Panda3D visualization")
    run_simulation(model=None, n_episodes=3,
                   use_renderer=True, random_agent=True)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Medical Triage RL Agent – Best Model Simulation"
    )
    parser.add_argument("--model",    type=str, default="",
                        help="Path to a saved model .zip file.")
    parser.add_argument("--algo",     type=str, default="auto",
                        choices=["auto", "dqn", "ppo", "a2c", "reinforce"],
                        help="Algorithm family to load.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of simulation episodes.")
    parser.add_argument("--random",   action="store_true",
                        help="Run random-action agent (static demo, no model).")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable Panda3D renderer (terminal output only).")
    args = parser.parse_args()

    if args.random:
        run_static_demo()
        return

    if args.model:
        # Load from explicit path
        from stable_baselines3 import DQN, PPO, A2C
        algo_cls_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "reinforce": A2C}
        cls = algo_cls_map.get(args.algo, DQN)
        model = cls.load(args.model)
        algo  = args.algo
    else:
        try:
            model, algo = load_best_model(args.algo)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            print("Run training first:")
            print("  python training/dqn_training.py --best")
            print("  python training/pg_training.py  --best ppo")
            sys.exit(1)

    print(f"\n[main] Running {algo.upper()} agent for {args.episodes} episodes")
    run_simulation(
        model,
        n_episodes   = args.episodes,
        use_renderer = not args.no_render,
        random_agent = False,
    )


if __name__ == "__main__":
    main()
