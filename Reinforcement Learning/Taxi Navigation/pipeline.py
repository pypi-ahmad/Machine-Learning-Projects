"""
Modern Reinforcement Learning Pipeline (April 2026)
Models: PPO (default), SAC (continuous), DQN (discrete baseline) — Stable-Baselines3
Data: Gymnasium environments (auto-downloaded)
"""
import os, warnings
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ENV_NAME = "Taxi-v3"
ALGO = "PPO"
TOTAL_TIMESTEPS = 100_000

# Simple discrete envs where DQN is a valid educational baseline
DISCRETE_ENVS = ("CliffWalking", "FrozenLake", "Taxi", "LunarLander", "CartPole", "MountainCar")


def train_single(algo_name, env, eval_env, save_dir, timesteps):
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir,
        log_path=save_dir, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    if algo_name == "SAC":
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=256, tau=0.005, gamma=0.99, verbose=1, device="auto")
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000,
                     batch_size=64, gamma=0.99, exploration_fraction=0.3,
                     target_update_interval=1000, verbose=1, device="auto")
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, verbose=1, device="auto")

    print(f"\n— Training {algo_name} on {ENV_NAME} for {timesteps} steps —")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"✓ {algo_name} — Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    model.save(os.path.join(save_dir, f"{algo_name.lower()}_{ENV_NAME}"))
    return algo_name, mean_reward, std_reward


def main():
    print("=" * 60)
    print(f"REINFORCEMENT LEARNING — {ALGO} on {ENV_NAME}")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    # Main algorithm (PPO or SAC)
    env = gym.make(ENV_NAME); eval_env = gym.make(ENV_NAME)
    name, reward, std = train_single(ALGO, env, eval_env, save_dir, TOTAL_TIMESTEPS)
    results.append((name, reward, std))
    env.close(); eval_env.close()

    # DQN baseline for simple discrete environments
    is_discrete = any(tag in ENV_NAME for tag in DISCRETE_ENVS)
    if is_discrete and ALGO != "DQN":
        try:
            env2 = gym.make(ENV_NAME); eval_env2 = gym.make(ENV_NAME)
            name, reward, std = train_single("DQN", env2, eval_env2, save_dir, TOTAL_TIMESTEPS)
            results.append((name, reward, std))
            env2.close(); eval_env2.close()
        except Exception as e:
            print(f"✗ DQN baseline: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    best = max(results, key=lambda x: x[1])
    for name, reward, std in results:
        marker = " 🏆" if name == best[0] else ""
        print(f"  {name}: {reward:.2f} ± {std:.2f}{marker}")


if __name__ == "__main__":
    main()
