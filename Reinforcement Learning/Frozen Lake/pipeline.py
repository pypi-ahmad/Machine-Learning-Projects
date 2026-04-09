"""
Modern Reinforcement Learning Pipeline (April 2026)
Models: PPO (default), SAC (continuous), DQN (baseline) — Stable-Baselines3
Data: Gymnasium environments (auto-downloaded)
"""
import os, warnings
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ENV_NAME = "FrozenLake-v1"
ALGO = "PPO"
TOTAL_TIMESTEPS = 100_000


def train_agent():
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    save_dir = os.path.dirname(os.path.abspath(__file__))

    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
        log_path=save_dir, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    if ALGO == "SAC":
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=256, tau=0.005, gamma=0.99, verbose=1, device="auto")
    elif ALGO == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000,
                     batch_size=64, gamma=0.99, exploration_fraction=0.3,
                     target_update_interval=1000, verbose=1, device="auto")
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, verbose=1, device="auto")

    print(f"Training {ALGO} on {ENV_NAME} for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"\n✓ {ALGO} — Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    model.save(os.path.join(save_dir, f"{ALGO.lower()}_{ENV_NAME}"))
    env.close(); eval_env.close()
    return model, mean_reward


def main():
    print("=" * 60)
    print(f"REINFORCEMENT LEARNING — {ALGO} on {ENV_NAME}")
    print("=" * 60)
    model, reward = train_agent()
    print(f"\n🏆 Final Mean Reward: {reward:.2f}")


if __name__ == "__main__":
    main()
