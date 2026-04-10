"""
Modern Reinforcement Learning Pipeline (April 2026)

Primary algorithm: SAC
  - SAC  (Stable-Baselines3) -- default for continuous-action envs
  - PPO  (Stable-Baselines3) -- default for discrete-action envs

Baselines (comparison):
  - PPO  (Stable-Baselines3) -- comparison when SAC is primary
  - DQN  (Stable-Baselines3) -- deep RL baseline for discrete-action envs
  - Q-learning (tabular)     -- educational baseline for small-state discrete envs

Environment: BipedalWalker-v3
Action space: auto-detected (discrete -> PPO+DQN, continuous -> SAC+PPO)

Compute requirements:
  - PPO : ~100K steps, 1-3 min on CPU, <1 min with GPU
  - DQN : ~100K steps, 1-3 min on CPU, <1 min with GPU
  - SAC : ~100K steps, 2-5 min on CPU, <1 min with GPU
  - Q-learning (tabular): <10s, CPU-only, no neural network

Dependencies: stable-baselines3, gymnasium, matplotlib, numpy
"""
import os, json, time, warnings
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ENV_NAME = "BipedalWalker-v3"
ALGO = "SAC"
TOTAL_TIMESTEPS = 100_000
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MAKE_KWARGS = dict()

# Discrete envs where DQN is a valid baseline
DISCRETE_ENVS = ("CliffWalking", "FrozenLake", "Taxi", "LunarLander", "CartPole", "MountainCar")
# Small-state discrete envs where tabular Q-learning is educational
TABULAR_ENVS = ("CliffWalking", "FrozenLake", "Taxi")


def train_sb3(algo_name, env_instance, eval_env, save_dir, timesteps):
    """Train a Stable-Baselines3 agent (PPO, SAC, or DQN)."""
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir,
        log_path=save_dir, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    if algo_name == "SAC":
        model = SAC("MlpPolicy", env_instance, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=256, tau=0.005, gamma=0.99, verbose=1, device="auto")
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env_instance, learning_rate=1e-4, buffer_size=50_000,
                     batch_size=64, gamma=0.99, exploration_fraction=0.3,
                     target_update_interval=1000, verbose=1, device="auto")
    else:
        model = PPO("MlpPolicy", env_instance, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, verbose=1, device="auto")

    t0 = time.perf_counter()
    print(f"  Training {algo_name} on {ENV_NAME} for {timesteps} steps ...")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    elapsed = time.perf_counter() - t0
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"  {algo_name} - Reward: {mean_reward:.2f} +/- {std_reward:.2f} ({elapsed:.1f}s)")
    model.save(os.path.join(save_dir, f"{algo_name.lower()}_{ENV_NAME}"))
    return algo_name, mean_reward, std_reward, elapsed


def train_q_table(env_name, n_episodes=10_000, lr=0.1, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    """Tabular Q-learning - educational baseline for small-state discrete envs."""
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    epsilon = eps_start
    rewards_log = []

    t0 = time.perf_counter()
    print(f"  Training Q-learning (tabular) on {env_name} for {n_episodes} episodes ...")
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) * (1 - terminated) - Q[state, action])
            state = next_state
            total_reward += reward
        rewards_log.append(total_reward)
        epsilon = max(eps_end, epsilon * eps_decay)

    env.close()

    # Evaluate learned policy
    eval_env = gym.make(env_name)
    eval_rewards = []
    for _ in range(100):
        state, _ = eval_env.reset()
        total_r = 0; done = False
        while not done:
            action = int(np.argmax(Q[state]))
            state, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_r += r
        eval_rewards.append(total_r)
    eval_env.close()

    elapsed = time.perf_counter() - t0
    mean_r = np.mean(eval_rewards)
    std_r = np.std(eval_rewards)
    print(f"  Q-learning - Reward: {mean_r:.2f} +/- {std_r:.2f} ({elapsed:.1f}s, {n_states} states x {n_actions} actions)")
    return "Q-learning", mean_r, std_r, elapsed, rewards_log


def plot_results(results, save_dir):
    """Bar chart comparing mean rewards across all algorithms."""
    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(names)])
    ax.set_ylabel("Mean Reward (20 eval episodes)")
    ax.set_title(f"RL Algorithm Comparison - {ENV_NAME}")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_eda(env_name, make_kwargs, save_dir):
    """Environment information summary."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    try:
        env = gym.make(env_name, **make_kwargs)
        print(f"  Environment: {env_name}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        is_continuous = hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0
        print(f"  Action type: {'continuous' if is_continuous else 'discrete'}")
        if hasattr(env, 'reward_range'):
            print(f"  Reward range: {env.reward_range}")
        env.close()
    except Exception as e:
        print(f"  Could not inspect environment: {e}")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"REINFORCEMENT LEARNING | {ENV_NAME}")
    print(f"Primary: {ALGO}  |  Baselines: auto-selected for action space")
    print("=" * 60)
    run_eda(ENV_NAME, MAKE_KWARGS, SAVE_DIR)
    results = []

    # === PRIMARY: SAC or PPO ===
    env = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env = gym.make(ENV_NAME, **MAKE_KWARGS)
    is_continuous = hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0
    act_type = "continuous" if is_continuous else "discrete"
    print(f"  Environment: {ENV_NAME} ({act_type} actions)")

    name, reward, std, dt = train_sb3(ALGO, env, eval_env, SAVE_DIR, TOTAL_TIMESTEPS)
    results.append((name, reward, std, dt))
    env.close(); eval_env.close()

    # === DQN BASELINE (discrete environments) ===
    is_discrete = any(tag in ENV_NAME for tag in DISCRETE_ENVS)
    if is_discrete and ALGO != "DQN":
        try:
            env2 = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env2 = gym.make(ENV_NAME, **MAKE_KWARGS)
            name, reward, std, dt = train_sb3("DQN", env2, eval_env2, SAVE_DIR, TOTAL_TIMESTEPS)
            results.append((name, reward, std, dt))
            env2.close(); eval_env2.close()
        except Exception as e:
            print(f"  DQN baseline failed: {e}")

    # === PPO COMPARISON (continuous environments where SAC is primary) ===
    if is_continuous and ALGO != "PPO":
        try:
            env3 = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env3 = gym.make(ENV_NAME, **MAKE_KWARGS)
            name, reward, std, dt = train_sb3("PPO", env3, eval_env3, SAVE_DIR, TOTAL_TIMESTEPS)
            results.append((name, reward, std, dt))
            env3.close(); eval_env3.close()
        except Exception as e:
            print(f"  PPO comparison failed: {e}")

    # === Q-LEARNING BASELINE (small-state discrete environments) ===
    is_tabular = any(tag in ENV_NAME for tag in TABULAR_ENVS)
    if is_tabular:
        try:
            name, reward, std, dt, _ = train_q_table(ENV_NAME)
            results.append((name, reward, std, dt))
        except Exception as e:
            print(f"  Q-learning baseline failed: {e}")

    # === SUMMARY ===
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    best = max(results, key=lambda x: x[1])
    for name, reward, std, dt in results:
        marker = " <- best" if name == best[0] else ""
        print(f"  {name:15s}  Reward: {reward:8.2f} +/- {std:6.2f}  ({dt:.1f}s){marker}")

    # Save metrics
    metrics = [{"algorithm": r[0], "mean_reward": r[1], "std_reward": r[2], "time_s": r[3]} for r in results]
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    plot_results(results, SAVE_DIR)
    print(f"  Saved: comparison.png, metrics.json")


if __name__ == "__main__":
    main()
