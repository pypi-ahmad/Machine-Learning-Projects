#!/usr/bin/env python3
"""
Frozen Lake — DQN
==================
Train a Deep Q-Network agent on the **stochastic** version of
**FrozenLake-v1** (4×4, ``is_slippery=True``) from OpenAI Gymnasium.

With slipperiness enabled, the intended action is executed only
one-third of the time; the agent may slide perpendicular to the
chosen direction.  This makes the problem significantly harder and
requires more episodes to learn a good policy.

The 16 discrete states (4×4 grid) are one-hot encoded before being
fed into the DQN.

Environment reference:
    https://gymnasium.farama.org/environments/toy_text/frozen_lake/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import random

from shared.utils import (
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    ensure_dir,
    dataset_prompt,
    parse_common_args,
    configure_cuda_allocator,
    run_metadata,
    save_metrics,
    write_split_manifest,
    missing_dependency_metrics,
    resolve_device_from_args,
)

logger = logging.getLogger(__name__)

# ── Hyper-parameters ─────────────────────────────────────────
ENV_NAME = "FrozenLake-v1"
MAP_SIZE = "4x4"
STATE_DIM = 16            # one-hot encoded (4×4 grid)
ACTION_DIM = 4            # left, down, right, up
NUM_EPISODES = 2000       # stochastic env needs more training
LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998         # slower decay for longer exploration
BATCH_SIZE = 64
BUFFER_SIZE = 20_000
TARGET_UPDATE = 10
HIDDEN = 128


# ── DQN ──────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── One-hot helper ───────────────────────────────────────────
def one_hot(state_int: int, dim: int = STATE_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[state_int] = 1.0
    return vec


# ── Training loop ────────────────────────────────────────────
def train_dqn(device: torch.device, output_dir: Path, num_episodes: int = NUM_EPISODES, batch_size: int = BATCH_SIZE):
    try:
        import gymnasium as gym
    except ImportError:
        missing_dependency_metrics(
            output_dir,
            missing=["gymnasium"],
            install_cmd='pip install -U "gymnasium[box2d]"',
        )

    env = gym.make(ENV_NAME, map_name=MAP_SIZE, is_slippery=True)

    policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN).to(device)
    target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)
    epsilon = EPS_START
    rewards_history: list[float] = []
    best_reward = -float("inf")

    for ep in range(1, num_episodes + 1):
        state_int, _ = env.reset()
        state = one_hot(state_int)
        ep_reward = 0.0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q.argmax(1).item()

            next_state_int, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = one_hot(next_state_int)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Train on a mini-batch
            if len(buffer) >= batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)
                s_t = torch.FloatTensor(s).to(device)
                a_t = torch.LongTensor(a).to(device)
                r_t = torch.FloatTensor(r).to(device)
                ns_t = torch.FloatTensor(ns).to(device)
                d_t = torch.FloatTensor(d).to(device)

                q_vals = policy_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1)[0]
                    target = r_t + GAMMA * next_q * (1 - d_t)

                loss = nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        rewards_history.append(ep_reward)
        best_reward = max(best_reward, ep_reward)

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            logger.info(
                "Episode %d — reward=%.1f  avg100=%.2f  eps=%.3f",
                ep, ep_reward, avg, epsilon,
            )

    env.close()

    # ── Save results ─────────────────────────────────────────
    # Reward curve (use larger window for stochastic env)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards_history, alpha=0.15, label="Episode reward")
    window = 100
    if len(rewards_history) >= window:
        avg_curve = np.convolve(rewards_history, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards_history)), avg_curve, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"DQN Training on {ENV_NAME} ({MAP_SIZE}, stochastic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved reward curve → %s", output_dir / "reward_curve.png")

    # Model weights
    model_path = output_dir / "dqn_frozen_lake.pth"
    torch.save(policy_net.state_dict(), model_path)
    logger.info("Saved model weights → %s", model_path)

    return policy_net, rewards_history


# ── Evaluation ───────────────────────────────────────────────
def evaluate_policy(policy_net, env, device, num_episodes=100, seed=42):
    """Run greedy evaluation episodes (no exploration)."""
    rewards = []
    successes = 0
    for ep in range(num_episodes):
        state_int, _ = env.reset(seed=seed + ep)
        state = one_hot(state_int)
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax(1).item()
            next_state_int, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = one_hot(next_state_int)
            total_reward += reward
        rewards.append(total_reward)
        if total_reward > 0:  # reached goal (reward == 1)
            successes += 1
    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / num_episodes,
        "num_eval_episodes": num_episodes,
    }


# ── Entry point ──────────────────────────────────────────────
def main() -> None:
    setup_logging()
    args = parse_common_args("DQN on FrozenLake-v1 stochastic")
    set_seed(args.seed)
    configure_cuda_allocator()

    paths = project_paths(__file__)
    output_dir = ensure_dir(paths["outputs"])
    device = resolve_device_from_args(args)

    if args.download_only:
        print("No dataset to download (Gymnasium env)")
        sys.exit(0)

    train_episodes = args.epochs or NUM_EPISODES
    eval_episodes = 100
    if args.mode == "smoke":
        train_episodes = 50
        eval_episodes = 20
    batch_size = args.batch_size or BATCH_SIZE

    dataset_prompt(
        name="OpenAI Gymnasium FrozenLake-v1 (4×4, stochastic)",
        official_links=["https://gymnasium.farama.org/environments/toy_text/frozen_lake/"],
        notes="Built-in Gymnasium environment — no download required. "
              "Using is_slippery=True for the stochastic variant.",
    )

    logger.info("Training DQN on %s (%s, stochastic) for %d episodes …", ENV_NAME, MAP_SIZE, train_episodes)
    policy_net, rewards_history = train_dqn(device, output_dir, train_episodes, batch_size)

    # ── Evaluation ────────────────────────────────────────────
    try:
        import gymnasium as gym
    except ImportError:
        missing_dependency_metrics(
            output_dir,
            missing=["gymnasium"],
            install_cmd='pip install -U "gymnasium[box2d]"',
        )
    eval_env = gym.make(ENV_NAME, map_name=MAP_SIZE, is_slippery=True)
    logger.info("Evaluating policy for %d episodes …", eval_episodes)
    eval_metrics = evaluate_policy(policy_net, eval_env, device,
                                   num_episodes=eval_episodes, seed=args.seed)
    eval_env.close()
    logger.info("Eval avg_reward=%.2f  success_rate=%.2f",
                eval_metrics["avg_reward"], eval_metrics["success_rate"])

    # ── Write split manifest ──────────────────────────────────
    write_split_manifest(
        output_dir,
        dataset_fp={"env_name": ENV_NAME, "map_size": MAP_SIZE, "is_slippery": True},
        split_method="rl_train_eval",
        seed=args.seed,
        counts={"train_episodes": train_episodes, "eval_episodes": eval_episodes},
    )

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        "avg_reward": eval_metrics["avg_reward"],
        "std_reward": eval_metrics["std_reward"],
        "success_rate": eval_metrics["success_rate"],
        "num_eval_episodes": eval_metrics["num_eval_episodes"],
        "train_episodes": train_episodes,
        "split": "eval",
        "status": "ok",
    }
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(output_dir, metrics, task_type="rl", mode=args.mode)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
