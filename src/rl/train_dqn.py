"""
====================================================
Train DQN for Dynamic Graph Pruning
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.rl.environment import GraphPruningEnv
from src.rl.dqn_agent import DQNAgent
from src.rl.replay_buffer import ReplayBuffer


def train(
    episodes=400,
    batch_size=32,
    target_update=5,
    epsilon_decay=0.99,
    epsilon_min=0.05,
    env=None,
    save_path="results/models/dqn_dynamic_pruning.pth",
    save_history=True,
    verbose=True,
):
    """
    epsilon_decay=0.99 with 400 episodes reaches epsilon_min (0.05)
    around episode ~300, leaving the last ~25% of training running
    mostly greedy so the reward curve can show genuine convergence
    rather than epsilon still being ~0.95 at the end (the previous
    episodes=10/decay=0.995 defaults never got anywhere near
    epsilon_min, so the saved policy was close to random).

    `env` defaults to a GraphPruningEnv on the canonical frozen
    dataset; pass a pre-built env (e.g. wrapping a freshly generated
    graph) to train on something else. `save_path=None` skips writing
    a checkpoint to disk and just returns the trained agent in memory
    (used by multi-graph evaluation, which trains/discards 30-50
    short-lived models and shouldn't litter results/models/).
    """

    if env is None:
        env = GraphPruningEnv()

    state = env.reset()

    state_dim = len(state)
    action_dim = env.edge_index.shape[1]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    replay_buffer = ReplayBuffer()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_history:
        os.makedirs("results/history", exist_ok=True)
        os.makedirs("results/figures", exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Starting DQN Training")
        print("=" * 60)

    reward_history = []
    epsilon_history = []
    loss_history = []

    for episode in range(episodes):

        state = env.reset()

        done = False

        total_reward = 0.0

        episode_loss = 0.0

        step_count = 0

        while not done:

            valid_actions = env.get_valid_actions()

            if len(valid_actions) == 0:
                break

            action = agent.select_action(state, valid_actions)

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            if len(replay_buffer) >= batch_size:

                loss = agent.learn(
                    replay_buffer,
                    batch_size
                )

                if loss is not None:
                    episode_loss += loss

            state = next_state

            total_reward += reward

            step_count += 1

        agent.decay_epsilon()

        if (episode + 1) % target_update == 0:

            agent.update_target()

        avg_loss = 0.0

        if step_count > 0:
            avg_loss = episode_loss / step_count

        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        loss_history.append(avg_loss)

        if verbose:
            print(
                f"Episode {episode+1:03d} | "
                f"Reward={total_reward:.3f} | "
                f"Loss={avg_loss:.4f} | "
                f"Epsilon={agent.epsilon:.3f}"
            )

    if save_path is not None:
        agent.save_model(save_path)

    history_df = pd.DataFrame({
        "episode": range(1, episodes + 1),
        "reward": reward_history,
        "epsilon": epsilon_history,
        "avg_loss": loss_history,
    })

    if save_history:
        history_df.to_csv("results/history/training_history.csv", index=False)
        _plot_reward_curve(history_df)

    if verbose:
        print("\nTraining Complete")
        if save_path is not None:
            print("Model Saved Successfully")
        if save_history:
            print("Reward history saved to results/history/training_history.csv")
            print("Reward curve saved to results/figures/dqn_training_reward.png")

    return agent, env, history_df


def _plot_reward_curve(history_df, window=20):
    """
    Reward-per-episode with a rolling mean, so a genuinely learning
    policy (reward trending up then stabilizing) is visually
    distinguishable from a flat/noisy one.
    """

    rolling_mean = history_df["reward"].rolling(window, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        history_df["episode"], history_df["reward"],
        alpha=0.3, label="Reward (raw)",
    )
    ax1.plot(
        history_df["episode"], rolling_mean,
        linewidth=2, label=f"Reward ({window}-episode rolling mean)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        history_df["episode"], history_df["epsilon"],
        color="gray", linestyle="--", alpha=0.6, label="Epsilon",
    )
    ax2.set_ylabel("Epsilon")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("DQN Training: Reward per Episode")
    plt.tight_layout()
    plt.savefig("results/figures/dqn_training_reward.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":

    train()