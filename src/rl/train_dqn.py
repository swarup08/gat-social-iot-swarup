"""
====================================================
Train DQN for Dynamic Graph Pruning
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

import os

from src.rl.environment import GraphPruningEnv
from src.rl.dqn_agent import DQNAgent
from src.rl.replay_buffer import ReplayBuffer


def train(
    episodes=10,
    batch_size=32,
    target_update=5,
):

    env = GraphPruningEnv()

    state = env.reset()

    state_dim = len(state)
    action_dim = env.edge_index.shape[1]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    replay_buffer = ReplayBuffer()

    os.makedirs("results/models", exist_ok=True)

    print("=" * 60)
    print("Starting DQN Training")
    print("=" * 60)

    for episode in range(episodes):

        state = env.reset()

        done = False

        total_reward = 0.0

        episode_loss = 0.0

        step_count = 0

        while not done:

            action = agent.select_action(state)

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

        print(
            f"Episode {episode+1:03d} | "
            f"Reward={total_reward:.3f} | "
            f"Loss={avg_loss:.4f} | "
            f"Epsilon={agent.epsilon:.3f}"
        )

    agent.save_model(
        "results/models/dqn_dynamic_pruning.pth"
    )

    print("\nTraining Complete")
    print("Model Saved Successfully")


if __name__ == "__main__":

    train()