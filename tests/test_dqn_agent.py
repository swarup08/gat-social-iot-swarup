"""
Test the DQN Agent
"""

from src.rl.dqn_agent import DQNAgent
from src.rl.environment import GraphPruningEnv


def main():

    env = GraphPruningEnv()

    state = env.reset()

    state_dim = len(state)

    action_dim = env.edge_index.shape[1]

    print("=" * 50)
    print("State Dimension :", state_dim)
    print("Action Dimension:", action_dim)
    print("=" * 50)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    action = agent.select_action(state)

    print("Selected Action:", action)

    next_state, reward, done, info = env.step(action)

    print("\nReward :", reward)
    print("Done   :", done)
    print("Info   :", info)


if __name__ == "__main__":
    main()