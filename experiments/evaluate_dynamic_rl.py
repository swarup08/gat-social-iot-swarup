"""
====================================================
Evaluate Trained DQN Agent
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

from src.rl.environment import GraphPruningEnv
from src.rl.dqn_agent import DQNAgent


MODEL_PATH = "results/models/dqn_dynamic_pruning.pth"


def evaluate():

    # -----------------------------
    # Create Environment
    # -----------------------------
    env = GraphPruningEnv()

    state = env.reset()

    # -----------------------------
    # Create Agent
    # -----------------------------
    state_dim = len(state)
    action_dim = env.edge_index.shape[1]

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )

    # -----------------------------
    # Load Trained Model
    # -----------------------------
    agent.load_model(MODEL_PATH)
    print("Valid Actions:", len(env.get_valid_actions()))
    print(env.get_valid_actions()[:10])

    print("=" * 60)
    print("Dynamic RL Evaluation")
    print("=" * 60)

    total_reward = 0.0

    infection_history = []
    connectivity_history = []

    done = False

    step = 0

   
    while not done:
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            print("No valid actions remaining.")
            break
        # Pure exploitation
        #action = agent.predict_action(state)
        action = agent.predict_action(
            state,
            valid_actions
        )

        next_state, reward, done, info = env.step(action)
        print("Action Index:", action)

        total_reward += reward

        infection_history.append(
            info["infection_ratio"]
        )

        connectivity_history.append(
            info["connectivity_ratio"]
        )

        print(
            f"Step {step+1:02d} | "
            f"Edge {info['selected_edge']} | "
            f"Infection={info['infection_ratio']:.3f} | "
            f"Connectivity={info['connectivity_ratio']:.3f} | "
            f"Reward={reward:.3f}"
            
        )
        print("Remaining Valid Actions:", len(env.get_valid_actions()))
        state = next_state

        step += 1

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Total Reward        : {total_reward:.3f}")
    print(f"Average Infection   : {sum(infection_history)/len(infection_history):.3f}")
    print(f"Average Connectivity: {sum(connectivity_history)/len(connectivity_history):.3f}")


if __name__ == "__main__":

    evaluate()