"""
====================================================
Reward Weight Ablation Study
Project : GAT-based Secure Social IoT Framework
====================================================
"""

import pandas as pd

from src.rl.environment import GraphPruningEnv
from src.rl.dqn_agent import DQNAgent

MODEL_PATH = "results/models/dqn_dynamic_pruning.pth"

REWARD_SETTINGS = [
    ("Security_Focused", 0.7, 0.2, 0.1),
    ("Balanced", 0.5, 0.3, 0.2),
    ("Utility_Focused", 0.3, 0.5, 0.2),
]

def evaluate_reward_setting(name, alpha, beta, gamma):

    print("\n" + "=" * 60)
    print(f"Reward Setting : {name}")
    print("=" * 60)

    # -------------------------
    # Environment
    # -------------------------
    env = GraphPruningEnv(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    state = env.reset()

    # -------------------------
    # Agent
    # -------------------------
    agent = DQNAgent(
        state_dim=len(state),
        action_dim=env.edge_index.shape[1],
    )

    agent.load_model(MODEL_PATH)

    total_reward = 0.0

    infection = []
    connectivity = []

    done = False

    while not done:

        valid_actions = env.get_valid_actions()

        if len(valid_actions) == 0:
            break

        action = agent.predict_action(
            state,
            valid_actions,
        )

        state, reward, done, info = env.step(action)

        total_reward += reward

        infection.append(
            info["infection_ratio"]
        )

        connectivity.append(
            info["connectivity_ratio"]
        )

    return {
        "Reward_Setting": name,
        "Average_Infection": sum(infection) / len(infection),
        "Average_Connectivity": sum(connectivity) / len(connectivity),
        "Total_Reward": total_reward,
    }
def run_ablation():

    results = []

    for name, alpha, beta, gamma in REWARD_SETTINGS:

        result = evaluate_reward_setting(
            name,
            alpha,
            beta,
            gamma,
        )

        results.append(result)

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print("Reward Weight Ablation Results")
    print("=" * 70)
    print(df)

    df.to_csv(
        "results/tables/reward_weight_ablation.csv",
        index=False,
    )


if __name__ == "__main__":

    run_ablation()