"""
====================================================
Evaluate Trained DQN Agent
Project : GAT-based Secure Social IoT Framework
Author  : Swarup Sarkar
====================================================
"""

from src.rl.environment import GraphPruningEnv
from src.rl.dqn_agent import DQNAgent
from src.xai.pruning_logger import PruningLogger


MODEL_PATH = "results/models/dqn_dynamic_pruning.pth"

def evaluate_dynamic_rl(
        verbose=True,
        num_seeds=3,
        infection_prob=0.25,
        return_history=False,
        max_steps=20,
        model_path=None,
):
    # -----------------------------
    # Create Environment
    # -----------------------------
    #env = GraphPruningEnv()

    env = GraphPruningEnv(
        num_seeds=num_seeds,
        infection_prob=infection_prob,
        max_steps=max_steps,
    )
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
    agent.load_model(model_path or MODEL_PATH)
    logger = PruningLogger()

    #print("Valid Actions:", len(env.get_valid_actions()))
    #print(env.get_valid_actions()[:10])

    if verbose:
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
        edge = info["selected_edge"]

        u = edge[0]
        v = edge[1]

        attention_score = float(
            env.edge_scores[action]
        )

        source_features = env.data.x[u].tolist()
        destination_features = env.data.x[v].tolist()

        logger.log(
            step=step + 1,
            source=u,
            destination=v,
            attention_score=attention_score,
            source_features=source_features,
            destination_features=destination_features,
            infection_ratio=info["infection_ratio"],
            connectivity_ratio=info["connectivity_ratio"],
            reward=reward,
        )
        #print("Action Index:", action)

        total_reward += reward

        infection_history.append(
            info["infection_ratio"]
        )

        connectivity_history.append(
            info["connectivity_ratio"]
        )

       # print(
        #    f"Step {step+1:02d} | "
        #    f"Edge {info['selected_edge']} | "
        #    f"Infection={info['infection_ratio']:.3f} | "
        #    f"Connectivity={info['connectivity_ratio']:.3f} | "
        #    f"Reward={reward:.3f}"
            
        #)
        if verbose:
            print(
                f"Step {step+1:02d} | "
                f"Edge {info['selected_edge']} | "
                f"Infection={info['infection_ratio']:.3f} | "
                f"Connectivity={info['connectivity_ratio']:.3f} | "
                f"Reward={reward:.3f}"
            )
        #print("Remaining Valid Actions:", len(env.get_valid_actions()))
        state = next_state

        step += 1

    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        print(f"Total Reward        : {total_reward:.3f}")
        print(f"Average Infection   : {sum(infection_history)/len(infection_history):.3f}")
        print(f"Average Connectivity: {sum(connectivity_history)/len(connectivity_history):.3f}")

    removed_edges = (
    env.original_graph.number_of_edges()
    -
    env.graph.number_of_edges()
)

    result = {
        "method": "Dynamic RL",
        "infection": sum(infection_history) / len(infection_history),
        "connectivity": sum(connectivity_history) / len(connectivity_history),
        "removed_edges": removed_edges,
        "total_reward": total_reward,
    }

    if return_history:
        return infection_history, connectivity_history

    return result
if __name__ == "__main__":
    evaluate_dynamic_rl()