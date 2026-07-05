import copy

from src.utils.data_loader import load_social_iot_dataset
from src.utils.pyg_to_networkx import pyg_to_networkx
from src.models.edge_importance_from_gat import compute_edge_importance
from src.simulator.botnet_simulator import run_botnet_simulation
import numpy as np
from src.rl.reward import compute_reward

class GraphPruningEnv:
    """
    Reinforcement Learning Environment for
    Dynamic Graph Pruning in Social IoT Networks.
    """

    def __init__(
            self,
            max_steps=20,
            infection_prob=0.25,
            recovery_prob=0.02,
        ):

        self.max_steps = max_steps
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob

        # Load frozen dataset
        self.data = load_social_iot_dataset()

        # Convert to NetworkX graph
        self.original_graph = pyg_to_networkx(self.data)

        # Working graph
        self.graph = copy.deepcopy(self.original_graph)

        # Load attention-based edge importance
        self.edge_index, self.edge_scores = compute_edge_importance()

        # Episode variables
        self.current_step = 0
        self.current_episode = 0

        # Infection history
        self.infection_history = []

        self.removed_edges = set()

    def reset(self):
        """
        Reset the environment before starting a new RL episode.
        """

        # Restore original graph
        self.graph = copy.deepcopy(self.original_graph)

        # Reset counters
        self.current_step = 0
        self.current_episode += 1

        # Clear infection history
        self.infection_history = []

        # Reset removed edge registry
        self.removed_edges.clear()
        
        # Return initial state
        return self.get_state()

    def get_state(self):
        """
        Compute the current RL state.
        """

        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Infection ratio
        if len(self.infection_history) == 0:
            infection_ratio = 0.0
        else:
            infection_ratio = (
                self.infection_history[-1] / num_nodes
            )

        # Remaining edge ratio
        remaining_edge_ratio = (
            num_edges /
            self.original_graph.number_of_edges()
        )

        # Average node degree
        degrees = [
            self.graph.degree(n)
            for n in self.graph.nodes()
        ]

        avg_degree = np.mean(degrees)

        # Attention statistics
        mean_attention = float(
            self.edge_scores.mean()
        )

        std_attention = float(
            self.edge_scores.std()
        )

        # High-risk node ratio
        labels = self.data.y.numpy()

        high_risk_ratio = (
            np.sum(labels == 1)
            /
            len(labels)
        )

        state = np.array(
            [
                infection_ratio,
                remaining_edge_ratio,
                avg_degree,
                mean_attention,
                std_attention,
                high_risk_ratio,
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action):
        """
        Execute one pruning action.
        """

        # -------------------------
        # Validate Action
        # -------------------------
        if action < 0 or action >= self.edge_index.shape[1]:
            raise ValueError("Invalid action selected.")

        edge = self.edge_index[:, action]

        u = int(edge[0])
        v = int(edge[1])

        # -------------------------
        # Edge already removed?
        # -------------------------
        if not self.graph.has_edge(u, v):

            info = {
                "selected_edge": (u, v),
                "message": "Edge already removed."
            }

            return self.get_state(), -1.0, False, info

        # -------------------------
        # Remove Edge
        # -------------------------
        self.graph.remove_edge(u, v)

        # -------------------------
        # Run Botnet Simulation
        # -------------------------
        infection_history = run_botnet_simulation(
            self.graph,
            infection_prob=self.infection_prob,
            recovery_prob=self.recovery_prob
        )

        self.infection_history = infection_history

        # -------------------------
        # Compute Metrics
        # -------------------------
        infection_ratio = infection_history[-1] / self.graph.number_of_nodes()

        connectivity_ratio = (
            self.graph.number_of_edges()
            /
            self.original_graph.number_of_edges()
        )

        removed_edges = (
            self.original_graph.number_of_edges()
            - self.graph.number_of_edges()
        )

        pruning_cost = (
            removed_edges
            /
            self.original_graph.number_of_edges()
        )

        # -------------------------
        # Compute Reward
        # -------------------------
        reward = compute_reward(
            infection_ratio,
            connectivity_ratio,
            pruning_cost
        )

        # -------------------------
        # Update Step
        # -------------------------
        self.current_step += 1

        done = self.current_step >= self.max_steps

        next_state = self.get_state()

        info = {
            "selected_edge": (u, v),
            "infection_ratio": infection_ratio,
            "connectivity_ratio": connectivity_ratio,
            "pruning_cost": pruning_cost
        }

        return next_state, reward, done, info
    def render(self):
            """
            Display the current environment status.
            """
            pass