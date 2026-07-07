import math
import random
import matplotlib.pyplot as plt


def _sigmoid(z):
    # clip to avoid overflow in math.exp for extreme z
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def edge_infection_probability(graph, u, v, weights=None, base_prob=None):
    """
    Feature-dependent infection probability, per the roadmap's
    p_uv = sigmoid(beta^T phi(x_u, x_v, e_uv)) formulation.

    Falls back to a flat `base_prob` if the graph carries no node/edge
    features (e.g. legacy callers), so this never crashes on graphs
    built before this change.
    """
    if weights is None:
        # Tuned so baseline (no pruning) infection settles around ~55%
        # of the network rather than saturating near 100%, leaving
        # real headroom for pruning strategies to show a difference.
        # See dev notes: bias=-4.0 gave baseline ~0.56 vs ~0.31 after
        # isolating the single highest-degree hub, over 30 trials.
        weights = {"anomaly": 2.0, "bandwidth": 1.5, "latency": 1.0, "bias": -4.0}

    v_attrs = graph.nodes[v]
    e_attrs = graph.edges[u, v]

    if "anomaly_score" not in v_attrs or "bandwidth" not in e_attrs:
        return base_prob if base_prob is not None else 0.1

    anomaly_v = v_attrs["anomaly_score"]              # 0..1
    bandwidth_norm = e_attrs["bandwidth"] / 10.0        # ~0..1
    latency_norm = e_attrs["latency"] / 100.0           # ~0..1

    z = (
        weights["anomaly"] * anomaly_v
        + weights["bandwidth"] * bandwidth_norm
        - weights["latency"] * latency_norm
        + weights["bias"]
    )
    return _sigmoid(z)


class BotnetSimulator:
    """
    Simulates botnet propagation on a Social IoT graph.

    infection_prob is now used only as a fallback for graphs without
    features; when node/edge features are present (see
    src/utils/pyg_to_networkx.py), infection probability is computed
    per-edge from edge_infection_probability() so that spread depends
    on the same features GAT attention is scoring, and so that pruning
    a high-risk edge actually changes the epidemic outcome.
    """

    def __init__(self, graph, infection_prob=0.3, recovery_prob=0.05, weights=None):
        self.graph = graph
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.weights = weights

        # Infection state: 0 = clean, 1 = infected
        self.state = {node: 0 for node in self.graph.nodes}

        # Infection history
        self.infection_history = []

    def initialize_infection(self, num_seeds=3, verbose=False):
        """
        Initialize infection with randomly selected seed nodes.

        Parameters
        ----------
        num_seeds : int
            Number of initially infected nodes.

        verbose : bool
            Print infected seed nodes if True.
        """

        # Reset all nodes to clean before each simulation
        self.state = {node: 0 for node in self.graph.nodes}

        seeds = random.sample(list(self.graph.nodes), num_seeds)

        for s in seeds:
            self.state[s] = 1

        if verbose:
            print(f"Initial infected nodes: {seeds}")

    def step(self):
        """
        Perform one infection propagation step.
        """

        new_state = self.state.copy()

        for u in self.graph.nodes:

            if self.state[u] == 1:

                # Try to infect neighbors
                for v in self.graph.successors(u):

                    if self.state[v] == 0:

                        p_uv = edge_infection_probability(
                            self.graph, u, v,
                            weights=self.weights,
                            base_prob=self.infection_prob,
                        )
                        if random.random() < p_uv:
                            new_state[v] = 1

                # Recovery
                if random.random() < self.recovery_prob:
                    new_state[u] = 0

        self.state = new_state

    def run(self, steps=20, verbose=False):
        """
        Run the botnet propagation simulation.

        Parameters
        ----------
        steps : int
            Number of simulation steps.

        verbose : bool
            Print infection statistics if True.
        """

        self.infection_history = []

        for t in range(steps):

            infected_count = sum(self.state.values())

            self.infection_history.append(infected_count)

            if verbose:
                print(f"Time {t}: Infected nodes = {infected_count}")

            self.step()

    def plot(self):
        """
        Plot infection curve.
        """

        plt.figure(figsize=(7, 5))
        plt.plot(self.infection_history, marker="o")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Infected Nodes")
        plt.title("Botnet Propagation in Social IoT Graph")
        plt.grid(True)
        plt.show()


# ----------------------------------------------------
# Wrapper Function (Used by RL Environment)
# ----------------------------------------------------
def run_botnet_simulation(
    G,
    steps=20,
    infection_prob=0.25,
    recovery_prob=0.02,
    num_seeds=3,
    weights=None,
):
    """
    Wrapper used by RL environment.
    Returns only infection history.
    """

    simulator = BotnetSimulator(
        G,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
        weights=weights,
    )

    simulator.initialize_infection(
        num_seeds=num_seeds,
        verbose=False,
    )

    simulator.run(
        steps=steps,
        verbose=False,
    )

    return simulator.infection_history


# ----------------------------------------------------
# Manual Testing
# ----------------------------------------------------
if __name__ == "__main__":

    from src.graph.social_iot_graph import SocialIoTGraph

    graph_generator = SocialIoTGraph(num_nodes=40)

    graph_generator.generate_nodes()

    graph_generator.generate_edges(p=0.08)

    G = graph_generator.graph

    simulator = BotnetSimulator(
        G,
        infection_prob=0.25,
        recovery_prob=0.02,
    )

    simulator.initialize_infection(
        num_seeds=3,
        verbose=True,
    )

    simulator.run(
        steps=25,
        verbose=True,
    )

    simulator.plot()