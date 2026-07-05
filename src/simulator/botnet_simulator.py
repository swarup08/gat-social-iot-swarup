import random
import matplotlib.pyplot as plt


class BotnetSimulator:
    """
    Simulates botnet propagation on a Social IoT graph.
    """

    def __init__(self, graph, infection_prob=0.3, recovery_prob=0.05):
        self.graph = graph
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob

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

                        if random.random() < self.infection_prob:
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
):
    """
    Wrapper used by RL environment.
    Returns only infection history.
    """

    simulator = BotnetSimulator(
        G,
        infection_prob=infection_prob,
        recovery_prob=recovery_prob,
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