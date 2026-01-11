import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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

        # History
        self.infection_history = []

    def initialize_infection(self, num_seeds=3):
        seeds = random.sample(list(self.graph.nodes), num_seeds)
        for s in seeds:
            self.state[s] = 1
        print(f"Initial infected nodes: {seeds}")

    def step(self):
        new_state = self.state.copy()

        for u in self.graph.nodes:
            if self.state[u] == 1:
                # Try to infect neighbors
                for v in self.graph.successors(u):
                    if self.state[v] == 0:
                        if random.random() < self.infection_prob:
                            new_state[v] = 1

                # Possible recovery
                if random.random() < self.recovery_prob:
                    new_state[u] = 0

        self.state = new_state

    def run(self, steps=20):
        for t in range(steps):
            infected_count = sum(self.state.values())
            self.infection_history.append(infected_count)
            print(f"Time {t}: Infected nodes = {infected_count}")
            self.step()

    def plot(self):
        plt.figure(figsize=(7, 5))
        plt.plot(self.infection_history, marker='o')
        plt.xlabel("Time step")
        plt.ylabel("Number of infected nodes")
        plt.title("Botnet Propagation in Social IoT Graph")
        plt.grid(True)
        plt.show()


# -----------------------------
# Testing the simulator
# -----------------------------
if __name__ == "__main__":
    from src.graph.social_iot_graph import SocialIoTGraph

    # Generate Social IoT Graph
    graph_generator = SocialIoTGraph(num_nodes=40)
    graph_generator.generate_nodes()
    graph_generator.generate_edges(p=0.08)
    G = graph_generator.graph

    # Create simulator
    simulator = BotnetSimulator(G, infection_prob=0.25, recovery_prob=0.02)

    # Initialize infection
    simulator.initialize_infection(num_seeds=3)

    # Run simulation
    simulator.run(steps=25)

    # Plot results
    simulator.plot()
