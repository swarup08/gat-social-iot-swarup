import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


class SocialIoTGraph:
    """
    This class generates a synthetic Social IoT graph.
    Nodes = IoT devices, users, cloud services
    Edges = communication / trust / service relationships
    """

    def __init__(self, num_nodes=50):
        self.num_nodes = num_nodes
        self.graph = nx.DiGraph()

        # Device types
        self.device_types = ["sensor", "actuator", "gateway", "user", "cloud"]

    def generate_nodes(self):
        for i in range(self.num_nodes):
            device_type = random.choice(self.device_types)

            # Node feature vector
            # [device_type_id, degree (initially 0), avg_packet_rate, anomaly_score]
            x_v = {
                "device_type": device_type,
                "avg_packet_rate": random.uniform(10, 100),
                "anomaly_score": random.uniform(0, 1)
            }

            self.graph.add_node(i, **x_v)

    def generate_edges(self, p=0.05):
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and random.random() < p:
                    # Edge feature vector
                    e_uv = {
                        "protocol": random.choice(["MQTT", "HTTP", "CoAP"]),
                        "bandwidth": random.uniform(0.5, 10),  # Mbps
                        "latency": random.uniform(1, 100)      # ms
                    }
                    self.graph.add_edge(u, v, **e_uv)

    def summary(self):
        print("Number of nodes:", self.graph.number_of_nodes())
        print("Number of edges:", self.graph.number_of_edges())

        device_count = {}
        for _, data in self.graph.nodes(data=True):
            dt = data["device_type"]
            device_count[dt] = device_count.get(dt, 0) + 1

        print("Device type distribution:")
        for k, v in device_count.items():
            print(f"  {k}: {v}")

    def visualize(self):
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, node_size=50, arrows=False)
        plt.title("Synthetic Social IoT Graph")
        plt.show()


if __name__ == "__main__":
    graph_generator = SocialIoTGraph(num_nodes=40)
    graph_generator.generate_nodes()
    graph_generator.generate_edges(p=0.08)
    graph_generator.summary()
    graph_generator.visualize()
