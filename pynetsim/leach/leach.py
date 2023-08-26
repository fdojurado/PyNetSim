import random
import matplotlib.pyplot as plt


class LEACH:

    def __init__(self, network):
        self.name = "LEACH"
        self.config = network.config
        self.network = network
        self.initialize_energy_parameters()

    def initialize_energy_parameters(self):
        protocol = self.config.network.protocol
        self.elect = protocol.eelect_nano * 10**-9
        self.etx = protocol.etx_nano * 10**-9
        self.erx = protocol.erx_nano * 10**-9
        self.eamp = protocol.eamp_pico * 10**-12
        self.eda = protocol.eda_nano * 10**-9
        self.packet_size = protocol.packet_size

    def select_cluster_heads(self, probability, tleft, num_cluster_heads):
        for node in self.network.nodes.values():
            if self.should_skip_node(node):
                continue

            if node.rounds_to_become_cluster_head > 0:
                node.rounds_to_become_cluster_head -= 1

            node.is_cluster_head = False
            node.cluster_id = 0

            if self.should_select_cluster_head(node, probability):
                num_cluster_heads = self.mark_as_cluster_head(
                    node, num_cluster_heads, tleft)

        cluster_heads = [
            node.node_id for node in self.network.nodes.values() if node.is_cluster_head]
        print(f"Cluster heads: {cluster_heads}")

    def should_skip_node(self, node):
        return node.node_id == 1 or node.energy <= 0

    def should_select_cluster_head(self, node, probability):
        return (not node.is_cluster_head and node.rounds_to_become_cluster_head == 0
                and random.random() < probability)

    def mark_as_cluster_head(self, node, num_cluster_heads, tleft):
        num_cluster_heads += 1
        node.is_cluster_head = True
        node.rounds_to_become_cluster_head = 1 / \
            self.config.network.protocol.cluster_head_percentage - tleft
        node.dst_to_sink = (
            (node.x - self.network.nodes[1].x)**2 + (node.y - self.network.nodes[1].y)**2)**0.5
        node.cluster_id = num_cluster_heads
        print(f"Node {node.node_id} is a cluster head.")
        return num_cluster_heads

    def create_clusters(self):
        cluster_heads_exist = any(
            node.is_cluster_head for node in self.network.nodes.values())
        if not cluster_heads_exist:
            print("There are no cluster heads.")
            self.clear_clusters()
            return False

        for node in self.network.nodes.values():
            if not node.is_cluster_head and node.node_id != 1:
                self.add_node_to_cluster(node)

        return True

    def clear_clusters(self):
        for node in self.network.nodes.values():
            node.cluster_id = 0

    def add_node_to_cluster(self, node):
        distances = {cluster_head.node_id: ((node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)**0.5
                     for cluster_head in self.network.nodes.values() if cluster_head.is_cluster_head}
        cluster_head_id = min(distances, key=distances.get)
        min_distance = distances[cluster_head_id]
        cluster_head = self.network.nodes[cluster_head_id]
        cluster_head.add_neighbor(node)
        node.add_neighbor(cluster_head)
        node.dst_to_cluster_head = min_distance
        node.cluster_id = cluster_head.cluster_id
        print(
            f"Node {node.node_id} is in the cluster of node {cluster_head_id}.")

    def energy_dissipation_non_cluster_heads(self, round):
        print("Energy dissipation for non-cluster heads")
        for node in self.network.nodes.values():
            if self.should_skip_node(node) or node.is_cluster_head:
                continue

            cluster_head = self.get_cluster_head(node)
            if cluster_head is None:
                self.transfer_data_to_sink(node, round)
            else:
                self.process_non_cluster_head(node, cluster_head, round)

    def get_cluster_head(self, node):
        return self.network.get_node_with_cluster_id(node.cluster_id)

    def transfer_data_to_sink(self, node, round):
        distance = node.dst_to_sink
        print("No cluster heads, transferring data to the sink.")
        print(f"Node {node.node_id} distance to sink: {distance}")
        ETx = self.elect * self.packet_size + self.eamp * self.packet_size * distance**2
        node.energy -= ETx
        self.network.remaining_energy -= ETx
        if node.energy <= 0:
            self.mark_node_as_dead(node, round)

    def process_non_cluster_head(self, node, cluster_head, round):
        distance = self.get_node_distance(node, cluster_head)
        print(f"Node {node.node_id} distance to cluster head: {distance}")
        ETx = self.calculate_tx_energy_dissipation(distance)
        node.energy -= ETx
        self.network.remaining_energy -= ETx
        ERx = (self.elect + self.eda) * self.packet_size
        cluster_head.energy -= ERx
        self.network.remaining_energy -= ERx
        if cluster_head.energy <= 0:
            print(f"Cluster head {cluster_head.node_id} is dead.")
            self.mark_node_as_dead(cluster_head, round)
            self.remove_cluster_head(cluster_head)
            self.remove_node_from_cluster(cluster_head)
            self.remove_cluster_head_from_cluster(cluster_head)
        if node.energy <= 0:
            print(f"Node {node.node_id} is dead.")
            self.mark_node_as_dead(node, round)
            self.remove_node_from_cluster(node)

    def get_node_distance(self, node, cluster_head):
        return node.dst_to_cluster_head if cluster_head.energy > 0 else node.dst_to_sink

    def calculate_tx_energy_dissipation(self, distance):
        return self.elect * self.packet_size + self.eamp * self.packet_size * distance**2

    def mark_node_as_dead(self, node, round):
        print(f"Node {node.node_id} is dead.")
        node.round_dead = round

    def remove_cluster_head(self, cluster_head):
        cluster_id = cluster_head.cluster_id
        for node in self.network.nodes.values():
            if node.cluster_id == cluster_id:
                node.cluster_id = 0

    def remove_node_from_cluster(self, node):
        for neighbor in node.neighbors.values():
            print(f"Removing node {node.node_id} from node {neighbor.node_id}")
            # if the node is not dead, remove it from the neighbor's neighbors
            if neighbor.energy > 0:
                neighbor.neighbors.pop(node.node_id)
        node.neighbors = {}

    def energy_dissipation_cluster_heads(self, round):
        for node in self.network.nodes.values():
            if not node.is_cluster_head or node.energy <= 0:
                continue
            distance = node.dst_to_sink
            print(
                f"Cluster head {node.node_id} with cluster id {node.cluster_id} distance to sink: {distance}")
            ETx = (self.elect + self.eda) * self.packet_size + \
                self.eamp * self.packet_size * distance**2
            node.energy -= ETx
            self.network.remaining_energy -= ETx
            if node.energy <= 0:
                self.mark_node_as_dead(node, round)
                self.remove_cluster_head_from_cluster(node)

    def remove_cluster_head_from_cluster(self, cluster_head):
        cluster_id = cluster_head.cluster_id
        for child in cluster_head.neighbors.values():
            if child.cluster_id == cluster_id:
                child.cluster_id = 0
                # Find the new cluster head for the child
                self.add_node_to_cluster(child)
        cluster_head.neighbors = {}

    def plot_clusters(self, round):
        plt.figure()
        self.plot_nodes()
        self.plot_cluster_connections()
        self.annotate_node_ids()
        self.plot_sink_connections()
        plt.title(f"Round {round}")
        plt.show()

    def plot_nodes(self):
        for node in self.network.nodes.values():
            if node.node_id == 1:
                node.color = "black"
            elif node.is_cluster_head:
                node.color = "red"
            else:
                node.color = "blue"
            plt.plot(node.x, node.y, 'o', color=node.color)

    def plot_cluster_connections(self):
        cluster_heads_exist = any(
            node.is_cluster_head for node in self.network.nodes.values())
        if not cluster_heads_exist:
            print("There are no cluster heads.")
            return

        for node in self.network.nodes.values():
            if node.is_cluster_head or node.node_id == 1:
                continue
            cluster_head = self.get_cluster_head(node)
            plt.plot([node.x, cluster_head.x], [
                     node.y, cluster_head.y], 'k--', linewidth=0.5)

        for node in self.network.nodes.values():
            if node.is_cluster_head:
                plt.plot([node.x, self.network.nodes[1].x], [
                         node.y, self.network.nodes[1].y], 'k-', linewidth=1)

    def annotate_node_ids(self):
        for node in self.network.nodes.values():
            plt.annotate(node.node_id, (node.x, node.y))

    def plot_sink_connections(self):
        for node in self.network.nodes.values():
            if node.node_id == 1:
                plt.plot([node.x, self.network.nodes[1].x], [
                         node.y, self.network.nodes[1].y], 'k-', linewidth=1)

    def plot_metrics(self, network_energy, network_energy_label, network_energy_unit,
                     network_energy_title, num_dead_nodes, num_dead_nodes_label,
                     num_dead_nodes_title,
                        num_alive_nodes, num_alive_nodes_label, num_alive_nodes_title):
        plt.figure()
        plt.plot(network_energy.keys(), network_energy.values())
        plt.xlabel("Round")
        plt.ylabel(f"{network_energy_label} ({network_energy_unit})")
        plt.title(network_energy_title)
        plt.show()

        plt.figure()
        plt.plot(num_dead_nodes.keys(), num_dead_nodes.values())
        plt.xlabel("Round")
        plt.ylabel(num_dead_nodes_label)
        plt.title(num_dead_nodes_title)
        plt.show()

        plt.figure()
        plt.plot(num_alive_nodes.keys(), num_alive_nodes.values())
        plt.xlabel("Round")
        plt.ylabel(num_alive_nodes_label)
        plt.title(num_alive_nodes_title)
        plt.show()

    def run(self):
        print("Running LEACH protocol...")
        num_nodes = self.config.network.num_sensor
        p = self.config.network.protocol.cluster_head_percentage

        for node in self.network.nodes.values():
            node.is_cluster_head = False

        # Dictionary to store the network energy vs rounds
        network_energy = {}
        # Dictionary to store the number of dead nodes vs rounds
        num_dead_nodes = {}
        # Dicitonary to store the number of alive nodes vs rounds
        num_alive_nodes = {}

        # print network initial energy
        energy = 0
        for node in self.network.nodes.values():
            energy += node.energy
        self.network.remaining_energy = energy

        round = 0
        while self.network.alive_nodes() > 0:
            round += 1
            print(f"Round {round}")
            num_cluster_heads = 0
            th = p / (1 - p * (round % (1 / p)))
            print(f"Threshold: {th}")
            tleft = round % (1 / p)
            self.select_cluster_heads(th, tleft, num_cluster_heads)
            # input("Press enter to continue...")
            chs_bool = self.create_clusters()
            self.energy_dissipation_non_cluster_heads(round)
            self.energy_dissipation_cluster_heads(round)
            # self.plot_clusters(round)
            # Store network energy vs rounds in a dictionary
            network_energy[round] = self.network.remaining_energy
            # Store number of dead nodes vs rounds in a dictionary
            num_dead_nodes[round] = num_nodes - self.network.alive_nodes()
            # Store number of alive nodes vs rounds in a dictionary
            num_alive_nodes[round] = self.network.alive_nodes()
        # Plot the network energy vs rounds
        self.plot_metrics(network_energy, "Network Energy", "J",
                          "Network Energy vs Rounds",
                          num_dead_nodes, "Number of Dead Nodes",
                          "Number of Dead Nodes vs Rounds",
                          num_alive_nodes, "Number of Alive Nodes",
                            "Number of Alive Nodes vs Rounds")
