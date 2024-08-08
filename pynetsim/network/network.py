#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

from pynetsim.statistics.stats import Statistics
from pynetsim.config import PROTOCOLS
from pynetsim.node.node import Node

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pynetsim.utils import RandomNumberGenerator


logger = logging.getLogger("Main")


def ensure_connected(func):
    def wrapper(*args, **kwargs):
        network = func(*args, **kwargs)

        # Check if the network is connected
        if not network.is_connected():
            raise ValueError("The network is not connected.")

        return network

    return wrapper


class Network:
    def __init__(self, config):
        self.num_nodes = config.network.num_sensor
        self.transmission_range = config.network.transmission_range
        self.width = config.network.width
        self.height = config.network.height
        self.config = config
        self.stats = Statistics(self, config)
        self.nodes = {}
        self.cluster_heads_per_round = {}
        self.rng = RandomNumberGenerator(self.config)

    def set_model(self, model):
        self.model = model

    # -----------------LEACH-----------------

    def is_centralized(self):
        return self.model.is_centralized()

    def round_callback(self, round: int):
        chs = [
            cluster_head.node_id for cluster_head in self if cluster_head.is_cluster_head]
        # Order in ascending order
        chs.sort()
        # input(f"Cluster heads at round {round}: {chs}")
        self.cluster_heads_per_round[round] = chs
        self.stats.generate_round_stats(round=round)

    def export_stats(self):
        self.stats.export_json()

    def set_stats_name(self, name):
        self.stats.name = name

    def get_cluster_head_ids_at_round(self, round: int):
        return self.cluster_heads_per_round[round]

    def max_distance_to_sink(self):
        max_distance = 0
        for node in self:
            if node.node_id == 1:
                continue
            distance = self.distance_to_sink(node)
            if distance > max_distance:
                max_distance = distance
        return max_distance

    def mark_as_cluster_head(self, node: Node, cluster_id: int):
        node.is_cluster_head = True
        node.cluster_id = cluster_id

    def mark_as_non_cluster_head(self, node):
        node.is_cluster_head = False
        node.cluster_id = 0

    def mark_as_main_cluster_head(self, node: Node, cluster_id: int):
        node.is_main_cluster_head = True
        node.mch_id = cluster_id

    def mark_as_non_main_cluster_head(self, node):
        node.is_main_cluster_head = False
        node.mch_id = 0

    def create_clusters(self):
        cluster_heads_exist = any(
            node.is_cluster_head for node in self)
        if not cluster_heads_exist:
            # print("There are no cluster heads.")
            # input("Press Enter to continue...")
            self.clear_clusters()
            return False

        for node in self:
            if not node.is_cluster_head and node.node_id != 1:
                self.add_node_to_cluster(node=node)

        return True

    def print_clusters(self):
        for node in self:
            if not node.is_cluster_head:
                ch = self.get_cluster_head(node)
                if ch:
                    print(
                        f"Node {node.node_id} is a member of cluster head {ch.node_id}")
            else:
                print(f"Node {node.node_id} is a cluster head.")

    def create_mch_clusters(self):
        mch_exist = any(
            node.is_main_cluster_head for node in self)
        # print all cluster head ids
        chs = [node.node_id for node in self if node.is_cluster_head]
        # print(f"create_mch_clusters, cluster heads: {chs}")
        if not mch_exist:
            return False

        for node in self:
            if self.should_skip_node(node):
                continue

            if not node.is_cluster_head:
                continue

            if not node.is_main_cluster_head:
                # print(f"Node {node.node_id} is not a main cluster head.")
                # self.mark_as_non_cluster_head(node)
                self.add_ch_to_mch(node)
                # print(
                #     f"Node {node.node_id} is a member of MCH {node.mch_id} (ID: {self.get_mch(node).node_id})")

        return True

    def add_ch_to_mch(self, node):
        distances = {mch.node_id: ((node.x - mch.x)**2 + (node.y - mch.y)**2)**0.5
                     for mch in self if mch.is_main_cluster_head}
        mch_id = min(distances, key=distances.get)
        min_distance = distances[mch_id]
        mch = self.get_node(mch_id)
        mch.add_neighbor(node)
        node.add_neighbor(mch)
        node.dst_to_mch = min_distance
        node.mch_id = mch.mch_id

    def add_node_to_cluster(self, node):
        distances = {cluster_head.node_id: ((node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)**0.5
                     for cluster_head in self if cluster_head.is_cluster_head}
        cluster_head_id = min(distances, key=distances.get)
        min_distance = distances[cluster_head_id]
        cluster_head = self.get_node(cluster_head_id)
        cluster_head.add_neighbor(node)
        node.add_neighbor(cluster_head)
        node.dst_to_cluster_head = min_distance
        node.cluster_id = cluster_head.cluster_id

    def remove_cluster_head(self, cluster_head):
        cluster_id = cluster_head.cluster_id
        for node in self:
            if node.cluster_id == cluster_id:
                node.cluster_id = 0

    def remove_node_from_cluster(self, node):
        for neighbor in node.neighbors.values():
            # print(f"Removing node {node.node_id} from node {neighbor.node_id}")
            # if the node is not dead, remove it from the neighbor's neighbors
            if neighbor.remaining_energy > 0:
                if node.node_id in neighbor.neighbors:
                    neighbor.neighbors.pop(node.node_id)
        node.neighbors = {}

    def get_cluster_head(self, node: Node):
        return self.get_node_with_cluster_id(node.cluster_id)

    def get_mch(self, node: Node):
        return self.get_node_with_mch_id(node.mch_id)

    def clear_clusters(self):
        for node in self:
            node.cluster_id = 0

    def get_node_with_cluster_id(self, cluster_id):
        for node in self:
            if node.cluster_id == cluster_id and node.is_cluster_head:
                return node
        return None

    def get_node_with_mch_id(self, mch_id) -> Node:
        for node in self:
            if node.mch_id == mch_id and node.is_main_cluster_head:
                return node
        return None

    # Get how many nodes are in the same mch id
    def get_num_nodes_in_mch(self, mch_id):
        num_nodes = 0
        for node in self:
            if node.mch_id == mch_id:
                num_nodes += 1
        return num_nodes

    def should_skip_node(self, node):
        return node.node_id == 1 or not self.alive(node)

    def alive(self, node: Node):
        return node.remaining_energy > 0

    def mark_node_as_dead(self, node, round):
        logger.info("Node %s is dead.", node.node_id)
        node.round_dead = round

    def alive_nodes(self):
        alive_nodes = 0
        for node in self:
            # exclude the sink node
            if self.should_skip_node(node):
                continue
            alive_nodes += 1
        return alive_nodes

    def dead_nodes(self):
        dead_nodes = 0
        for node in self:
            if not self.alive(node):
                dead_nodes += 1
        return dead_nodes

    def remaining_energy(self):
        remaining_energy = 0
        for node in self:
            if self.should_skip_node(node):
                continue
            remaining_energy += node.remaining_energy
        return remaining_energy

    def average_remaining_energy(self):
        alive_nodes = self.alive_nodes()
        if alive_nodes == 0:
            return 0
        return self.remaining_energy() / alive_nodes

    def average_drain_rate(self):
        drain_rate = 0
        for node in self:
            if self.should_skip_node(node):
                continue
            drain_rate += node.drain_rate
        return drain_rate / self.alive_nodes()

    def average_pdr(self):
        pdr = 0
        # CHeck if there are alive nodes
        alive_nodes = self.alive_nodes()
        if alive_nodes == 0:
            return 0
        for node in self:
            if self.should_skip_node(node):
                continue
            pdr += node.pdr()
        return pdr / alive_nodes

    def average_plr(self):
        plr = 0
        # CHeck if there are alive nodes
        alive_nodes = self.alive_nodes()
        if alive_nodes == 0:
            return 0
        for node in self:
            if self.should_skip_node(node):
                continue
            plr += node.plr()
        return plr / alive_nodes

    def control_packets_energy(self):
        energy = 0
        for node in self:
            if node.node_id == 1:
                continue
            energy += node.energy_control_packets
        return energy

    def control_pkt_bits(self):
        bits = 0
        for node in self:
            if node.node_id == 1:
                continue
            bits += node.control_pkt_bits
        return bits

    def energy_dissipated(self):
        energy_dissipated = 0
        for node in self:
            if node.node_id == 1:
                continue
            energy_dissipated += node.energy_dissipated
        return energy_dissipated

    def pkts_sent_to_bs(self):
        pkts = 0
        for node in self:
            if node.node_id == 1:
                continue
            pkts += node.pkts_sent_to_bs
        return pkts

    def pkts_recv_by_bs(self):
        sink = self.get_node(1)
        return sink.pkt_received

    def get_cluster_head_ids(self):
        cluster_heads = []
        for node in self:
            if node.node_id == 1:
                continue
            if node.is_cluster_head:
                cluster_heads.append(node.node_id)
        return cluster_heads

    def get_cluster_ids(self):
        cluster_ids = []
        for node in self:
            if self.should_skip_node(node):
                continue
            if node.cluster_id not in cluster_ids:
                cluster_ids.append(node.cluster_id)
        return cluster_ids

    def get_nodes_in_cluster(self, cluster_id):
        nodes = []
        for node in self:
            if self.should_skip_node(node):
                continue
            if node.cluster_id == cluster_id:
                nodes.append(node)
        return nodes

    def get_nodes_not_in_cluster(self, cluster_id):
        nodes = []
        for node in self:
            if self.should_skip_node(node):
                continue
            if node.cluster_id != cluster_id:
                nodes.append(node)
        return nodes

    def get_clusters_average_energy(self):
        clusters_average_energy = {}
        for cluster_id in self.get_cluster_ids():
            cluster_nodes = self.get_nodes_in_cluster(cluster_id)
            cluster_energy = 0
            for node in cluster_nodes:
                cluster_energy += node.remaining_energy
            clusters_average_energy[cluster_id] = cluster_energy / \
                len(cluster_nodes)
        return clusters_average_energy

    def get_clusters_variance_energy(self):
        # Calculate the variance of the get_clusters_average_energy values
        clusters_average_energy = self.get_clusters_average_energy()
        values = clusters_average_energy.values()
        return np.var(list(values))

    def num_cluster_heads(self):
        num_cluster_heads = 0
        for node in self:
            if self.should_skip_node(node):
                continue
            if node.is_cluster_head:
                num_cluster_heads += 1
        return num_cluster_heads

    def distance_to_sink(self, node):
        return ((node.x - self.nodes[1].x)**2 + (node.y - self.nodes[1].y)**2)**0.5

    def distance_between_nodes(self, node1: Node, node2: Node):
        return ((node1.x - node2.x)**2 + (node1.y - node2.y)**2)**0.5

    def calculate_energy_tx_non_ch(self, src: int, dst: int):
        return self.model.calculate_energy_tx_non_ch(src=src, dst=dst)

    def calculate_energy_tx_ch(self, src: int):
        return self.model.calculate_energy_tx_ch(src=src)

    def calculate_energy_rx_ch_per_node(self):
        return self.model.calculate_energy_rx()

    # -----------------Network creation-----------------

    def is_connected(self):
        # Check if the network is connected
        # If not, then we need to create more nodes
        # until the network is connected
        # We use a BFS to check if the network is connected
        # We start from the sink node
        # We use a queue to store the nodes that we have visited
        # We use a set to store the nodes that we have visited
        # We use a set to store the nodes that we have not visited
        visited = set()
        not_visited = set()
        queue = []
        # Add the sink node to the queue and the visited set
        queue.append(self.nodes[1])
        visited.add(self.nodes[1])
        # Add all the other nodes to the not_visited set
        for node in self:
            if node.node_id != 1:
                not_visited.add(node)
        # Print the visted and not_visited sets
        # print("Visited: ", [node.node_id for node in visited])
        # print("Not visited: ", [node.node_id for node in not_visited])
        # Start the BFS
        while len(queue) > 0:
            # Get the first node in the queue
            node = queue.pop(0)
            # print("Node: ", node.node_id)
            # Get the neighbors of the node
            neighbors = node.get_neighbors()
            # print("Neighbors: ", [neighbor.node_id for neighbor in neighbors])
            # For each neighbor, if it is not visited
            # then add it to the queue and the visited set
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    not_visited.remove(neighbor)
        # Print the visted and not_visited sets
        # print("BFS Visited: ", [node.node_id for node in visited])
        # print("BFS Not visited: ", [node.node_id for node in not_visited])
        # If the not_visited set is empty, then the network is connected
        return len(not_visited) == 0

    @ensure_connected
    def create_nodes(self):
        # But we need to make sure is a connected graph, meaning that

        # If there are not nodes in the config file, then create the nodes randomly
        # Check if nodes list in [config] is empty
        if len(self.config.network.nodes) == 0:
            print("Creating nodes randomly...")
            # every node has a path to the sink node.
            for i in range(1, self.num_nodes + 1):
                if i == 1:
                    # Create the sink node
                    # x,y locations are fixed at the middle of the network
                    x = self.config.network.width / 2
                    y = self.config.network.height / 2
                    node = Node(i, x, y, energy=2)
                    self.nodes[i] = node
                    continue
                x = self.rng.get_uniform(0, self.width)
                y = self.rng.get_uniform(0, self.height)
                node = Node(
                    i, x, y, energy=self.config.network.protocol.init_energy)
                self.nodes[i] = node  # node_id: node
            # Set the sink node
            self.nodes[1].set_sink()

            # Calculate the neighbors for each node
            for node in self:
                for other_node in self:
                    if node.node_id != other_node.node_id:
                        if not self.config.network.protocol.name == 'LEACH':
                            if node.is_within_range(other_node, self.transmission_range):
                                node.add_neighbor(other_node)
                        else:
                            node.add_neighbor(other_node)
        else:
            # If there are nodes in the config file, then create the nodes from the config file
            for node in self.config.network.nodes:
                self.nodes[node.node_id] = Node(
                    node.node_id, node.x, node.y, node.type_node, node.energy)

            # Calculate the neighbors for each node
            for node in self:
                for other_node in self:
                    if node.node_id != other_node.node_id:
                        # if not (self.config.network.protocol.name == 'LEACH' or self.config.network.protocol.name == 'LEACH-C' or
                        #         self.config.network.protocol.name == 'LEACH-R' or self.config.network.protocol.name == 'LEACH-RL' or
                        #         self.config.network.protocol.name == 'LEACH-RT' or self.config.network.protocol.name == 'LEACH-K' or
                        #         self.config.network.protocol.name == 'LEACH-CE-E'):
                        #     if node.is_within_range(other_node, self.transmission_range):
                        #         node.add_neighbor(other_node)
                        # else:
                        node.add_neighbor(other_node)

        # Set the distance to the sink node for each node
        for node in self:
            node.dst_to_sink = self.distance_to_sink(node)
            self.mark_as_non_cluster_head(node)

        return self

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def rm_node(self, node: Node):
        self.nodes.pop(node.node_id)

    def get_nodes(self):
        return self.nodes.values()

    def get_num_nodes(self):
        return len(self.nodes)

    def get_last_node_id(self):
        return self.num_nodes

    def __iter__(self):
        return iter(self.nodes.values())

    def __next__(self):
        try:
            return next(self.node_iterator)
        except StopIteration:
            raise StopIteration

    # -----------------Network operations-----------------
    def initialize(self):
        print("Initializing network...")
        # If there is no nodes in the config file, then create the nodes
        self.create_nodes()
        # Print a Error message if the network is not connected
        # This is not longer working because of the decorator
        if not self.is_connected():
            print("Error: The network is not connected!")
            self.plot_network()
            return False
        # Set all nodes to not be cluster heads
        for node in self:
            node.is_cluster_head = False
        # Register callback to the network model
        self.model.register_round_complete_callback(self.round_callback)
        # Call the round_callback to initialize the stats at round 0
        self.round_callback(0)
        return True

    def plot_network(self):
        for node in self:
            plt.plot(node.x, node.y, 'bo')
            for neighbor in node.neighbors.values():
                # Create a dashed line between the node and its neighbor, light grey color
                plt.plot([node.x, neighbor.x], [
                         node.y, neighbor.y], 'k--', linewidth=0.5)
        # Annotate the Node IDs
        for node in self:
            plt.annotate(node.node_id, (node.x, node.y))
        # print the sink node in red
        for node in self:
            if node.type == "Sink":
                plt.plot(node.x, node.y, 'ro')
                plt.annotate(node.node_id, (node.x, node.y))
        plt.title("Network")
        plt.show()

    def start(self):
        logger.info("Starting the network...")
        # Create a graph
        graph = nx.Graph()
        # Add the nodes to the graph
        for node in self:
            graph.add_node(node.node_id)
        # Add the edges to the graph
        for node in self:
            for neighbor in node.neighbors.values():
                weight = ((node.x - neighbor.x)**2 +
                          (node.y - neighbor.y)**2)**0.5
                graph.add_edge(node.node_id, neighbor.node_id,
                               weight=weight, label=weight)

        # Compute the shortest path from each node to the sink node
        shortest_paths = nx.single_source_dijkstra_path(graph, 1)
        logger.debug("Shortest paths: %s", shortest_paths)

        # Add the routing entries to each node.
        # The entries are of the form (sink, hop_n, hop_n-1, ..., source)
        for node in self:
            # skip the sink node
            if node.node_id == 1:
                continue
            for destination, path in shortest_paths.items():
                if node.node_id != destination:
                    continue
                if len(path) < 2:
                    continue
                next_hop = path[-2]
                node.add_routing_entry(1, next_hop)

        # Print the routing table for each node
        # for node in self:
        #     node.print_routing_table(rich=True)

        # Plot the network
        # self.plot_network()

        # Run the protocol
        self.run_protocol()

    # -----------------Protocol-----------------

    def run_protocol(self):
        # Run the protocol
        # Get the protocol from the config file
        protocol = self.config.network.protocol.name
        # Get the protocol class from the protocol name
        protocol_class = PROTOCOLS[protocol]
        # Create an instance of the protocol class
        protocol_instance = protocol_class(self, net_model=self.model)
        # Run the protocol
        protocol_instance.run()
