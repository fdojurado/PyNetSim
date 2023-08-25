from pynetsim.node.node import Node

import networkx as nx
import matplotlib.pyplot as plt
import random


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
        self.nodes = {}

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
        for node in self.nodes.values():
            if node.node_id != 1:
                not_visited.add(node)
        # Print the visted and not_visited sets
        print("Visited: ", [node.node_id for node in visited])
        print("Not visited: ", [node.node_id for node in not_visited])
        # Start the BFS
        while len(queue) > 0:
            # Get the first node in the queue
            node = queue.pop(0)
            print("Node: ", node.node_id)
            # Get the neighbors of the node
            neighbors = node.get_neighbors()
            print("Neighbors: ", [neighbor.node_id for neighbor in neighbors])
            # For each neighbor, if it is not visited
            # then add it to the queue and the visited set
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    not_visited.remove(neighbor)
        # Print the visted and not_visited sets
        print("BFS Visited: ", [node.node_id for node in visited])
        print("BFS Not visited: ", [node.node_id for node in not_visited])
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
                x = random.uniform(0, self.width)
                y = random.uniform(0, self.height)
                node = Node(i, x, y)
                self.nodes[i] = node  # node_id: node
            # Set the sink node
            self.nodes[1].set_sink()

            # Calculate the neighbors for each node
            for node in self.nodes.values():
                for other_node in self.nodes.values():
                    if node.node_id != other_node.node_id and node.is_within_range(other_node, self.transmission_range):
                        node.add_neighbor(other_node)
        else:
            # If there are nodes in the config file, then create the nodes from the config file
            for node in self.config.network.nodes:
                self.nodes[node.node_id] = Node(
                    node.node_id, node.x, node.y, node.type_node)

            # Calculate the neighbors for each node
            for node in self.nodes.values():
                for other_node in self.nodes.values():
                    if node.node_id != other_node.node_id and node.is_within_range(other_node, self.transmission_range):
                        node.add_neighbor(other_node)

        return self

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_nodes(self):
        return self.nodes.values()

    def get_num_nodes(self):
        return len(self.nodes)

    def get_last_node_id(self):
        return self.num_nodes

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
        return True

    def plot_network(self):
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'bo')
            for neighbor in node.neighbors.values():
                # Create a dashed line between the node and its neighbor, light grey color
                plt.plot([node.x, neighbor.x], [
                         node.y, neighbor.y], 'k--', linewidth=0.5)
        # Annotate the Node IDs
        for node in self.nodes.values():
            plt.annotate(node.node_id, (node.x, node.y))
        # print the sink node in red
        for node in self.nodes.values():
            if node.type == "Sink":
                plt.plot(node.x, node.y, 'ro')
                plt.annotate(node.node_id, (node.x, node.y))
        plt.title("Network")
        plt.show()

    def start(self):
        print("Starting network...")
        # Create a graph
        graph = nx.Graph()
        # Add the nodes to the graph
        for node in self.nodes.values():
            graph.add_node(node.node_id)
        # Add the edges to the graph
        for node in self.nodes.values():
            for neighbor in node.neighbors.values():
                weight = ((node.x - neighbor.x)**2 +
                          (node.y - neighbor.y)**2)**0.5
                graph.add_edge(node.node_id, neighbor.node_id,
                               weight=weight, label=weight)

        # Compute the shortest path from each node to the sink node
        shortest_paths = nx.single_source_dijkstra_path(graph, 1)
        print("Shortest paths: ", shortest_paths)

        # Plot the network
        self.plot_network()
