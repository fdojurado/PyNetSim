from pynetsim.node.neighbors import Neighbors


class Node:
    def __init__(self, node_id: int, x: float, y: float, type_node: str = "Sensor"):
        self.node_id = node_id
        self.x = x
        self.y = y
        # Type of node, e.g. "sink", "sensor"
        self.type = type_node
        self.neighbors = {}

    def set_sink(self):
        self.type = "Sink"

    def add_neighbor(self, neighbor):
        self.neighbors[neighbor.node_id] = neighbor

    def get_neighbors(self):
        return self.neighbors.values()

    def get_num_neighbors(self):
        return len(self.neighbors)

    def is_within_range(self, other_node, transmission_range):
        distance = ((self.x - other_node.x)**2 +
                    (self.y - other_node.y)**2)**0.5
        return distance <= transmission_range
