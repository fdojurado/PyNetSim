

class NodesConfig:

    def __init__(self, node_id=None, x=None, y=None, type_node=None, energy=None):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.type_node = type_node
        self.energy = energy
        self.neighbors = {}

    @classmethod
    def from_json(cls, json_object=None):
        if json_object is None:
            json_object = {}

        if json_object is not None:
            # Loop through the nodes and create a list of nodes
            nodes = []
            for node in json_object:
                nodes.append(cls(node_id=node.get("node_id"),
                                 x=node.get("x"),
                                 y=node.get("y"),
                                 energy=node.get("energy"),
                                 type_node=node.get("type_node")))
            return nodes
