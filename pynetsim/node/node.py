from pynetsim.tsch.schedule import TSCHSchedule


class Node:
    def __init__(self, node_id: int, x: float, y: float, type_node: str = "Sensor",
                 energy: float = 0):
        self.node_id = node_id
        self.x = x
        self.y = y
        # Type of node, e.g. "sink", "sensor"
        self.type = type_node
        self.neighbors = {}
        self.routing_table = {}
        self.tsch_schedule = {}
        # LEACH
        self.__is_cluster_head = False
        self.cluster_id = 0
        self.rounds_to_become_cluster_head = 0
        self.energy = energy
        self.__dst_to_sink = 0
        self.dst_to_cluster_head = 0
        self.round_dead = 0

    @property
    def is_cluster_head(self):
        return self.__is_cluster_head

    @is_cluster_head.setter
    def is_cluster_head(self, value: bool):
        assert isinstance(value, bool)
        self.__is_cluster_head = value

    @property
    def dst_to_sink(self):
        if self.__dst_to_sink == 0:
            self.__dst_to_sink = ((self.x - self.neighbors[1].x)**2 +
                                  (self.y - self.neighbors[1].y)**2)**0.5
        return self.__dst_to_sink

    @dst_to_sink.setter
    def dst_to_sink(self, value: float):
        assert isinstance(value, float)
        self.__dst_to_sink = value

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

    # -----------------Routing operations-----------------
    def add_routing_entry(self, destination, next_hop):
        self.routing_table[destination] = next_hop

    def get_next_hop(self, destination):
        return self.routing_table[destination]

    def get_routing_table(self):
        return self.routing_table

    def print_routing_table(self):
        print("Routing table for node %d" % self.node_id)
        for destination, next_hop in self.routing_table.items():
            print("Destination: %d, Next hop: %d" % (destination, next_hop))

    # -----------------TSCH operations-----------------
    def add_tsch_entry(self, ts, channel, cell_type, dst_id=None):
        self.tsch_schedule[ts] = TSCHSchedule(
            cell_type=cell_type, dst_id=dst_id, ch=channel, ts=ts)

    def get_tsch_entry(self, ts):
        return self.tsch_schedule[ts]

    def get_tsch_schedule(self):
        return self.tsch_schedule
