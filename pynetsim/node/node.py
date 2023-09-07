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
        self.energy_dissipated = {}
        self.pkts_sent_to_bs = 0
        self.dst_to_sink = 0
        self.dst_to_cluster_head = 0
        self.round_dead = 0
        self.packet_sent = 0
        self.packet_received = 0
        self.__energy_control_packets = {}
        self.__control_packet_bits = {}

    @property
    def is_cluster_head(self):
        return self.__is_cluster_head

    @is_cluster_head.setter
    def is_cluster_head(self, value: bool):
        assert isinstance(value, bool)
        self.__is_cluster_head = value

    def energy_dissipation(self, energy: float, round: int):
        self.energy -= energy
        # if round not in self.energy_dissipated added otherwise sum
        if round not in self.energy_dissipated:
            self.energy_dissipated[round] = energy
        else:
            self.energy_dissipated[round] += energy

    def increase_packets_sent_to_bs(self):
        self.pkts_sent_to_bs += 1

    def increase_packet_sent(self):
        self.packet_sent += 1

    def increase_packet_received(self):
        self.packet_received += 1

    def packet_delivery_ratio(self):
        if self.packet_sent == 0:
            return 0
        return self.packet_received / self.packet_sent

    def packet_loss_ratio(self):
        if self.packet_sent == 0:
            return 0
        return 1 - self.packet_delivery_ratio()

    def add_control_packet_bits(self, round: int, bits: int):
        self.__control_packet_bits[round] = bits

    def get_control_packet_bits(self, round: int):
        return self.__control_packet_bits[round]

    def get_last_round_control_packet_bits(self):
        # if the key is empty, return 0
        if not self.__control_packet_bits:
            return 0
        return self.get_control_packet_bits(max(self.__control_packet_bits.keys()))

    def get_last_round_energy_dissipated(self):
        # if the key is empty, return 0
        if not self.energy_dissipated:
            return 0
        return self.energy_dissipated[max(self.energy_dissipated.keys())]

    def clear_control_packet_bits(self):
        self.__control_packet_bits = {}

    def clear_energy_dissipated(self):
        self.energy_dissipated = {}

    def add_energy_control_packet(self, round: int, energy: float):
        self.__energy_control_packets[round] = energy

    def get_energy_control_packet(self, round: int):
        return self.__energy_control_packets[round]

    def get_last_round_energy_control_packet(self):
        if not self.__energy_control_packets:
            return 0
        return self.get_energy_control_packet(max(self.__energy_control_packets.keys()))

    def clear_energy_control_packet(self):
        self.__energy_control_packets = {}
    # @property
    # def dst_to_sink(self):
    #     if self.__dst_to_sink == 0:
    #         self.__dst_to_sink = ((self.x - self.neighbors[1].x)**2 +
    #                               (self.y - self.neighbors[1].y)**2)**0.5
    #     return self.__dst_to_sink

    # @dst_to_sink.setter
    # def dst_to_sink(self, value: float):
    #     assert isinstance(value, float)
    #     self.__dst_to_sink = value

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
