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

from rich.console import Console
from rich.table import Table
from pynetsim.statistics.stats import Statistics
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
        self.__is_main_cluster_head = False
        self.cluster_id = 0
        self.mch_id = 0
        self.rounds_to_become_cluster_head = 0
        self.__remaining_energy = energy
        self.__initial_energy = energy
        self.__drain_rate = 0
        self.__energy_dissipated = 0
        self.__pkts_sent_to_bs = 0
        self.dst_to_sink = 0
        self.dst_to_cluster_head = 0
        self.dst_to_mch = 0
        self.round_dead = 0
        self.__pkt_sent = 0
        self.__pkt_received = 0
        self.__energy_control_packets = 0
        self.__control_pkt_bits = 0

    @property
    def is_cluster_head(self):
        return self.__is_cluster_head

    @is_cluster_head.setter
    def is_cluster_head(self, value: bool):
        assert isinstance(value, bool)
        self.__is_cluster_head = value

    @property
    def is_main_cluster_head(self):
        return self.__is_main_cluster_head

    @is_main_cluster_head.setter
    def is_main_cluster_head(self, value: bool):
        assert isinstance(value, bool)
        self.__is_main_cluster_head = value

    @property
    def remaining_energy(self):
        return self.__remaining_energy

    @property
    def initial_energy(self):
        return self.__initial_energy

    @property
    def drain_rate(self):
        return self.__drain_rate

    @drain_rate.setter
    def drain_rate(self, value: float):
        self.__drain_rate = value

    @remaining_energy.setter
    def remaining_energy(self, value: float):
        # This is only used for the RL based algorithms
        self.__remaining_energy = value

    @property
    def energy_dissipated(self):
        return self.__energy_dissipated

    @property
    def energy_control_packets(self):
        return self.__energy_control_packets

    @property
    def control_pkt_bits(self):
        return self.__control_pkt_bits

    @property
    def pkts_sent_to_bs(self):
        return self.__pkts_sent_to_bs

    @property
    def pkt_sent(self):
        return self.__pkt_sent

    @property
    def pkt_received(self):
        return self.__pkt_received

    def energy_dissipation(self, energy: float, round: int):
        # update the drain rate
        self.drain_rate = energy
        # If remaining energy is empty, add the initial energy
        self.__remaining_energy -= energy
        # if round not in self.energy_dissipated added otherwise sum
        self.__energy_dissipated += energy

    def energy_dissipation_control_packets(self, energy: float, bits: float):
        self.__energy_control_packets += energy
        self.__control_pkt_bits += bits

    def inc_pkts_sent_to_bs(self):
        self.__pkts_sent_to_bs += 1

    def inc_pkts_sent(self):
        self.__pkt_sent += 1

    def inc_pkts_received(self):
        self.__pkt_received += 1

    def pdr(self):
        if self.pkt_sent == 0:
            return 0
        return self.pkt_received / self.pkt_sent

    def plr(self):
        if self.pkt_sent == 0:
            return 0
        return 1 - self.pdr()

    def clear_stats(self):
        self.__pkt_sent = 0
        self.__pkt_received = 0
        self.__energy_dissipated = 0
        self.__energy_control_packets = 0
        self.__control_pkt_bits = 0
        self.__pkts_sent_to_bs = 0

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

    def print_routing_table(self, rich=False):
        print("Routing table for node %d" % self.node_id)
        if not rich:
            for destination, next_hop in self.routing_table.items():
                print("Destination: %d, Next hop: %d" %
                      (destination, next_hop))
            return
        # print tich table
        table = Table(title="Routing table for node %d" % self.node_id)
        table.add_column("Destination", justify="right", style="cyan")
        table.add_column("Next hop", justify="right", style="magenta")
        for destination, next_hop in self.routing_table.items():
            table.add_row(str(destination), str(next_hop))
        console = Console()
        console.print(table)

    # -----------------TSCH operations-----------------

    def add_tsch_entry(self, ts, channel, cell_type, dst_id=None):
        self.tsch_schedule[ts] = TSCHSchedule(
            cell_type=cell_type, dst_id=dst_id, ch=channel, ts=ts)

    def get_tsch_entry(self, ts):
        return self.tsch_schedule[ts]

    def get_tsch_schedule(self):
        return self.tsch_schedule
