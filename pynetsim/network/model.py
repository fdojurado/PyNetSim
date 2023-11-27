# Create an abstract class for network models
from abc import ABC, abstractmethod
from typing import Optional, Callable


class NetworkModel(ABC):
    def __init__(self, name="default", config=object, network=object):
        self.__name = name
        self.config = config
        self.network = network
        self.round_callback: Optional[Callable] = None
        self.get_energy_conversion_factors()

    @property
    def name(self):
        return self.__name

    def register_round_complete_callback(self, callback):
        self.round_callback = callback

    def set_network(self, network):
        self.network = network

    def get_energy_conversion_factors(self):
        self.eelect = self.config.network.protocol.eelect
        self.eamp = self.config.network.protocol.eamp
        self.efs = self.config.network.protocol.efs
        self.eda = self.config.network.protocol.eda
        self.d_0 = (self.efs/self.eamp)**0.5
        self.packet_size = self.config.network.protocol.packet_size
        self.eelect_ps = self.eelect * self.packet_size
        self.eelect_eda_ps = (self.eelect+self.eda)*self.packet_size

    def energy_dissipation_non_cluster_heads(self, round: int):
        for node in self.network:
            if self.network.should_skip_node(node) or node.is_cluster_head:
                continue

            cluster_head = self.network.get_cluster_head(node)

            if cluster_head is None:
                self.transfer_data_to_sink(node=node, round=round)
            else:
                self.energy_dissipation_non_cluster_head(
                    node=node, cluster_head=cluster_head, round=round)

    @abstractmethod
    def calculate_energy_tx_non_ch(self, src: int, dst: int):
        pass

    @abstractmethod
    def calculate_energy_tx_ch(self, src: int):
        pass

    @abstractmethod
    def energy_dissipation_control_packets(self, round: int):
        pass

    @abstractmethod
    def calculate_energy_rx(self):
        pass

    def energy_dissipated(self, node: object, energy: float, round: int):
        node.energy_dissipation(energy=energy, round=round)

    def energy_dissipation_non_cluster_head(self, node: object, cluster_head: object, round: int):
        ETx = self.calculate_energy_tx_non_ch(
            src=node.node_id, dst=cluster_head.node_id)
        self.energy_dissipated(node=node, energy=ETx, round=round)
        node.inc_pkts_sent()
        if not self.network.alive(cluster_head):
            return
        node.inc_pkts_received()
        ERx = self.calculate_energy_rx()
        self.energy_dissipated(node=cluster_head, energy=ERx, round=round)
        if not self.network.alive(cluster_head):
            # print(f"Cluster head {cluster_head.node_id} is dead.")
            self.network.mark_node_as_dead(cluster_head, round)
            # self.network.remove_cluster_head(cluster_head=cluster_head)
            self.network.remove_node_from_cluster(cluster_head)
        if not self.network.alive(node):
            # print(f"Node {node.node_id} is dead.")
            self.network.mark_node_as_dead(node, round)
            self.network.remove_node_from_cluster(node)

    def energy_dissipation_cluster_heads(self, round: int):
        for node in self.network:
            if self.network.should_skip_node(node) or not node.is_cluster_head:
                continue
            ETx = self.calculate_energy_tx_ch(src=node.node_id)
            self.energy_dissipated(node=node, energy=ETx, round=round)
            node.inc_pkts_sent_to_bs()
            node.inc_pkts_sent()
            # Get the sink node
            sink = self.network.get_node(1)
            sink.inc_pkts_received()
            node.inc_pkts_received()
            if not self.network.alive(node):
                self.network.mark_node_as_dead(node, round)
                self.network.remove_node_from_cluster(node)

    def transfer_data_to_sink(self, node, round: int):
        if not self.network.alive(node):
            return
        # print("No cluster heads, transferring data to the sink.")
        # print(f"Node {node.node_id} distance to sink: {distance}")
        ETx = self.calculate_energy_tx_non_ch(src=node.node_id, dst=1)
        self.energy_dissipated(node=node, energy=ETx, round=round)
        node.inc_pkts_sent()
        # node.inc_pkts_sent_to_bs()
        # network.remaining_energy -= ETx
        if not self.network.alive(node):
            self.network.mark_node_as_dead(node, round)

    def dissipate_energy(self, round: int):
        self.energy_dissipation_non_cluster_heads(round=round)
        self.energy_dissipation_cluster_heads(round=round)
        self.energy_dissipation_control_packets(round=round)
        # Fire the callback if it exists
        if self.round_callback is not None:
            self.round_callback(round=round)
