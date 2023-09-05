# Create an abstract class for network models
import matplotlib.pyplot as plt
import json

from abc import ABC, abstractmethod

NANO = 1e-9
PICO = 1e-12


class NetworkModel(ABC):
    def __init__(self, name="default", config=object, network=object):
        self.__name = name
        self.config = config
        self.network = network
        self.get_energy_conversion_factors()
        pass

    @property
    def name(self):
        return self.__name

    def get_energy_conversion_factors(self):
        self.elect = self.config.network.protocol.eelect_nano * NANO
        self.etx = self.config.network.protocol.etx_nano * NANO
        self.erx = self.config.network.protocol.erx_nano * NANO
        self.eamp = self.config.network.protocol.eamp_pico * PICO
        self.eda = self.config.network.protocol.eda_nano * NANO
        self.packet_size = self.config.network.protocol.packet_size

    def energy_dissipation_non_cluster_heads(self, round: int):
        for node in self.network:
            if self.network.should_skip_node(node) or node.is_cluster_head:
                continue

            cluster_head = self.network.get_cluster_head(node)

            if cluster_head is None:
                self.transfer_data_to_sink(node)
            else:
                self.energy_dissipation_non_cluster_head(
                    node=node, cluster_head=cluster_head, round=round)

    @abstractmethod
    def calculate_energy_tx_non_ch(self, distance: float):
        pass

    @abstractmethod
    def calculate_energy_tx_ch(self, distance: float):
        pass

    @abstractmethod
    def calculate_energy_rx(self):
        pass

    def energy_dissipated(self, node: object, energy: float):
        node.energy -= energy

    def energy_dissipation_non_cluster_head(self, node: object, cluster_head: object, round: int):
        if not self.network.alive(node):
            return
        distance = node.dst_to_cluster_head
        ETx = self.calculate_energy_tx_non_ch(distance=distance)
        self.energy_dissipated(node=node, energy=ETx)
        node.increase_packet_sent()
        if not self.network.alive(cluster_head):
            return
        node.increase_packet_received()
        ERx = self.calculate_energy_rx()
        self.energy_dissipated(node=cluster_head, energy=ERx)
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
            distance = node.dst_to_sink
            ETx = self.calculate_energy_tx_ch(distance=distance)
            self.energy_dissipated(node=node, energy=ETx)
            node.increase_packet_sent()
            node.increase_packet_received()
            if not self.network.alive(node):
                self.network.mark_node_as_dead(node, round)
                self.network.remove_node_from_cluster(node)

    def transfer_data_to_sink(self, node):
        if not self.network.alive(node):
            return
        distance = node.dst_to_sink
        # print("No cluster heads, transferring data to the sink.")
        # print(f"Node {node.node_id} distance to sink: {distance}")
        ETx = self.elect * self.packet_size + self.eamp * self.packet_size * distance**2
        node.energy -= ETx
        # network.remaining_energy -= ETx
        if not self.network.alive(node):
            self.network.mark_node_as_dead(node, round)

    def dissipate_energy(self, round: int):
        self.energy_dissipation_non_cluster_heads(round=round)
        self.energy_dissipation_cluster_heads(round=round)
