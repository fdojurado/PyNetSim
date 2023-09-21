from pynetsim.network.model import NetworkModel


class Extended(NetworkModel):

    def __init__(self, config, network):
        super().__init__(name="Extended", config=config, network=network)

    def select_eamp(self, distance: float):
        eamp = 0
        if distance <= self.d_0:
            eamp = self.packet_size * self.efs * distance**2
        else:
            eamp = self.packet_size * self.eamp * distance**4
        return eamp

    def calculate_energy_tx_non_ch(self, distance: float):
        return self.elect * self.packet_size + self.select_eamp(distance=distance)

    def calculate_energy_tx_ch(self, distance: float):
        return (self.elect+self.eda)*self.packet_size + self.select_eamp(distance=distance)

    def energy_dissipation_control_packets(self, round: int):
        # # This is only processed by centralized algorithms
        if self.config.network.protocol.name == "LEACH":
            return
        if round-1 <= 0:
            prev_chs = []
        else:
            prev_chs = self.network.get_cluster_head_ids_at_round(
                round=round-1)
        curr_chs = self.network.get_cluster_head_ids()
        # How many different cluster heads from the previous round are there?
        diff = len(set(curr_chs) - set(prev_chs))
        # print(f"Current cluster heads: {curr_chs}")
        # print(f"Previous cluster heads: {prev_chs}")
        # print(f"Diff: {diff}")
        # if there are no new cluster heads, then there is no need to transmit
        # a control packet
        if diff == 0:
            # print node_ids of cluster heads
            # print("No new cluster heads")
            return
        # chs = self.network.num_cluster_heads()
        # pkt_size = (4*chs+15) * 8
        pkt_size = 400
        # Reduce the energy of all nodes by the energy required to transmit the
        # control packet
        energy = self.elect * pkt_size
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            self.energy_dissipated(node=node, energy=energy, round=round)
            # node.energy_control_packets(energy=energy)
            node.energy_dissipation_control_packets(
                energy=energy, bits=pkt_size)
            if not self.network.alive(node):
                self.network.mark_node_as_dead(node, round)
            #     self.network.remove_node_from_cluster(node)

    def calculate_energy_rx(self):
        return self.elect * self.packet_size
