from pynetsim.network.model import NetworkModel


class Extended(NetworkModel):

    def __init__(self, config, network):
        super().__init__(name="Extended", config=config, network=network)

    def init(self):
        # Calculate the eamp from every node to
        self.eamp_matrix = self.calculate_eamp_matrix()
        # control packet transmission energy
        self.ctrl_pkt_energy = self.eelect * 1000

    def calculate_eamp_matrix(self):
        eamp_matrix = {}
        # print self.network
        for node in self.network:
            eamp_matrix[node.node_id] = {}
            for node2 in self.network:
                eamp_matrix[node.node_id][node2.node_id] = self.select_eamp(
                    distance=self.network.distance_between_nodes(node, node2))
        return eamp_matrix

    def select_eamp(self, distance: float):
        eamp = 0
        if distance <= self.d_0:
            eamp = self.packet_size * self.efs * distance**2
        else:
            eamp = self.packet_size * self.eamp * distance**4
        return eamp

    def calculate_energy_tx_non_ch(self, src: int, dst: int):
        return self.eelect_ps + self.eamp_matrix[src][dst]

    def calculate_energy_tx_ch(self, src: int):
        node = self.network.get_node(src)
        if node.is_main_cluster_head:
            return self.eelect_eda_ps + self.eamp_matrix[src][1]
        else:
            if node.mch_id != 0:
                dst = self.network.get_node_with_mch_id(node.mch_id)
                return self.eelect_eda_ps + self.eamp_matrix[src][dst.node_id]
            else:
                return self.eelect_eda_ps + self.eamp_matrix[src][1]

    def energy_dissipation_control_packets(self, round: int):
        # print("Energy dissipation control packets")
        # # This is only processed by centralized algorithms
        if self.config.network.protocol.name == "LEACH" or self.config.network.protocol.name == "LEACH-EE" or self.config.network.protocol.name == "LEACH-D":
            print("LEACH-based protocol, skipping")
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
        # print("New cluster heads, transmitting control packet")
        # chs = self.network.num_cluster_heads()
        # pkt_size = (4*chs+15) * 8
        # Reduce the energy of all nodes by the energy required to transmit the
        # control packet
        for node in self.network:
            if self.network.should_skip_node(node):
                continue
            self.energy_dissipated(
                node=node, energy=self.ctrl_pkt_energy, round=round)
            # node.energy_control_packets(energy=energy)
            node.energy_dissipation_control_packets(
                energy=self.ctrl_pkt_energy, bits=400)
            if not self.network.alive(node):
                self.network.mark_node_as_dead(node, round)
            #     self.network.remove_node_from_cluster(node)

    def calculate_energy_rx(self):
        return self.eelect_ps
