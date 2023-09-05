from pynetsim.network.model import NetworkModel


class Extended(NetworkModel):

    def __init__(self, config, network):
        self.dst_th = config.network.protocol.dst_threshold
        super().__init__(name="Extended", config=config, network=network)

    def select_eamp(self, distance: float):
        eamp = 0
        if distance < self.dst_th:
            eamp = self.packet_size * self.efs * distance**2
        else:
            eamp = self.packet_size * self.eamp * distance**4
        return eamp

    def calculate_energy_tx_non_ch(self, distance: float):
        return self.elect * self.packet_size + self.select_eamp(distance=distance)

    def calculate_energy_tx_ch(self, distance: float):
        return (self.elect+self.eda)*self.packet_size + self.select_eamp(distance=distance)

    def calculate_energy_control_packets(self):
        pass

    def calculate_energy_rx(self):
        return self.elect * self.packet_size
