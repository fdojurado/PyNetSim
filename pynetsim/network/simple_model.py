from pynetsim.network.model import NetworkModel


class Simple(NetworkModel):

    def __init__(self, config, network):
        super().__init__(name="Simple", config=config, network=network)

    def calculate_energy_tx_non_ch(self, distance: float):
        return self.elect * self.packet_size + self.eamp * self.packet_size * distance**2

    def calculate_energy_tx_ch(self, distance: float):
        return (self.elect+self.eda)*self.packet_size + self.eamp * self.packet_size * distance**2

    def energy_dissipation_control_packets(self, round: int):
        pass

    def calculate_energy_rx(self):
        return (self.elect + self.eda) * self.packet_size
