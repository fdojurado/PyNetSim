

class LEACH_HRL:

    def __init__(self, network):
        self.name = "LEACH-HRL"
        self.config = network.config
        self.network = network
        self.elect = self.config.network.protocol.eelect_nano * 10**-9
        self.etx = self.config.network.protocol.etx_nano * 10**-9
        self.erx = self.config.network.protocol.erx_nano * 10**-9
        self.eamp = self.config.network.protocol.eamp_pico * 10**-12
        self.eda = self.config.network.protocol.eda_nano * 10**-9
        self.packet_size = self.config.network.protocol.packet_size

    def run(self):
        print(f"Running {self.name} protocol...")

        for node in self.network.nodes.values():
            node.is_cluster_head = False

        network_energy = {}
        num_dead_nodes = {}
        num_alive_nodes = {}

        # Set all dst_to_sink for all nodes
        for node in self.network.nodes.values():
            node.dst_to_sink = self.network.distance_to_sink(node)
