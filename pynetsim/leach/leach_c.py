

class LEACH_C:

    def __init__(self, network):
        self.name = "LEACH_C"
        self.config = network.config
        self.network = network
        self.elect = self.config.network.protocol.eelect_nano * 10**-9
        self.etx = self.config.network.protocol.etx_nano * 10**-9
        self.erx = self.config.network.protocol.erx_nano * 10**-9
        self.eamp = self.config.netowrk.protocol.eamp_pico * 10**-12
        self.eda = self.config.network.protocol.eda_nano * 10**-9
        self.packet_size = self.config.network.protocol.packet_size
