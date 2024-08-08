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
