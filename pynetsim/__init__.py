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

import logging

from pynetsim.network.network import Network
from pynetsim.config import load_config, print_rich_table_config, NETWORK_MODELS

# -------------------- Create logger --------------------
logger = logging.getLogger("Main")


class PyNetSim:
    def __init__(self,
                 config=None,
                 print_rich=False):
        self.config = load_config(config)
        self.network = Network(config=self.config)
        self.net_model = NETWORK_MODELS[self.config.network.model](
            config=self.config, network=self.network)
        self.network.set_model(self.net_model)
        self.network.initialize()
        self.net_model.init()
        # print the configurations
        logger.debug("PyNetSim configuration:\n%s", self.config)
        if print_rich:
            print_rich_table_config(self.config, self.network)

    def run(self):
        self.network.start()
