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


import pynetsim.leach.leach_milp as leach_milp
import sys
import os
import random
import logging

from pynetsim.leach.leach_milp.leach_ce_e import LEACH_CE_E
from pynetsim.network.network import Network
from pynetsim.config import load_config, NETWORK_MODELS

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SELF_PATH, "config.yml")

# -------------------- Create logger --------------------
logger = logging.getLogger("Main")


def main():
    # Load config
    config = load_config(CONFIG_FILE)
    logger.info(f"Loading config from {CONFIG_FILE}")

    # Define the range of numbers
    alpha_min = 6
    alpha_max = 13
    beta_min = 0
    beta_max = 4
    gamma_min = 4
    gamma_max = 13

    # Define the number of combinations you want
    num_combinations = 100

    # Generate random combinations
    combinations = []

    for _ in range(num_combinations):
        alpha = random.uniform(alpha_min, alpha_max)
        beta = random.uniform(beta_min, beta_max)
        gamma = random.uniform(gamma_min, gamma_max)
        combination = [alpha, beta, gamma]
        # combination = [random.uniform(min_value, max_value) for _ in range(3)]
        combinations.append(combination)

    print(f"Number of combinations: {len(combinations)}")

    for combo in combinations:
        print(f"Combo: {combo}")
        network = Network(config=config)
        network_model = NETWORK_MODELS[config.network.model](
            config=config, network=network)
        network.set_model(network_model)
        # Generate a random initial energy for each node.
        # for node in config.network.nodes:
        #     random_energy = random.uniform(0, 0.1)
        #     node.energy = random_energy
        # Lets create the object of the LEACH-CE-E protocol.
        leach_ce_e = LEACH_CE_E(network, network_model,
                                alpha=combo[0], beta=combo[1], gamma=combo[2])
        network.initialize()
        # network.stats.name = f"LEACH-CE-E_{combo[0]}_{combo[1]}_{combo[2]}"
        # Lets run the protocol.
        leach_ce_e.run()


if __name__ == "__main__":
    main()
    sys.exit(0)
